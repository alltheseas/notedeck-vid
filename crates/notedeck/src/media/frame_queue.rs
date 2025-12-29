//! Frame queue for video playback.
//!
//! This module provides a thread-safe ring buffer for decoded video frames,
//! enabling smooth playback by decoupling decoding from rendering.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[cfg(all(feature = "ffmpeg", not(target_os = "android")))]
use super::audio_decoder::AudioDecoder;
use super::video::{VideoDecoderBackend, VideoFrame};

/// Default number of frames to buffer ahead.
const DEFAULT_BUFFER_SIZE: usize = 5;

/// Commands sent to the decode thread.
#[derive(Debug, Clone)]
pub enum DecodeCommand {
    /// Start or resume decoding
    Play,
    /// Pause decoding
    Pause,
    /// Seek to a specific position
    Seek(Duration),
    /// Stop the decode thread
    Stop,
}

/// A thread-safe queue of decoded video frames.
///
/// The FrameQueue manages a ring buffer of decoded frames with a producer
/// (decode thread) that fills the buffer and a consumer (render thread)
/// that takes frames for display.
pub struct FrameQueue {
    /// The decoded frames ready for display
    frames: Arc<Mutex<VecDeque<VideoFrame>>>,
    /// Maximum number of frames to buffer
    capacity: usize,
    /// Condition variable for signaling when frames are available
    frame_available: Arc<Condvar>,
    /// Condition variable for signaling when space is available
    space_available: Arc<Condvar>,
    /// Flag indicating the queue is being flushed (for seeking)
    flushing: Arc<AtomicBool>,
    /// Flag indicating end of stream reached
    eos: Arc<AtomicBool>,
}

impl FrameQueue {
    /// Creates a new frame queue with the specified capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            frames: Arc::new(Mutex::new(VecDeque::with_capacity(capacity))),
            capacity,
            frame_available: Arc::new(Condvar::new()),
            space_available: Arc::new(Condvar::new()),
            flushing: Arc::new(AtomicBool::new(false)),
            eos: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Creates a new frame queue with the default capacity.
    pub fn with_default_capacity() -> Self {
        Self::new(DEFAULT_BUFFER_SIZE)
    }

    /// Pushes a frame onto the queue.
    ///
    /// This will block if the queue is full, unless the queue is being flushed.
    /// Returns false if the queue is being flushed and the frame should be discarded.
    pub fn push(&self, frame: VideoFrame) -> bool {
        let mut frames = self.frames.lock().unwrap();

        // Wait for space if queue is full
        while frames.len() >= self.capacity {
            if self.flushing.load(Ordering::Acquire) {
                return false;
            }
            frames = self.space_available.wait(frames).unwrap();
        }

        // Check again after waiting
        if self.flushing.load(Ordering::Acquire) {
            return false;
        }

        frames.push_back(frame);
        self.frame_available.notify_one();
        true
    }

    /// Pushes a frame without blocking.
    ///
    /// Returns false if the queue is full or being flushed.
    pub fn try_push(&self, frame: VideoFrame) -> bool {
        if self.flushing.load(Ordering::Acquire) {
            return false;
        }

        let mut frames = self.frames.lock().unwrap();
        if frames.len() >= self.capacity {
            return false;
        }

        frames.push_back(frame);
        self.frame_available.notify_one();
        true
    }

    /// Takes the next frame from the queue.
    ///
    /// Returns None if the queue is empty and end-of-stream has been reached.
    pub fn pop(&self) -> Option<VideoFrame> {
        let mut frames = self.frames.lock().unwrap();

        let frame = frames.pop_front();
        if frame.is_some() {
            self.space_available.notify_one();
        }
        frame
    }

    /// Takes the next frame, blocking until one is available.
    ///
    /// Returns None if end-of-stream is reached and the queue is empty.
    pub fn pop_blocking(&self, timeout: Duration) -> Option<VideoFrame> {
        let mut frames = self.frames.lock().unwrap();

        // Wait for a frame if the queue is empty
        if frames.is_empty() {
            if self.eos.load(Ordering::Acquire) {
                return None;
            }

            let (new_frames, timeout_result) =
                self.frame_available.wait_timeout(frames, timeout).unwrap();
            frames = new_frames;

            if timeout_result.timed_out() && frames.is_empty() {
                return None;
            }
        }

        let frame = frames.pop_front();
        if frame.is_some() {
            self.space_available.notify_one();
        }
        frame
    }

    /// Peeks at the next frame without removing it.
    pub fn peek(&self) -> Option<VideoFrame> {
        let frames = self.frames.lock().unwrap();
        frames.front().cloned()
    }

    /// Returns the presentation timestamp of the next frame without removing it.
    pub fn peek_pts(&self) -> Option<Duration> {
        let frames = self.frames.lock().unwrap();
        frames.front().map(|f| f.pts)
    }

    /// Returns the number of frames currently in the queue.
    pub fn len(&self) -> usize {
        self.frames.lock().unwrap().len()
    }

    /// Returns true if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true if the queue is full.
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// Clears all frames from the queue.
    ///
    /// This sets the flushing flag to prevent new frames from being added,
    /// clears the queue, then resets the flushing flag.
    pub fn flush(&self) {
        self.flushing.store(true, Ordering::Release);

        // Wake up any blocked producers
        self.space_available.notify_all();

        {
            let mut frames = self.frames.lock().unwrap();
            frames.clear();
        }

        self.eos.store(false, Ordering::Release);
        self.flushing.store(false, Ordering::Release);
    }

    /// Marks that end-of-stream has been reached.
    pub fn set_eos(&self) {
        self.eos.store(true, Ordering::Release);
        self.frame_available.notify_all();
    }

    /// Returns true if end-of-stream has been reached.
    pub fn is_eos(&self) -> bool {
        self.eos.load(Ordering::Acquire)
    }

    /// Resets the end-of-stream flag.
    pub fn clear_eos(&self) {
        self.eos.store(false, Ordering::Release);
    }
}

impl Default for FrameQueue {
    fn default() -> Self {
        Self::with_default_capacity()
    }
}

/// A video decode thread that fills a frame queue.
///
/// This runs decoding on a separate thread to avoid blocking the render thread.
pub struct DecodeThread {
    /// Handle to the decode thread
    handle: Option<JoinHandle<()>>,
    /// Channel to send commands to the decode thread
    command_tx: crossbeam_channel::Sender<DecodeCommand>,
    /// The frame queue being filled
    frame_queue: Arc<FrameQueue>,
    /// Flag to signal the thread should stop
    stop_flag: Arc<AtomicBool>,
}

impl DecodeThread {
    /// Creates and starts a new decode thread.
    ///
    /// The thread will start in a paused state.
    pub fn new<D: VideoDecoderBackend + Send + 'static>(
        decoder: D,
        frame_queue: Arc<FrameQueue>,
    ) -> Self {
        let (command_tx, command_rx) = crossbeam_channel::unbounded();
        let stop_flag = Arc::new(AtomicBool::new(false));

        let queue = Arc::clone(&frame_queue);
        let stop = Arc::clone(&stop_flag);

        let handle = thread::spawn(move || {
            decode_loop(decoder, queue, command_rx, stop);
        });

        Self {
            handle: Some(handle),
            command_tx,
            frame_queue,
            stop_flag,
        }
    }

    /// Starts or resumes decoding.
    pub fn play(&self) {
        let _ = self.command_tx.send(DecodeCommand::Play);
    }

    /// Pauses decoding.
    pub fn pause(&self) {
        let _ = self.command_tx.send(DecodeCommand::Pause);
    }

    /// Seeks to a specific position.
    ///
    /// This will flush the frame queue and start decoding from the new position.
    pub fn seek(&self, position: Duration) {
        self.frame_queue.flush();
        let _ = self.command_tx.send(DecodeCommand::Seek(position));
    }

    /// Stops the decode thread.
    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Release);
        let _ = self.command_tx.send(DecodeCommand::Stop);
    }

    /// Returns a reference to the frame queue.
    pub fn frame_queue(&self) -> &Arc<FrameQueue> {
        &self.frame_queue
    }
}

impl Drop for DecodeThread {
    fn drop(&mut self) {
        self.stop();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// The main decode loop running on the decode thread.
fn decode_loop<D: VideoDecoderBackend>(
    mut decoder: D,
    frame_queue: Arc<FrameQueue>,
    command_rx: crossbeam_channel::Receiver<DecodeCommand>,
    stop_flag: Arc<AtomicBool>,
) {
    let mut playing = false;

    loop {
        // Check for stop signal
        if stop_flag.load(Ordering::Acquire) {
            break;
        }

        // Process commands (non-blocking)
        while let Ok(cmd) = command_rx.try_recv() {
            match cmd {
                DecodeCommand::Play => {
                    playing = true;
                    frame_queue.clear_eos();
                }
                DecodeCommand::Pause => {
                    playing = false;
                }
                DecodeCommand::Seek(position) => {
                    if let Err(e) = decoder.seek(position) {
                        tracing::error!("Seek failed: {}", e);
                    }
                    frame_queue.clear_eos();
                }
                DecodeCommand::Stop => {
                    return;
                }
            }
        }

        if !playing {
            // Wait for a command when paused
            match command_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(DecodeCommand::Play) => {
                    playing = true;
                    frame_queue.clear_eos();
                }
                Ok(DecodeCommand::Seek(position)) => {
                    if let Err(e) = decoder.seek(position) {
                        tracing::error!("Seek failed: {}", e);
                    }
                }
                Ok(DecodeCommand::Stop) => return,
                Ok(DecodeCommand::Pause) => {}
                Err(_) => continue,
            }
            continue;
        }

        // Don't decode if queue is full
        if frame_queue.is_full() {
            thread::sleep(Duration::from_millis(5));
            continue;
        }

        // Decode the next frame
        match decoder.decode_next() {
            Ok(Some(frame)) => {
                if !frame_queue.push(frame) {
                    // Queue was flushed, likely due to seek
                    continue;
                }
            }
            Ok(None) => {
                // End of stream
                frame_queue.set_eos();
                playing = false;
            }
            Err(e) => {
                tracing::error!("Decode error: {}", e);
                // Continue trying to decode
                thread::sleep(Duration::from_millis(10));
            }
        }
    }
}

// ============================================================================
// Audio decoding thread
// ============================================================================

/// An audio decode thread that decodes audio and sends samples to a channel.
/// The actual audio playback happens on this thread to avoid Send/Sync issues.
#[cfg(all(feature = "ffmpeg", not(target_os = "android")))]
pub struct AudioThread {
    /// Handle to the audio thread
    handle: Option<JoinHandle<()>>,
    /// Channel to send commands to the audio thread
    command_tx: crossbeam_channel::Sender<DecodeCommand>,
    /// Flag to signal the thread should stop
    stop_flag: Arc<AtomicBool>,
    /// Audio handle for volume/mute control (shared with UI)
    audio_handle: super::audio::AudioHandle,
}

#[cfg(all(feature = "ffmpeg", not(target_os = "android")))]
impl AudioThread {
    /// Creates and starts a new audio decode thread.
    pub fn new(url: &str) -> Option<Self> {
        let (command_tx, command_rx) = crossbeam_channel::unbounded();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let audio_handle = super::audio::AudioHandle::new();
        audio_handle.set_available(true);

        let stop = Arc::clone(&stop_flag);
        let handle_clone = audio_handle.clone();
        let url_owned = url.to_string();

        let handle = thread::spawn(move || {
            audio_thread_main(url_owned, handle_clone, command_rx, stop);
        });

        Some(Self {
            handle: Some(handle),
            command_tx,
            stop_flag,
            audio_handle,
        })
    }

    /// Returns the audio handle for UI control.
    pub fn handle(&self) -> super::audio::AudioHandle {
        self.audio_handle.clone()
    }

    /// Starts or resumes audio playback.
    pub fn play(&self) {
        let _ = self.command_tx.send(DecodeCommand::Play);
    }

    /// Pauses audio playback.
    pub fn pause(&self) {
        let _ = self.command_tx.send(DecodeCommand::Pause);
    }

    /// Seeks to a specific position.
    pub fn seek(&self, position: Duration) {
        let _ = self.command_tx.send(DecodeCommand::Seek(position));
    }

    /// Stops the audio thread.
    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Release);
        let _ = self.command_tx.send(DecodeCommand::Stop);
    }
}

#[cfg(all(feature = "ffmpeg", not(target_os = "android")))]
impl Drop for AudioThread {
    fn drop(&mut self) {
        self.stop();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// The main audio thread function - creates player and runs decode loop.
#[cfg(all(feature = "ffmpeg", not(target_os = "android")))]
fn audio_thread_main(
    url: String,
    handle: super::audio::AudioHandle,
    command_rx: crossbeam_channel::Receiver<DecodeCommand>,
    stop_flag: Arc<AtomicBool>,
) {
    use super::audio::{AudioConfig, AudioPlayer};

    // Create audio player on this thread (OutputStream is not Send)
    // Pass the shared handle so mute/volume controls work
    let mut player =
        match AudioPlayer::new_with_handle(AudioConfig::default(), Some(handle.clone())) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!("Failed to create audio player: {}", e);
                handle.set_available(false);
                return;
            }
        };

    // Get device sample rate and create decoder with it
    let device_sample_rate = player.device_sample_rate();
    let mut decoder = match AudioDecoder::new(&url, device_sample_rate) {
        Ok(d) => d,
        Err(e) => {
            tracing::error!("Failed to create audio decoder: {}", e);
            handle.set_available(false);
            return;
        }
    };

    let mut playing = false;

    loop {
        // Check for stop signal
        if stop_flag.load(Ordering::Acquire) {
            break;
        }

        // Process commands (non-blocking)
        while let Ok(cmd) = command_rx.try_recv() {
            match cmd {
                DecodeCommand::Play => {
                    playing = true;
                    player.play();
                }
                DecodeCommand::Pause => {
                    playing = false;
                    player.pause();
                }
                DecodeCommand::Seek(position) => {
                    player.clear();
                    if let Err(e) = decoder.seek(position) {
                        tracing::error!("Audio seek failed: {}", e);
                    }
                }
                DecodeCommand::Stop => {
                    return;
                }
            }
        }

        if !playing {
            // Wait for a command when paused
            match command_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(DecodeCommand::Play) => {
                    playing = true;
                    player.play();
                }
                Ok(DecodeCommand::Seek(position)) => {
                    player.clear();
                    if let Err(e) = decoder.seek(position) {
                        tracing::error!("Audio seek failed: {}", e);
                    }
                }
                Ok(DecodeCommand::Stop) => return,
                Ok(DecodeCommand::Pause) => {}
                Err(_) => continue,
            }
            continue;
        }

        // Decode the next audio samples
        match decoder.decode_next() {
            Ok(Some(samples)) => {
                player.queue_samples(samples);
            }
            Ok(None) => {
                // End of stream
                playing = false;
            }
            Err(e) => {
                tracing::error!("Audio decode error: {}", e);
                thread::sleep(Duration::from_millis(10));
            }
        }

        // Small sleep to prevent busy loop and let sink process
        thread::sleep(Duration::from_millis(5));
    }
}

/// A simple frame scheduler that determines which frame to display.
///
/// This handles frame timing based on presentation timestamps.
pub struct FrameScheduler {
    /// The current playback position
    current_position: Duration,
    /// The last frame that was displayed
    current_frame: Option<VideoFrame>,
    /// Time when playback started (or was resumed)
    playback_start_time: Option<std::time::Instant>,
    /// Position when playback started
    playback_start_position: Duration,
}

impl FrameScheduler {
    /// Creates a new frame scheduler.
    pub fn new() -> Self {
        Self {
            current_position: Duration::ZERO,
            current_frame: None,
            playback_start_time: None,
            playback_start_position: Duration::ZERO,
        }
    }

    /// Starts or resumes playback.
    pub fn start(&mut self) {
        self.playback_start_time = Some(std::time::Instant::now());
        self.playback_start_position = self.current_position;
    }

    /// Pauses playback.
    pub fn pause(&mut self) {
        if let Some(start) = self.playback_start_time.take() {
            self.current_position = self.playback_start_position + start.elapsed();
        }
    }

    /// Seeks to a new position.
    pub fn seek(&mut self, position: Duration) {
        self.current_position = position;
        self.current_frame = None;

        if self.playback_start_time.is_some() {
            self.playback_start_time = Some(std::time::Instant::now());
            self.playback_start_position = position;
        }
    }

    /// Returns the current playback position.
    pub fn position(&self) -> Duration {
        match self.playback_start_time {
            Some(start) => self.playback_start_position + start.elapsed(),
            None => self.current_position,
        }
    }

    /// Returns true if playback is active.
    pub fn is_playing(&self) -> bool {
        self.playback_start_time.is_some()
    }

    /// Gets the next frame to display from the queue.
    ///
    /// This will return the appropriate frame based on the current playback
    /// position, dropping frames if we're behind schedule.
    pub fn get_next_frame(&mut self, queue: &FrameQueue) -> Option<VideoFrame> {
        let current_pos = self.position();

        // Keep popping frames until we find one that should be displayed now
        // or we're ahead of schedule
        loop {
            match queue.peek_pts() {
                Some(next_pts) => {
                    if next_pts <= current_pos {
                        // This frame should have been displayed already, take it
                        if let Some(frame) = queue.pop() {
                            // If this is the first frame or we're catching up,
                            // use this frame
                            if self.current_frame.is_none()
                                || frame.pts >= self.current_frame.as_ref().unwrap().pts
                            {
                                self.current_frame = Some(frame.clone());
                                return Some(frame);
                            }
                            // Otherwise, keep looking for a more recent frame
                            continue;
                        }
                    } else {
                        // We're ahead of schedule, return current frame
                        return self.current_frame.clone();
                    }
                }
                None => {
                    // Queue is empty, return current frame
                    return self.current_frame.clone();
                }
            }
        }
    }

    /// Returns the current frame without advancing.
    pub fn current_frame(&self) -> Option<&VideoFrame> {
        self.current_frame.as_ref()
    }
}

impl Default for FrameScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::video::{CpuFrame, DecodedFrame, PixelFormat, Plane};

    fn make_test_frame(pts: Duration) -> VideoFrame {
        let plane = Plane {
            data: vec![128; 100],
            stride: 10,
        };
        let cpu_frame = CpuFrame::new(PixelFormat::Yuv420p, 10, 10, vec![plane]);
        VideoFrame::new(pts, DecodedFrame::Cpu(cpu_frame))
    }

    #[test]
    fn test_frame_queue_push_pop() {
        let queue = FrameQueue::new(3);

        queue.push(make_test_frame(Duration::from_millis(0)));
        queue.push(make_test_frame(Duration::from_millis(33)));
        queue.push(make_test_frame(Duration::from_millis(66)));

        assert_eq!(queue.len(), 3);
        assert!(queue.is_full());

        let frame = queue.pop().unwrap();
        assert_eq!(frame.pts, Duration::from_millis(0));

        assert_eq!(queue.len(), 2);
        assert!(!queue.is_full());
    }

    #[test]
    fn test_frame_queue_flush() {
        let queue = FrameQueue::new(5);

        queue.push(make_test_frame(Duration::from_millis(0)));
        queue.push(make_test_frame(Duration::from_millis(33)));

        assert_eq!(queue.len(), 2);

        queue.flush();

        assert!(queue.is_empty());
        assert!(!queue.is_eos());
    }

    #[test]
    fn test_frame_scheduler_position() {
        let mut scheduler = FrameScheduler::new();

        assert_eq!(scheduler.position(), Duration::ZERO);

        scheduler.seek(Duration::from_secs(10));
        assert_eq!(scheduler.position(), Duration::from_secs(10));

        scheduler.start();
        std::thread::sleep(Duration::from_millis(50));
        assert!(scheduler.position() >= Duration::from_secs(10));

        scheduler.pause();
        let pos = scheduler.position();
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(scheduler.position(), pos);
    }
}
