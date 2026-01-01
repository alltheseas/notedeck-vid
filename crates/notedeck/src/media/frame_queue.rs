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
    /// Set muted state (Android only - audio controlled by ExoPlayer)
    SetMuted(bool),
    /// Set volume level (Android only - audio controlled by ExoPlayer)
    SetVolume(f32),
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
    /// Shared duration (updated by decode thread, read by UI thread)
    duration: Arc<Mutex<Option<Duration>>>,
    /// Shared dimensions (updated by decode thread, read by UI thread)
    dimensions: Arc<Mutex<Option<(u32, u32)>>>,
    /// Shared buffering percentage (0-100, updated by decode thread)
    buffering_percent: Arc<std::sync::atomic::AtomicI32>,
}

impl DecodeThread {
    /// Creates and starts a new decode thread.
    ///
    /// The thread will start in a paused state.
    pub fn new<D: VideoDecoderBackend + Send + 'static>(
        decoder: D,
        frame_queue: Arc<FrameQueue>,
    ) -> Self {
        use std::sync::atomic::AtomicI32;

        let (command_tx, command_rx) = crossbeam_channel::unbounded();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let duration = Arc::new(Mutex::new(None));
        let dimensions = Arc::new(Mutex::new(None));
        let buffering_percent = Arc::new(AtomicI32::new(0)); // Start unbuffered, decoder will update

        let queue = Arc::clone(&frame_queue);
        let stop = Arc::clone(&stop_flag);
        let dur = Arc::clone(&duration);
        let dims = Arc::clone(&dimensions);
        let buf = Arc::clone(&buffering_percent);

        let handle = thread::spawn(move || {
            decode_loop(decoder, queue, command_rx, stop, dur, dims, buf);
        });

        Self {
            handle: Some(handle),
            command_tx,
            frame_queue,
            stop_flag,
            duration,
            dimensions,
            buffering_percent,
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
        // Immediately show buffering indicator - HTTP streams need to rebuffer after seek
        self.buffering_percent.store(0, Ordering::Relaxed);
        let _ = self.command_tx.send(DecodeCommand::Seek(position));
    }

    /// Stops the decode thread.
    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Release);
        let _ = self.command_tx.send(DecodeCommand::Stop);
    }

    /// Sets the muted state (Android only - audio is controlled by ExoPlayer).
    pub fn set_muted(&self, muted: bool) {
        let _ = self.command_tx.send(DecodeCommand::SetMuted(muted));
    }

    /// Sets the volume level (Android only - audio is controlled by ExoPlayer).
    pub fn set_volume(&self, volume: f32) {
        let _ = self.command_tx.send(DecodeCommand::SetVolume(volume));
    }

    /// Returns a reference to the frame queue.
    pub fn frame_queue(&self) -> &Arc<FrameQueue> {
        &self.frame_queue
    }

    /// Returns the current known duration (updated by decode thread).
    pub fn duration(&self) -> Option<Duration> {
        *self.duration.lock().unwrap()
    }

    /// Returns the current known dimensions (updated by decode thread).
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        *self.dimensions.lock().unwrap()
    }

    /// Returns the current buffering percentage (0-100).
    pub fn buffering_percent(&self) -> i32 {
        self.buffering_percent.load(Ordering::Relaxed)
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
    shared_duration: Arc<Mutex<Option<Duration>>>,
    shared_dimensions: Arc<Mutex<Option<(u32, u32)>>>,
    shared_buffering: Arc<std::sync::atomic::AtomicI32>,
) {
    let mut playing = false;
    let mut last_metadata_check = std::time::Instant::now();

    // Decode one frame immediately for preview (before waiting for Play command)
    // This allows showing the first frame without starting playback
    // Try multiple times since streaming decoders (HTTP, ExoPlayer) need time to buffer
    let mut preview_attempts = 0;
    let max_preview_attempts = 30; // Try for up to ~3 seconds for slow HTTP streams

    loop {
        match decoder.decode_next() {
            Ok(Some(frame)) => {
                // Check if this is a real frame (not a 1x1 placeholder)
                let (w, h) = frame.dimensions();
                if w > 1 && h > 1 {
                    tracing::info!("Decoded preview frame at {:?} ({}x{})", frame.pts, w, h);
                    let _ = frame_queue.try_push(frame);
                    break;
                } else {
                    // Placeholder frame, keep trying
                    preview_attempts += 1;
                    if preview_attempts >= max_preview_attempts {
                        tracing::debug!("Max preview attempts reached, using placeholder");
                        let _ = frame_queue.try_push(frame);
                        break;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
            }
            Ok(None) => {
                // For HTTP streams, None often means "still buffering" not "EOS"
                preview_attempts += 1;
                if preview_attempts >= max_preview_attempts {
                    tracing::debug!(
                        "No preview frame available after {} attempts",
                        preview_attempts
                    );
                    break;
                }
                // Wait a bit before retrying
                thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                tracing::warn!("Failed to decode preview frame: {}", e);
                break;
            }
        }
    }

    // Wait for metadata to become available (ExoPlayer needs time to determine duration/dimensions)
    // This is important because pausing too early may prevent ExoPlayer from reporting metadata
    let metadata_wait_start = std::time::Instant::now();
    let metadata_timeout = Duration::from_secs(3);

    loop {
        let duration_opt = decoder.duration();
        let has_duration = duration_opt.is_some();
        let dims = decoder.dimensions();
        let has_dimensions = dims.0 > 1 && dims.1 > 1; // >1 to exclude placeholder

        if has_duration && has_dimensions {
            *shared_duration.lock().unwrap() = duration_opt;
            *shared_dimensions.lock().unwrap() = Some(dims);
            break;
        }

        if metadata_wait_start.elapsed() > metadata_timeout {
            tracing::warn!("Timeout waiting for video metadata");
            // Store whatever we have
            if let Some(dur) = duration_opt {
                *shared_duration.lock().unwrap() = Some(dur);
            }
            if dims.0 > 0 && dims.1 > 0 {
                *shared_dimensions.lock().unwrap() = Some(dims);
            }
            break;
        }

        thread::sleep(Duration::from_millis(100));
    }

    // Pause the decoder after getting preview frame (for decoders like ExoPlayer that auto-play)
    if let Err(e) = decoder.pause() {
        tracing::debug!("Failed to pause after preview: {}", e);
    }

    // Note: We no longer count consecutive Nones for EOS detection.
    // Instead, we rely on decoder.is_eof() which checks actual decoder state.

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
                    // Resume the underlying decoder (e.g., ExoPlayer on Android)
                    if let Err(e) = decoder.resume() {
                        tracing::error!("Failed to resume decoder: {}", e);
                    }
                }
                DecodeCommand::Pause => {
                    playing = false;
                    // Pause the underlying decoder (e.g., ExoPlayer on Android)
                    if let Err(e) = decoder.pause() {
                        tracing::error!("Failed to pause decoder: {}", e);
                    }
                }
                DecodeCommand::Seek(position) => {
                    // Flush queue again AFTER seek completes to catch any frames
                    // that snuck in between DecodeThread::seek() flush and now
                    frame_queue.flush();
                    if let Err(e) = decoder.seek(position) {
                        tracing::error!("Seek failed: {}", e);
                    }
                    frame_queue.clear_eos();
                }
                DecodeCommand::Stop => {
                    return;
                }
                DecodeCommand::SetMuted(muted) => {
                    if let Err(e) = decoder.set_muted(muted) {
                        tracing::error!("Failed to set muted: {}", e);
                    }
                }
                DecodeCommand::SetVolume(volume) => {
                    if let Err(e) = decoder.set_volume(volume) {
                        tracing::error!("Failed to set volume: {}", e);
                    }
                }
            }
        }

        // Update buffering percentage immediately (important for UI feedback)
        shared_buffering.store(decoder.buffering_percent(), Ordering::Relaxed);

        // Periodically update the shared duration and dimensions (every 500ms)
        if last_metadata_check.elapsed() > Duration::from_millis(500) {
            // Update duration
            if let Some(dur) = decoder.duration() {
                *shared_duration.lock().unwrap() = Some(dur);
            }

            // Update dimensions
            let dims = decoder.dimensions();
            if dims.0 > 0 && dims.1 > 0 {
                *shared_dimensions.lock().unwrap() = Some(dims);
            }

            last_metadata_check = std::time::Instant::now();
        }

        if !playing {
            // Wait for a command when paused
            match command_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(DecodeCommand::Play) => {
                    playing = true;
                    frame_queue.clear_eos();
                    // Resume the underlying decoder (e.g., ExoPlayer on Android)
                    if let Err(e) = decoder.resume() {
                        tracing::error!("Failed to resume decoder: {}", e);
                    }
                }
                Ok(DecodeCommand::Seek(position)) => {
                    // Flush queue again AFTER seek completes to catch any frames
                    // that snuck in between DecodeThread::seek() flush and now
                    frame_queue.flush();
                    if let Err(e) = decoder.seek(position) {
                        tracing::error!("Seek failed: {}", e);
                    }
                    frame_queue.clear_eos();
                }
                Ok(DecodeCommand::Stop) => return,
                Ok(DecodeCommand::Pause) => {
                    // Pause the underlying decoder (e.g., ExoPlayer on Android)
                    if let Err(e) = decoder.pause() {
                        tracing::error!("Failed to pause decoder: {}", e);
                    }
                }
                Ok(DecodeCommand::SetMuted(muted)) => {
                    if let Err(e) = decoder.set_muted(muted) {
                        tracing::error!("Failed to set muted: {}", e);
                    }
                }
                Ok(DecodeCommand::SetVolume(volume)) => {
                    if let Err(e) = decoder.set_volume(volume) {
                        tracing::error!("Failed to set volume: {}", e);
                    }
                }
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
                tracing::trace!("Decoded frame at {:?}", frame.pts);
                if !frame_queue.push(frame) {
                    // Queue was flushed, likely due to seek
                    tracing::debug!("Frame rejected by queue (flushing)");
                    continue;
                }
            }
            Ok(None) => {
                // Check if decoder actually reached end of stream
                // This is much more reliable than counting consecutive Nones,
                // which can false-positive during HTTP stream rebuffering
                if decoder.is_eof() {
                    frame_queue.set_eos();
                    playing = false;
                    tracing::debug!("End of stream confirmed by decoder");
                } else {
                    // Buffering - log occasionally to track progress
                    tracing::trace!("decode_next returned None (buffering)");
                }
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
                // SetMuted and SetVolume are handled by the video decoder thread
                DecodeCommand::SetMuted(_) | DecodeCommand::SetVolume(_) => {}
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
                // SetMuted and SetVolume are handled by the video decoder thread
                Ok(DecodeCommand::SetMuted(_)) | Ok(DecodeCommand::SetVolume(_)) => {}
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
/// The scheduler only advances position when frames are actually being delivered,
/// preventing the scroll bar from advancing during buffering.
pub struct FrameScheduler {
    /// The current playback position (updated from frame PTS)
    current_position: Duration,
    /// The last frame that was displayed
    current_frame: Option<VideoFrame>,
    /// Time when playback started (or was resumed) - only set after first frame arrives
    playback_start_time: Option<std::time::Instant>,
    /// Position when playback started (synced to frame PTS)
    playback_start_position: Duration,
    /// True if we're waiting for the first frame after play/seek
    waiting_for_first_frame: bool,
    /// True if playback has been requested (even if waiting for first frame)
    playback_requested: bool,
    /// True if we're stalled (queue empty during playback)
    stalled: bool,
}

impl FrameScheduler {
    /// Creates a new frame scheduler.
    pub fn new() -> Self {
        Self {
            current_position: Duration::ZERO,
            current_frame: None,
            playback_start_time: None,
            playback_start_position: Duration::ZERO,
            waiting_for_first_frame: false,
            playback_requested: false,
            stalled: false,
        }
    }

    /// Starts or resumes playback.
    /// Note: The clock doesn't actually start until the first frame arrives.
    pub fn start(&mut self) {
        self.playback_requested = true;
        self.waiting_for_first_frame = true;
        self.stalled = false;
        // Don't set playback_start_time yet - wait for first frame
    }

    /// Pauses playback.
    pub fn pause(&mut self) {
        self.playback_requested = false;
        self.waiting_for_first_frame = false;
        self.stalled = false;
        if let Some(start) = self.playback_start_time.take() {
            self.current_position = self.playback_start_position + start.elapsed();
        }
    }

    /// Seeks to a new position.
    pub fn seek(&mut self, position: Duration) {
        self.current_position = position;
        self.current_frame = None;
        self.stalled = false;

        if self.playback_requested {
            // Wait for first frame at new position before resuming clock
            self.waiting_for_first_frame = true;
            self.playback_start_time = None;
        }
    }

    /// Returns the current playback position.
    pub fn position(&self) -> Duration {
        // If stalled (queue empty during playback), return the last known position
        // to prevent the scroll bar from advancing during buffering
        if self.stalled {
            return self.current_position;
        }

        match self.playback_start_time {
            Some(start) => self.playback_start_position + start.elapsed(),
            None => self.current_position,
        }
    }

    /// Returns true if playback is active (clock is running).
    pub fn is_playing(&self) -> bool {
        self.playback_start_time.is_some()
    }

    /// Returns true if playback has been requested (even if buffering).
    pub fn is_playback_requested(&self) -> bool {
        self.playback_requested
    }

    /// Called when a frame is received to sync the clock.
    /// If we were waiting for the first frame, this starts the clock.
    fn on_frame_received(&mut self, frame_pts: Duration) {
        if self.waiting_for_first_frame && self.playback_requested {
            // First frame after play/seek - start the clock synced to frame PTS
            self.playback_start_time = Some(std::time::Instant::now());
            self.playback_start_position = frame_pts;
            self.waiting_for_first_frame = false;
            tracing::debug!("Clock started at frame PTS {:?}", frame_pts);
        }
    }

    /// Gets the next frame to display from the queue.
    ///
    /// This will return the appropriate frame based on the current playback
    /// position, dropping frames if we're behind schedule.
    pub fn get_next_frame(&mut self, queue: &FrameQueue) -> Option<VideoFrame> {
        // If waiting for first frame, accept any frame to start the clock
        if self.waiting_for_first_frame {
            if let Some(frame) = queue.pop() {
                self.on_frame_received(frame.pts);
                self.current_frame = Some(frame.clone());
                self.stalled = false;
                return Some(frame);
            }
            // No frame yet, return current frame (likely None)
            return self.current_frame.clone();
        }

        let current_pos = self.position();

        // Keep popping frames until we find one that should be displayed now
        // or we're ahead of schedule
        loop {
            match queue.peek_pts() {
                Some(next_pts) => {
                    // We have frames - clear stall state and resync clock if needed
                    if self.stalled {
                        self.stalled = false;
                        // Resync clock to continue from where we stalled
                        self.playback_start_time = Some(std::time::Instant::now());
                        self.playback_start_position = self.current_position;
                        tracing::debug!("Resuming from stall at {:?}", self.current_position);
                    }

                    // Accept frame if:
                    // 1. It's at or before current position (normal case), OR
                    // 2. We have no current frame (after seek) and it's within 500ms
                    let should_accept = next_pts <= current_pos
                        || (self.current_frame.is_none()
                            && next_pts <= current_pos + Duration::from_millis(500));

                    if should_accept {
                        // This frame should be displayed
                        if let Some(frame) = queue.pop() {
                            // If this is the first frame or we're catching up,
                            // use this frame
                            if self.current_frame.is_none()
                                || frame.pts >= self.current_frame.as_ref().unwrap().pts
                            {
                                self.current_position = frame.pts;
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
                    // Queue is empty - we're stalled (buffering)
                    if self.playback_start_time.is_some() && !self.stalled {
                        // Capture current position before stalling
                        self.current_position = self.playback_start_position
                            + self.playback_start_time.unwrap().elapsed();
                        self.stalled = true;
                        tracing::debug!("Stalled at {:?} (queue empty)", self.current_position);
                    }
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
