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
use super::video::{VideoDecoderBackend, VideoError, VideoFrame, VideoMetadata};

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
    /// Returns None if end-of-stream is reached and the queue is empty,
    /// or if the timeout expires.
    ///
    /// Uses a loop to handle spurious wakeups correctly.
    pub fn pop_blocking(&self, timeout: Duration) -> Option<VideoFrame> {
        use std::time::Instant;

        let deadline = Instant::now() + timeout;
        let mut frames = self.frames.lock().unwrap();

        // Loop to handle spurious wakeups
        while frames.is_empty() {
            // Check EOS first
            if self.eos.load(Ordering::Acquire) {
                return None;
            }

            // Calculate remaining time
            let now = Instant::now();
            if now >= deadline {
                return None; // Timeout expired
            }
            let remaining = deadline - now;

            let (new_frames, timeout_result) = self
                .frame_available
                .wait_timeout(frames, remaining)
                .unwrap();
            frames = new_frames;

            // If we timed out and still empty, give up
            if timeout_result.timed_out() && frames.is_empty() {
                return None;
            }
            // Otherwise loop continues to recheck conditions
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

    /// Clears all frames from the queue for seeking.
    ///
    /// This sets the flushing flag to prevent new frames from being added,
    /// clears the queue, then resets both eos and flushing flags.
    ///
    /// The ordering of clearing eos before flushing is intentional:
    /// 1. Set flushing=true - blocks producers from pushing new frames
    /// 2. Clear the queue
    /// 3. Clear eos=false - reset end-of-stream state for new content
    /// 4. Clear flushing=false - allow producers to push new frames
    ///
    /// This ordering ensures that when flushing=false is visible, eos=false
    /// is also visible (Release ordering guarantees this). Producers check
    /// flushing before pushing, so they won't push until step 4, by which
    /// time eos has already been cleared.
    pub fn flush(&self) {
        self.flushing.store(true, Ordering::Release);

        // Wake up any blocked producers
        self.space_available.notify_all();

        {
            let mut frames = self.frames.lock().unwrap();
            frames.clear();
        }

        // Clear eos before flushing so consumers see consistent state
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

    /// Creates and starts a new decode thread using a factory function.
    ///
    /// The factory is called on the decode thread itself, which is necessary for
    /// decoders that require thread-local initialization (e.g., Windows COM objects).
    /// This ensures the decoder is created on the same thread where it will be used.
    ///
    /// Note: The decoder returned by the factory does NOT need to implement Send
    /// because it is created on and confined to the decode thread.
    ///
    /// Returns the DecodeThread and a receiver for the initialization result.
    /// The receiver will receive either Ok(VideoMetadata) on success or Err(VideoError) on failure.
    /// The caller MUST check this receiver to know if initialization succeeded.
    ///
    /// The thread will start in a paused state.
    pub fn new_from_factory<F, D>(
        factory: F,
        frame_queue: Arc<FrameQueue>,
    ) -> (
        Self,
        crossbeam_channel::Receiver<Result<VideoMetadata, VideoError>>,
    )
    where
        F: FnOnce() -> Result<D, VideoError> + Send + 'static,
        D: VideoDecoderBackend + 'static,
    {
        use std::sync::atomic::AtomicI32;

        let (command_tx, command_rx) = crossbeam_channel::unbounded();
        let (init_tx, init_rx) = crossbeam_channel::bounded(1);
        let stop_flag = Arc::new(AtomicBool::new(false));
        let duration = Arc::new(Mutex::new(None));
        let dimensions = Arc::new(Mutex::new(None));
        let buffering_percent = Arc::new(AtomicI32::new(0));

        let queue = Arc::clone(&frame_queue);
        let stop = Arc::clone(&stop_flag);
        let dur = Arc::clone(&duration);
        let dims = Arc::clone(&dimensions);
        let buf = Arc::clone(&buffering_percent);

        let handle = thread::spawn(move || {
            // Create decoder on this thread (important for COM/thread-local init)
            match factory() {
                Ok(decoder) => {
                    // Send metadata back to caller before starting decode loop
                    let metadata = decoder.metadata().clone();
                    *dur.lock().unwrap() = metadata.duration;
                    *dims.lock().unwrap() = Some((metadata.width, metadata.height));
                    let _ = init_tx.send(Ok(metadata));

                    decode_loop(decoder, queue, command_rx, stop, dur, dims, buf);
                }
                Err(e) => {
                    tracing::error!("Failed to create decoder on decode thread: {}", e);
                    let _ = init_tx.send(Err(e));
                }
            }
        });

        (
            Self {
                handle: Some(handle),
                command_tx,
                frame_queue,
                stop_flag,
                duration,
                dimensions,
                buffering_percent,
            },
            init_rx,
        )
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

/// Result of processing a decode command.
enum CommandResult {
    /// Continue processing, optionally updating playing state
    Continue(Option<bool>),
    /// Stop the decode loop
    Stop,
}

/// Processes a single decode command. Returns the result to apply.
fn process_decode_command<D: VideoDecoderBackend>(
    cmd: DecodeCommand,
    decoder: &mut D,
    frame_queue: &FrameQueue,
) -> CommandResult {
    match cmd {
        DecodeCommand::Stop => return CommandResult::Stop,
        DecodeCommand::Play => {
            frame_queue.clear_eos();
            if let Err(e) = decoder.resume() {
                tracing::error!("Failed to resume decoder: {}", e);
            }
            return CommandResult::Continue(Some(true));
        }
        DecodeCommand::Pause => {
            if let Err(e) = decoder.pause() {
                tracing::error!("Failed to pause decoder: {}", e);
            }
            return CommandResult::Continue(Some(false));
        }
        DecodeCommand::Seek(position) => {
            frame_queue.flush();
            if let Err(e) = decoder.seek(position) {
                tracing::error!("Seek failed: {}", e);
            }
            frame_queue.clear_eos();
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
    CommandResult::Continue(None)
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
        // Check for early termination (user closed video)
        if stop_flag.load(Ordering::Acquire) {
            tracing::debug!("Preview loop interrupted by stop signal");
            return;
        }

        match decoder.decode_next() {
            Ok(Some(frame)) => {
                // Check if this is a real frame (not a 1x1 placeholder)
                let (w, h) = frame.dimensions();
                if w > 1 && h > 1 {
                    tracing::info!(
                        "video: decoded preview frame at {:?} ({}x{})",
                        frame.pts,
                        w,
                        h
                    );
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
                tracing::warn!("video: failed to decode preview frame: {}", e);
                break;
            }
        }
    }

    // Wait for metadata to become available (ExoPlayer needs time to determine duration/dimensions)
    // This is important because pausing too early may prevent ExoPlayer from reporting metadata
    let metadata_wait_start = std::time::Instant::now();
    let metadata_timeout = Duration::from_secs(3);

    loop {
        // Check for early termination (user closed video)
        if stop_flag.load(Ordering::Acquire) {
            tracing::debug!("Metadata loop interrupted by stop signal");
            return;
        }

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
            match process_decode_command(cmd, &mut decoder, &frame_queue) {
                CommandResult::Stop => return,
                CommandResult::Continue(Some(new_playing)) => playing = new_playing,
                CommandResult::Continue(None) => {}
            }
        }

        // Update buffering percentage immediately (important for UI feedback)
        shared_buffering.store(decoder.buffering_percent(), Ordering::Relaxed);

        // Periodically update the shared duration and dimensions (every 500ms)
        if last_metadata_check.elapsed() > Duration::from_millis(500) {
            if let Some(dur) = decoder.duration() {
                *shared_duration.lock().unwrap() = Some(dur);
            }
            let dims = decoder.dimensions();
            if dims.0 > 0 && dims.1 > 0 {
                *shared_dimensions.lock().unwrap() = Some(dims);
            }
            last_metadata_check = std::time::Instant::now();
        }

        // When paused, wait for commands
        if !playing {
            let cmd = match command_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(cmd) => cmd,
                Err(_) => continue,
            };
            match process_decode_command(cmd, &mut decoder, &frame_queue) {
                CommandResult::Stop => return,
                CommandResult::Continue(Some(new_playing)) => playing = new_playing,
                CommandResult::Continue(None) => {}
            }
            continue;
        }

        // Don't decode if queue is full
        if frame_queue.is_full() {
            thread::sleep(Duration::from_millis(5));
            continue;
        }

        // Decode the next frame
        let frame = match decoder.decode_next() {
            Ok(Some(frame)) => frame,
            Ok(None) if decoder.is_eof() => {
                frame_queue.set_eos();
                playing = false;
                tracing::debug!("End of stream confirmed by decoder");
                continue;
            }
            Ok(None) => {
                tracing::trace!("decode_next returned None (buffering)");
                continue;
            }
            Err(e) => {
                tracing::error!("Decode error: {}", e);
                thread::sleep(Duration::from_millis(10));
                continue;
            }
        };

        tracing::trace!("Decoded frame at {:?}", frame.pts);
        if !frame_queue.push(frame) {
            tracing::debug!("Frame rejected by queue (flushing)");
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

/// Processes a single audio command. Returns the result to apply.
#[cfg(all(feature = "ffmpeg", not(target_os = "android")))]
fn process_audio_command(
    cmd: DecodeCommand,
    player: &mut super::audio::AudioPlayer,
    decoder: &mut AudioDecoder,
) -> CommandResult {
    match cmd {
        DecodeCommand::Stop => CommandResult::Stop,
        DecodeCommand::Play => {
            player.play();
            CommandResult::Continue(Some(true))
        }
        DecodeCommand::Pause => {
            player.pause();
            CommandResult::Continue(Some(false))
        }
        DecodeCommand::Seek(position) => {
            player.clear();
            if let Err(e) = decoder.seek(position) {
                tracing::error!("Audio seek failed: {}", e);
            }
            CommandResult::Continue(None)
        }
        // SetMuted and SetVolume are handled by the video decoder thread
        DecodeCommand::SetMuted(_) | DecodeCommand::SetVolume(_) => CommandResult::Continue(None),
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
        if stop_flag.load(Ordering::Acquire) {
            break;
        }

        // Process commands (non-blocking)
        while let Ok(cmd) = command_rx.try_recv() {
            match process_audio_command(cmd, &mut player, &mut decoder) {
                CommandResult::Stop => return,
                CommandResult::Continue(Some(new_playing)) => playing = new_playing,
                CommandResult::Continue(None) => {}
            }
        }

        // When paused, wait for commands
        if !playing {
            let cmd = match command_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(cmd) => cmd,
                Err(_) => continue,
            };
            match process_audio_command(cmd, &mut player, &mut decoder) {
                CommandResult::Stop => return,
                CommandResult::Continue(Some(new_playing)) => playing = new_playing,
                CommandResult::Continue(None) => {}
            }
            continue;
        }

        // Decode the next audio samples
        let samples = match decoder.decode_next() {
            Ok(Some(samples)) => samples,
            Ok(None) => {
                playing = false;
                continue;
            }
            Err(e) => {
                tracing::error!("Audio decode error: {}", e);
                thread::sleep(Duration::from_millis(10));
                continue;
            }
        };

        player.queue_samples(samples);
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
        self.get_next_frame_with_clock(queue, None)
    }

    /// Gets the next frame using an external audio clock for A/V sync.
    ///
    /// When `audio_clock_pos` is Some, video timing is driven by the audio
    /// playback position rather than wall-clock time. This provides accurate
    /// A/V synchronization.
    pub fn get_next_frame_with_clock(
        &mut self,
        queue: &FrameQueue,
        audio_clock_pos: Option<Duration>,
    ) -> Option<VideoFrame> {
        // If waiting for first frame, accept any frame to start the clock
        if self.waiting_for_first_frame {
            let Some(frame) = queue.pop() else {
                return self.current_frame.clone();
            };
            self.on_frame_received(frame.pts);
            self.current_frame = Some(frame.clone());
            self.stalled = false;
            return Some(frame);
        }

        // Use audio clock if available, otherwise fall back to wall-clock
        let current_pos = audio_clock_pos.unwrap_or_else(|| self.position());

        // Keep popping frames until we find one that should be displayed now
        loop {
            let Some(next_pts) = queue.peek_pts() else {
                // Queue is empty - we're stalled (buffering)
                self.handle_stall();
                return self.current_frame.clone();
            };

            // We have frames - clear stall state and resync clock if needed
            self.clear_stall_if_needed();

            // Accept frame if:
            // 1. It's at or before current position (normal case), OR
            // 2. We have no current frame (after seek) and it's within 500ms
            let should_accept = next_pts <= current_pos
                || (self.current_frame.is_none()
                    && next_pts <= current_pos + Duration::from_millis(500));

            if !should_accept {
                // We're ahead of schedule, return current frame
                return self.current_frame.clone();
            }

            let Some(frame) = queue.pop() else { continue };

            // Skip if this frame is older than what we already have
            if let Some(ref current) = self.current_frame {
                if frame.pts < current.pts {
                    continue;
                }
            }

            self.current_position = frame.pts;
            self.current_frame = Some(frame.clone());
            return Some(frame);
        }
    }

    /// Handles entering stall state when queue is empty.
    fn handle_stall(&mut self) {
        if self.stalled || self.playback_start_time.is_none() {
            return;
        }
        self.current_position =
            self.playback_start_position + self.playback_start_time.unwrap().elapsed();
        self.stalled = true;
        tracing::debug!("Stalled at {:?} (queue empty)", self.current_position);
    }

    /// Clears stall state and resyncs clock when frames become available.
    fn clear_stall_if_needed(&mut self) {
        if !self.stalled {
            return;
        }
        self.stalled = false;
        self.playback_start_time = Some(std::time::Instant::now());
        self.playback_start_position = self.current_position;
        tracing::debug!("Resuming from stall at {:?}", self.current_position);
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

// ============================================================================
// Windows-specific decode loop with native audio
// ============================================================================

/// Windows-specific decode thread that handles both video and audio.
///
/// This uses a single IMFSourceReader to read both video and audio samples,
/// which is required because IMFSourceReader is not thread-safe for concurrent reads.
#[cfg(all(target_os = "windows", feature = "windows-native-video"))]
pub struct WindowsDecodeThread {
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
    /// Audio queue for decoded audio frames
    audio_queue: Option<Arc<super::windows_audio::AudioQueue>>,
}

#[cfg(all(target_os = "windows", feature = "windows-native-video"))]
impl WindowsDecodeThread {
    /// Creates and starts a new Windows decode thread using a factory function.
    ///
    /// The factory is called on the decode thread for COM thread safety.
    /// Returns the thread and init result receiver containing metadata + audio resources.
    pub fn new_from_factory<F>(
        factory: F,
        frame_queue: Arc<FrameQueue>,
    ) -> (
        Self,
        crossbeam_channel::Receiver<
            Result<
                (
                    super::video::VideoMetadata,
                    Option<super::windows_audio::AudioFormatInfo>,
                    Option<Arc<super::windows_audio::AudioQueue>>,
                    Option<Arc<super::windows_audio::AudioClock>>,
                ),
                super::video::VideoError,
            >,
        >,
    )
    where
        F: FnOnce() -> Result<super::windows_video::WindowsVideoDecoder, super::video::VideoError>
            + Send
            + 'static,
    {
        use std::sync::atomic::AtomicI32;

        let (command_tx, command_rx) = crossbeam_channel::unbounded();
        let (init_tx, init_rx) = crossbeam_channel::bounded(1);
        let stop_flag = Arc::new(AtomicBool::new(false));
        let duration = Arc::new(Mutex::new(None));
        let dimensions = Arc::new(Mutex::new(None));
        let buffering_percent = Arc::new(AtomicI32::new(0));

        let queue = Arc::clone(&frame_queue);
        let stop = Arc::clone(&stop_flag);
        let dur = Arc::clone(&duration);
        let dims = Arc::clone(&dimensions);
        let buf = Arc::clone(&buffering_percent);

        let handle = thread::spawn(move || {
            match factory() {
                Ok(decoder) => {
                    let metadata = decoder.metadata().clone();
                    let audio_format = decoder.audio_format().cloned();

                    *dur.lock().unwrap() = metadata.duration;
                    *dims.lock().unwrap() = Some((metadata.width, metadata.height));

                    // Create audio queue and clock if audio is available
                    let (audio_queue, audio_clock) = if decoder.has_audio() {
                        if let Some(ref fmt) = audio_format {
                            let aq = super::windows_audio::AudioQueue::new(50);
                            let ac =
                                Arc::new(super::windows_audio::AudioClock::new(fmt.sample_rate));
                            (Some(aq), Some(ac))
                        } else {
                            (None, None)
                        }
                    } else {
                        (None, None)
                    };

                    // Send init result with queue/clock for playback to use
                    let _ = init_tx.send(Ok((
                        metadata,
                        audio_format,
                        audio_queue.clone(),
                        audio_clock,
                    )));

                    windows_decode_loop(
                        decoder,
                        queue,
                        audio_queue,
                        command_rx,
                        stop,
                        dur,
                        dims,
                        buf,
                    );
                }
                Err(e) => {
                    tracing::error!("Failed to create Windows decoder: {}", e);
                    let _ = init_tx.send(Err(e));
                }
            }
        });

        (
            Self {
                handle: Some(handle),
                command_tx,
                frame_queue,
                stop_flag,
                duration,
                dimensions,
                buffering_percent,
                audio_queue: None, // Will be set after init completes
            },
            init_rx,
        )
    }

    /// Sets the audio queue (called after init completes with audio format).
    pub fn set_audio_queue(&mut self, queue: Arc<super::windows_audio::AudioQueue>) {
        self.audio_queue = Some(queue);
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
    pub fn seek(&self, position: Duration) {
        self.frame_queue.flush();
        self.buffering_percent.store(0, Ordering::Relaxed);
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

    /// Returns the current known duration.
    pub fn duration(&self) -> Option<Duration> {
        *self.duration.lock().unwrap()
    }

    /// Returns the current known dimensions.
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        *self.dimensions.lock().unwrap()
    }

    /// Returns the current buffering percentage.
    pub fn buffering_percent(&self) -> i32 {
        self.buffering_percent.load(Ordering::Relaxed)
    }
}

#[cfg(all(target_os = "windows", feature = "windows-native-video"))]
impl Drop for WindowsDecodeThread {
    fn drop(&mut self) {
        self.stop();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Windows-specific decode loop that reads both video and audio.
///
/// This is the P0 "single-threaded demux loop" - IMFSourceReader is not
/// thread-safe, so both streams must be read from the same thread.
#[cfg(all(target_os = "windows", feature = "windows-native-video"))]
fn windows_decode_loop(
    mut decoder: super::windows_video::WindowsVideoDecoder,
    frame_queue: Arc<FrameQueue>,
    audio_queue: Option<Arc<super::windows_audio::AudioQueue>>,
    command_rx: crossbeam_channel::Receiver<DecodeCommand>,
    stop_flag: Arc<AtomicBool>,
    shared_duration: Arc<Mutex<Option<Duration>>>,
    shared_dimensions: Arc<Mutex<Option<(u32, u32)>>>,
    shared_buffering: Arc<std::sync::atomic::AtomicI32>,
) {
    use super::video::VideoDecoderBackend;

    let mut playing = false;
    let mut last_metadata_check = std::time::Instant::now();
    let has_audio = audio_queue.is_some();

    // Decode one frame immediately for preview
    let mut preview_attempts = 0;
    let max_preview_attempts = 30;

    loop {
        if stop_flag.load(Ordering::Acquire) {
            tracing::debug!("Windows preview loop interrupted by stop signal");
            return;
        }

        match decoder.decode_next() {
            Ok(Some(frame)) => {
                let (w, h) = frame.dimensions();
                if w > 1 && h > 1 {
                    tracing::info!(
                        "windows: decoded preview frame at {:?} ({}x{})",
                        frame.pts,
                        w,
                        h
                    );
                    let _ = frame_queue.try_push(frame);
                    break;
                } else {
                    preview_attempts += 1;
                    if preview_attempts >= max_preview_attempts {
                        let _ = frame_queue.try_push(frame);
                        break;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
            }
            Ok(None) => {
                preview_attempts += 1;
                if preview_attempts >= max_preview_attempts {
                    break;
                }
                thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                tracing::warn!("windows: failed to decode preview frame: {}", e);
                break;
            }
        }
    }

    // Wait for metadata
    let metadata_wait_start = std::time::Instant::now();
    let metadata_timeout = Duration::from_secs(3);

    loop {
        if stop_flag.load(Ordering::Acquire) {
            return;
        }

        let duration_opt = decoder.duration();
        let has_duration = duration_opt.is_some();
        let dims = decoder.dimensions();
        let has_dimensions = dims.0 > 1 && dims.1 > 1;

        if has_duration && has_dimensions {
            *shared_duration.lock().unwrap() = duration_opt;
            *shared_dimensions.lock().unwrap() = Some(dims);
            break;
        }

        if metadata_wait_start.elapsed() > metadata_timeout {
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

    // Pause after preview
    if let Err(e) = decoder.pause() {
        tracing::debug!("Failed to pause after preview: {}", e);
    }

    // Main decode loop - reads both video and audio
    loop {
        if stop_flag.load(Ordering::Acquire) {
            break;
        }

        // Process commands
        while let Ok(cmd) = command_rx.try_recv() {
            match cmd {
                DecodeCommand::Stop => return,
                DecodeCommand::Play => {
                    frame_queue.clear_eos();
                    if let Err(e) = decoder.resume() {
                        tracing::error!("Failed to resume decoder: {}", e);
                    }
                    playing = true;
                }
                DecodeCommand::Pause => {
                    if let Err(e) = decoder.pause() {
                        tracing::error!("Failed to pause decoder: {}", e);
                    }
                    playing = false;
                }
                DecodeCommand::Seek(position) => {
                    frame_queue.flush();
                    // Clear audio queue on seek
                    if let Some(ref aq) = audio_queue {
                        aq.clear();
                    }
                    if let Err(e) = decoder.seek(position) {
                        tracing::error!("Seek failed: {}", e);
                    }
                    frame_queue.clear_eos();
                }
                DecodeCommand::SetMuted(_) | DecodeCommand::SetVolume(_) => {
                    // Audio volume/mute handled by WindowsAudioPlayback
                }
            }
        }

        shared_buffering.store(decoder.buffering_percent(), Ordering::Relaxed);

        // Update metadata periodically
        if last_metadata_check.elapsed() > Duration::from_millis(500) {
            if let Some(dur) = decoder.duration() {
                *shared_duration.lock().unwrap() = Some(dur);
            }
            let dims = decoder.dimensions();
            if dims.0 > 0 && dims.1 > 0 {
                *shared_dimensions.lock().unwrap() = Some(dims);
            }
            last_metadata_check = std::time::Instant::now();
        }

        // When paused, wait for commands
        if !playing {
            match command_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(cmd) => match cmd {
                    DecodeCommand::Stop => return,
                    DecodeCommand::Play => {
                        frame_queue.clear_eos();
                        if let Err(e) = decoder.resume() {
                            tracing::error!("Failed to resume decoder: {}", e);
                        }
                        playing = true;
                    }
                    DecodeCommand::Pause => playing = false,
                    DecodeCommand::Seek(position) => {
                        frame_queue.flush();
                        if let Some(ref aq) = audio_queue {
                            aq.clear();
                        }
                        if let Err(e) = decoder.seek(position) {
                            tracing::error!("Seek failed: {}", e);
                        }
                        frame_queue.clear_eos();
                    }
                    _ => {}
                },
                Err(_) => continue,
            }
            continue;
        }

        // Interleaved audio/video decode loop
        // Read audio first (audio needs to stay ahead for smooth playback)
        if has_audio {
            if let Some(ref aq) = audio_queue {
                // Read audio samples if queue has space
                if aq.len() < 40 {
                    match decoder.read_audio_sample() {
                        Ok(Some(audio_frame)) => {
                            if !aq.push(audio_frame) {
                                tracing::trace!("Audio queue full, dropping frame");
                            }
                        }
                        Ok(None) => {
                            // No audio sample available (EOS or gap)
                        }
                        Err(e) => {
                            tracing::warn!("Audio decode error: {}", e);
                        }
                    }
                }
            }
        }

        // Read video
        if frame_queue.is_full() {
            thread::sleep(Duration::from_millis(5));
            continue;
        }

        let frame = match decoder.decode_next() {
            Ok(Some(frame)) => frame,
            Ok(None) if decoder.is_eof() => {
                frame_queue.set_eos();
                playing = false;
                tracing::debug!("Windows: End of stream");
                continue;
            }
            Ok(None) => {
                continue;
            }
            Err(e) => {
                tracing::error!("Video decode error: {}", e);
                thread::sleep(Duration::from_millis(10));
                continue;
            }
        };

        tracing::trace!("Decoded video frame at {:?}", frame.pts);
        if !frame_queue.push(frame) {
            tracing::debug!("Video frame rejected (flushing)");
        }
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
