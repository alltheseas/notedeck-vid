//! Video player widget for egui.
//!
//! This module provides a video player widget that can be embedded
//! in egui UIs. It handles:
//! - Video decoding via FFmpeg (or ExoPlayer on Android)
//! - Hardware acceleration (VideoToolbox, VAAPI, D3D11VA, MediaCodec)
//! - GPU texture upload via wgpu
//! - YUV to RGB color space conversion
//! - Frame timing and synchronization
//! - Playback controls (play, pause, seek)
//! - Audio volume/mute control
//!
//! # Usage
//!
//! ```ignore
//! use notedeck::media::{VideoPlayer, VideoPlayerExt};
//!
//! // Create a video player with wgpu render state
//! let mut player = VideoPlayer::with_wgpu(
//!     "https://example.com/video.mp4",
//!     &wgpu_render_state,
//! )
//! .with_autoplay(true)
//! .with_loop(true)
//! .with_controls(true);
//!
//! // In your egui update loop:
//! egui::CentralPanel::default().show(ctx, |ui| {
//!     let size = egui::vec2(640.0, 360.0);
//!     let response = player.show(ui, size);
//!
//!     if response.state_changed {
//!         // Handle playback state changes
//!     }
//! });
//! ```
//!
//! # Platform Support
//!
//! - **macOS**: VideoToolbox hardware acceleration
//! - **Windows**: D3D11VA hardware acceleration
//! - **Linux**: VAAPI hardware acceleration
//! - **Android**: ExoPlayer with MediaCodec

use std::sync::{Arc, Mutex};
use std::time::Duration;

use egui::{Response, Sense, Ui, Vec2};
use egui_wgpu::wgpu;

#[cfg(target_os = "android")]
use super::android_video::AndroidVideoDecoder;
use super::audio::AudioHandle;
#[cfg(all(feature = "ffmpeg", not(target_os = "android")))]
use super::frame_queue::AudioThread;
use super::frame_queue::{DecodeThread, FrameQueue, FrameScheduler};
#[cfg(all(target_os = "linux", feature = "linux-gstreamer-video"))]
use super::linux_video_gst::GStreamerDecoder;
#[cfg(all(target_os = "macos", feature = "macos-native-video"))]
use super::macos_video::MacOSVideoDecoder;
use super::video::{
    CpuFrame, PixelFormat, VideoDecoderBackend, VideoError, VideoMetadata, VideoState,
};
use super::video_controls::{VideoControls, VideoControlsConfig, VideoControlsResponse};
#[cfg(all(feature = "ffmpeg", not(target_os = "android")))]
use super::video_decoder::FfmpegDecoder;
use super::video_texture::{VideoRenderCallback, VideoRenderResources, VideoTexture};

/// Shared state for pending frame to be rendered.
/// This allows the prepare callback to access frame data for texture creation/upload.
#[derive(Default)]
pub struct PendingFrame {
    /// The CPU frame data to upload
    pub frame: Option<CpuFrame>,
    /// Whether the texture needs to be recreated (dimensions/format changed)
    pub needs_recreate: bool,
}

/// A video player widget for egui.
///
/// This widget handles video playback, including:
/// - Decoding video frames on a background thread
/// - Uploading frames to GPU textures
/// - Rendering frames via wgpu
/// - Basic playback controls (play, pause, seek)
pub struct VideoPlayer {
    /// Current playback state
    state: VideoState,
    /// Video metadata
    metadata: Option<VideoMetadata>,
    /// The frame queue for decoded frames
    frame_queue: Arc<FrameQueue>,
    /// The decode thread
    decode_thread: Option<DecodeThread>,
    /// Frame scheduler for timing
    scheduler: FrameScheduler,
    /// Current video texture
    texture: Arc<Mutex<Option<VideoTexture>>>,
    /// Whether the player has been initialized
    initialized: bool,
    /// The URL being played
    url: String,
    /// Whether to autoplay
    autoplay: bool,
    /// Whether to loop playback
    loop_playback: bool,
    /// Whether audio is muted
    muted: bool,
    /// wgpu device for texture creation (internally Arc'd by wgpu)
    device: Option<wgpu::Device>,
    /// wgpu queue for texture upload (internally Arc'd by wgpu)
    queue: Option<wgpu::Queue>,
    /// Pending frame data for the render callback to process
    pending_frame: Arc<Mutex<PendingFrame>>,
    /// Whether to show controls overlay
    show_controls: bool,
    /// Controls configuration
    controls_config: VideoControlsConfig,
    /// Audio handle for volume/mute control
    audio_handle: AudioHandle,
    /// Audio decode/playback thread
    #[cfg(all(feature = "ffmpeg", not(target_os = "android")))]
    audio_thread: Option<AudioThread>,
    /// Background thread for async initialization
    init_thread: Option<std::thread::JoinHandle<()>>,
    /// Receiver for async initialization result
    init_receiver:
        Option<std::sync::mpsc::Receiver<Result<Box<dyn VideoDecoderBackend + Send>, VideoError>>>,
}

impl VideoPlayer {
    /// Creates a new video player for the given URL.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            state: VideoState::Loading,
            metadata: None,
            frame_queue: Arc::new(FrameQueue::with_default_capacity()),
            decode_thread: None,
            scheduler: FrameScheduler::new(),
            texture: Arc::new(Mutex::new(None)),
            initialized: false,
            url: url.into(),
            autoplay: false,
            loop_playback: false,
            muted: false,
            device: None,
            queue: None,
            pending_frame: Arc::new(Mutex::new(PendingFrame::default())),
            show_controls: true,
            controls_config: VideoControlsConfig::default(),
            audio_handle: AudioHandle::new(),
            #[cfg(all(feature = "ffmpeg", not(target_os = "android")))]
            audio_thread: None,
            init_thread: None,
            init_receiver: None,
        }
    }

    /// Creates a new video player with wgpu render state.
    ///
    /// This is the preferred way to create a video player as it allows
    /// immediate texture creation and upload.
    pub fn with_wgpu(url: impl Into<String>, wgpu_render_state: &egui_wgpu::RenderState) -> Self {
        // Register video render resources if not already done
        {
            let renderer = wgpu_render_state.renderer.read();
            if renderer
                .callback_resources
                .get::<VideoRenderResources>()
                .is_none()
            {
                drop(renderer);
                VideoRenderResources::register(wgpu_render_state);
            }
        }

        Self {
            state: VideoState::Loading,
            metadata: None,
            frame_queue: Arc::new(FrameQueue::with_default_capacity()),
            decode_thread: None,
            scheduler: FrameScheduler::new(),
            texture: Arc::new(Mutex::new(None)),
            initialized: false,
            url: url.into(),
            autoplay: false,
            loop_playback: false,
            muted: false,
            device: Some(wgpu_render_state.device.clone()),
            queue: Some(wgpu_render_state.queue.clone()),
            pending_frame: Arc::new(Mutex::new(PendingFrame::default())),
            show_controls: true,
            controls_config: VideoControlsConfig::default(),
            audio_handle: AudioHandle::new(),
            #[cfg(all(feature = "ffmpeg", not(target_os = "android")))]
            audio_thread: None,
            init_thread: None,
            init_receiver: None,
        }
    }

    /// Sets whether the video should autoplay.
    pub fn with_autoplay(mut self, autoplay: bool) -> Self {
        self.autoplay = autoplay;
        self
    }

    /// Sets whether the video should loop.
    pub fn with_loop(mut self, loop_playback: bool) -> Self {
        self.loop_playback = loop_playback;
        self
    }

    /// Sets whether audio is muted.
    pub fn with_muted(mut self, muted: bool) -> Self {
        self.muted = muted;
        self.audio_handle.set_muted(muted);
        self
    }

    /// Sets whether to show controls overlay.
    pub fn with_controls(mut self, show_controls: bool) -> Self {
        self.show_controls = show_controls;
        self
    }

    /// Sets the controls configuration.
    pub fn with_controls_config(mut self, config: VideoControlsConfig) -> Self {
        self.controls_config = config;
        self
    }

    /// Starts async initialization of the video player.
    ///
    /// This spawns a background thread to open the video and prepare for playback.
    /// The player will show a loading state until initialization completes.
    pub fn start_async_init(&mut self) {
        if self.initialized || self.init_thread.is_some() {
            return;
        }

        let url = self.url.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        self.init_receiver = Some(rx);

        // Spawn background thread for initialization
        let handle = std::thread::spawn(move || {
            // Open the video with platform-specific decoder
            #[cfg(all(target_os = "macos", feature = "macos-native-video"))]
            let result: Result<Box<dyn VideoDecoderBackend + Send>, VideoError> = {
                match MacOSVideoDecoder::new(&url) {
                    Ok(d) => {
                        tracing::info!("Using macOS VideoToolbox hardware decoder");
                        Ok(Box::new(d) as Box<dyn VideoDecoderBackend + Send>)
                    }
                    Err(e) => {
                        tracing::warn!(
                            "macOS VideoToolbox decoder failed, falling back to FFmpeg: {:?}",
                            e
                        );
                        #[cfg(feature = "ffmpeg")]
                        {
                            FfmpegDecoder::new(&url)
                                .map(|d| Box::new(d) as Box<dyn VideoDecoderBackend + Send>)
                        }
                        #[cfg(not(feature = "ffmpeg"))]
                        {
                            Err(VideoError::DecoderInit(format!(
                                "macOS decoder failed and no FFmpeg fallback available: {:?}",
                                e
                            )))
                        }
                    }
                }
            };

            #[cfg(target_os = "android")]
            let result: Result<Box<dyn VideoDecoderBackend + Send>, VideoError> = {
                tracing::info!("Using Android ExoPlayer decoder for {}", url);
                AndroidVideoDecoder::new(&url)
                    .map(|d| Box::new(d) as Box<dyn VideoDecoderBackend + Send>)
            };

            #[cfg(all(target_os = "linux", feature = "linux-gstreamer-video"))]
            let result: Result<Box<dyn VideoDecoderBackend + Send>, VideoError> = {
                match GStreamerDecoder::new(&url) {
                    Ok(d) => {
                        tracing::info!("Using Linux GStreamer decoder");
                        Ok(Box::new(d) as Box<dyn VideoDecoderBackend + Send>)
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Linux GStreamer decoder failed, falling back to FFmpeg: {:?}",
                            e
                        );
                        #[cfg(feature = "ffmpeg")]
                        {
                            FfmpegDecoder::new(&url)
                                .map(|d| Box::new(d) as Box<dyn VideoDecoderBackend + Send>)
                        }
                        #[cfg(not(feature = "ffmpeg"))]
                        {
                            Err(VideoError::DecoderInit(format!(
                                "GStreamer decoder failed and no FFmpeg fallback available: {:?}",
                                e
                            )))
                        }
                    }
                }
            };

            #[cfg(all(
                feature = "ffmpeg",
                not(any(
                    target_os = "android",
                    all(target_os = "macos", feature = "macos-native-video"),
                    all(target_os = "linux", feature = "linux-gstreamer-video")
                ))
            ))]
            let result: Result<Box<dyn VideoDecoderBackend + Send>, VideoError> =
                FfmpegDecoder::new(&url)
                    .map(|d| Box::new(d) as Box<dyn VideoDecoderBackend + Send>);

            // Fallback when no decoder is available at compile time
            #[cfg(not(any(
                target_os = "android",
                all(target_os = "macos", feature = "macos-native-video"),
                all(target_os = "linux", feature = "linux-gstreamer-video"),
                feature = "ffmpeg"
            )))]
            let result: Result<Box<dyn VideoDecoderBackend + Send>, VideoError> = {
                let _ = &url; // Silence unused variable warning
                Err(VideoError::DecoderInit(
                    "No video decoder available (enable ffmpeg feature)".to_string(),
                ))
            };

            let _ = tx.send(result);
        });

        self.init_thread = Some(handle);
    }

    /// Checks if async initialization is complete and finishes setup.
    /// Returns true if initialization is complete (success or error).
    fn check_init_complete(&mut self) -> bool {
        if self.initialized {
            return true;
        }

        let receiver = match self.init_receiver.as_ref() {
            Some(rx) => rx,
            None => return false,
        };

        // Non-blocking check for initialization result
        match receiver.try_recv() {
            Ok(Ok(decoder)) => {
                // Store metadata
                self.metadata = Some(decoder.metadata().clone());

                // Create and start the decode thread
                let frame_queue = Arc::clone(&self.frame_queue);
                let decode_thread = DecodeThread::new(decoder, frame_queue);

                self.decode_thread = Some(decode_thread);

                // Start audio thread if available (not for GStreamer - it handles audio internally)
                #[cfg(all(
                    feature = "ffmpeg",
                    not(target_os = "android"),
                    not(all(target_os = "linux", feature = "linux-gstreamer-video"))
                ))]
                {
                    if let Some(audio_thread) = AudioThread::new(&self.url) {
                        self.audio_handle = audio_thread.handle();
                        tracing::info!("Audio playback initialized for {}", self.url);
                        self.audio_thread = Some(audio_thread);
                    }
                }

                // For GStreamer, mark audio as available (GStreamer handles playback internally)
                #[cfg(all(target_os = "linux", feature = "linux-gstreamer-video"))]
                {
                    self.audio_handle.set_available(true);
                    tracing::info!("GStreamer audio playback enabled for {}", self.url);
                }

                self.state = VideoState::Ready;
                self.initialized = true;
                self.init_thread = None;
                self.init_receiver = None;

                // Start playback if autoplay is enabled
                if self.autoplay {
                    self.play();
                }

                true
            }
            Ok(Err(e)) => {
                self.state = VideoState::Error(e);
                self.init_thread = None;
                self.init_receiver = None;
                true
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                // Still initializing
                false
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.state = VideoState::Error(VideoError::Generic("Init thread crashed".into()));
                self.init_thread = None;
                self.init_receiver = None;
                true
            }
        }
    }

    /// Initializes the video player synchronously (legacy, causes UI freeze).
    #[allow(dead_code)]
    pub fn initialize(&mut self) -> Result<(), VideoError> {
        if self.initialized {
            return Ok(());
        }

        // Open the video with platform-specific decoder
        #[cfg(all(target_os = "macos", feature = "macos-native-video"))]
        let decoder: Box<dyn VideoDecoderBackend + Send> = {
            match MacOSVideoDecoder::new(&self.url) {
                Ok(d) => {
                    tracing::info!("Using macOS VideoToolbox hardware decoder");
                    Box::new(d)
                }
                Err(e) => {
                    tracing::warn!(
                        "macOS VideoToolbox decoder failed, falling back to FFmpeg: {:?}",
                        e
                    );
                    #[cfg(feature = "ffmpeg")]
                    {
                        Box::new(FfmpegDecoder::new(&self.url)?)
                    }
                    #[cfg(not(feature = "ffmpeg"))]
                    {
                        return Err(VideoError::DecoderInit(format!(
                            "macOS decoder failed and no FFmpeg fallback available: {:?}",
                            e
                        )));
                    }
                }
            }
        };
        #[cfg(target_os = "android")]
        let decoder: Box<dyn VideoDecoderBackend + Send> = {
            tracing::info!("Using Android ExoPlayer decoder for {}", self.url);
            Box::new(AndroidVideoDecoder::new(&self.url)?)
        };

        #[cfg(all(target_os = "linux", feature = "linux-gstreamer-video"))]
        let decoder: Box<dyn VideoDecoderBackend + Send> = {
            match GStreamerDecoder::new(&self.url) {
                Ok(d) => {
                    tracing::info!("Using Linux GStreamer decoder");
                    Box::new(d)
                }
                Err(e) => {
                    tracing::warn!(
                        "Linux GStreamer decoder failed, falling back to FFmpeg: {:?}",
                        e
                    );
                    #[cfg(feature = "ffmpeg")]
                    {
                        Box::new(FfmpegDecoder::new(&self.url)?)
                    }
                    #[cfg(not(feature = "ffmpeg"))]
                    {
                        return Err(VideoError::DecoderInit(format!(
                            "GStreamer decoder failed and no FFmpeg fallback available: {:?}",
                            e
                        )));
                    }
                }
            }
        };

        #[cfg(all(
            feature = "ffmpeg",
            not(any(
                target_os = "android",
                all(target_os = "macos", feature = "macos-native-video"),
                all(target_os = "linux", feature = "linux-gstreamer-video")
            ))
        ))]
        let decoder: Box<dyn VideoDecoderBackend + Send> = Box::new(FfmpegDecoder::new(&self.url)?);

        // Fallback when no decoder is available at compile time
        #[cfg(not(any(
            target_os = "android",
            all(target_os = "macos", feature = "macos-native-video"),
            all(target_os = "linux", feature = "linux-gstreamer-video"),
            feature = "ffmpeg"
        )))]
        {
            Err(VideoError::DecoderInit(
                "No video decoder available (enable ffmpeg feature)".to_string(),
            ))
        }

        // Code that requires a decoder - only compiled when one is available
        #[cfg(any(
            target_os = "android",
            all(target_os = "macos", feature = "macos-native-video"),
            all(target_os = "linux", feature = "linux-gstreamer-video"),
            feature = "ffmpeg"
        ))]
        {
            self.metadata = Some(decoder.metadata().clone());

            let frame_queue = Arc::clone(&self.frame_queue);
            let decode_thread = DecodeThread::new(decoder, frame_queue);

            self.decode_thread = Some(decode_thread);
            self.state = VideoState::Ready;
            self.initialized = true;

            if self.autoplay {
                self.play();
            }

            Ok(())
        }
    }

    /// Starts or resumes playback.
    pub fn play(&mut self) {
        if let Some(ref thread) = self.decode_thread {
            thread.play();
            self.scheduler.start();
            self.state = VideoState::Playing {
                position: self.scheduler.position(),
            };

            // Start audio playback
            #[cfg(all(feature = "ffmpeg", not(target_os = "android")))]
            if let Some(ref audio_thread) = self.audio_thread {
                audio_thread.play();
            }
        }
    }

    /// Pauses playback.
    pub fn pause(&mut self) {
        if let Some(ref thread) = self.decode_thread {
            thread.pause();
            self.scheduler.pause();
            self.state = VideoState::Paused {
                position: self.scheduler.position(),
            };

            // Pause audio playback
            #[cfg(all(feature = "ffmpeg", not(target_os = "android")))]
            if let Some(ref audio_thread) = self.audio_thread {
                audio_thread.pause();
            }
        }
    }

    /// Toggles between play and pause.
    pub fn toggle_playback(&mut self) {
        match self.state {
            VideoState::Playing { .. } => self.pause(),
            VideoState::Paused { .. } | VideoState::Ready | VideoState::Ended => self.play(),
            _ => {}
        }
    }

    /// Seeks to a specific position.
    pub fn seek(&mut self, position: Duration) {
        // Log seek origin for debugging unexpected seeks
        if position == Duration::ZERO {
            tracing::debug!(
                "Seek to ZERO requested from state={:?}, loop_playback={}",
                self.state,
                self.loop_playback
            );
        }

        // Clear EOS immediately to prevent loop_playback from racing with this seek
        self.frame_queue.clear_eos();

        if let Some(ref thread) = self.decode_thread {
            thread.seek(position);
            self.scheduler.seek(position);

            // Seek audio
            #[cfg(all(feature = "ffmpeg", not(target_os = "android")))]
            if let Some(ref audio_thread) = self.audio_thread {
                audio_thread.seek(position);
            }

            // Update state with new position
            match self.state {
                VideoState::Playing { .. } => {
                    self.state = VideoState::Playing { position };
                }
                VideoState::Paused { .. } => {
                    self.state = VideoState::Paused { position };
                }
                VideoState::Ended => {
                    self.state = VideoState::Paused { position };
                }
                _ => {}
            }
        }
    }

    /// Returns the current playback state.
    pub fn state(&self) -> &VideoState {
        &self.state
    }

    /// Returns the video metadata if available.
    pub fn metadata(&self) -> Option<&VideoMetadata> {
        self.metadata.as_ref()
    }

    /// Returns the current playback position.
    pub fn position(&self) -> Duration {
        self.scheduler.position()
    }

    /// Returns the video duration if known.
    pub fn duration(&self) -> Option<Duration> {
        // First try to get duration from the decode thread (updated dynamically, e.g., from ExoPlayer callbacks)
        if let Some(ref thread) = self.decode_thread {
            if let Some(dur) = thread.duration() {
                return Some(dur);
            }
        }
        // Fall back to metadata duration
        self.metadata.as_ref().and_then(|m| m.duration)
    }

    /// Returns the video dimensions (width, height).
    ///
    /// This checks the decode thread first for dynamically updated dimensions
    /// (e.g., from ExoPlayer callbacks on Android), then falls back to metadata.
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        // First try to get dimensions from the decode thread
        if let Some(ref thread) = self.decode_thread {
            if let Some(dims) = thread.dimensions() {
                return Some(dims);
            }
        }
        // Fall back to metadata dimensions
        self.metadata.as_ref().map(|m| (m.width, m.height))
    }

    /// Returns true if the video is currently playing.
    pub fn is_playing(&self) -> bool {
        self.scheduler.is_playing()
    }

    /// Returns the current buffering percentage (0-100).
    ///
    /// For network streams, this indicates how much data has been buffered.
    /// Returns 100 for local files or when buffering state is unknown.
    pub fn buffering_percent(&self) -> i32 {
        if let Some(ref thread) = self.decode_thread {
            thread.buffering_percent()
        } else {
            100 // Assume buffered if no decode thread yet
        }
    }

    /// Returns the audio handle for volume/mute control.
    pub fn audio_handle(&self) -> &AudioHandle {
        &self.audio_handle
    }

    /// Returns the current volume (0-100).
    pub fn volume(&self) -> u32 {
        self.audio_handle.volume()
    }

    /// Sets the volume (0-100).
    pub fn set_volume(&mut self, volume: u32) {
        self.audio_handle.set_volume(volume);

        // On Android and Linux with GStreamer, audio is controlled through the decode thread
        #[cfg(any(
            target_os = "android",
            all(target_os = "linux", feature = "linux-gstreamer-video")
        ))]
        if let Some(ref decode_thread) = self.decode_thread {
            // Convert 0-100 to 0.0-1.0
            decode_thread.set_volume(volume as f32 / 100.0);
        }
    }

    /// Returns whether audio is muted.
    pub fn is_muted(&self) -> bool {
        self.audio_handle.is_muted()
    }

    /// Toggles the mute state.
    pub fn toggle_mute(&mut self) {
        self.audio_handle.toggle_mute();
        self.muted = self.audio_handle.is_muted();

        // On Android and Linux with GStreamer, audio is controlled through the decode thread
        #[cfg(any(
            target_os = "android",
            all(target_os = "linux", feature = "linux-gstreamer-video")
        ))]
        if let Some(ref decode_thread) = self.decode_thread {
            decode_thread.set_muted(self.muted);
        }
    }

    /// Shows the video player widget.
    ///
    /// This renders the current video frame and handles user interactions.
    pub fn show(&mut self, ui: &mut Ui, size: Vec2) -> VideoPlayerResponse {
        // Allocate space for the video
        let (rect, response) = ui.allocate_exact_size(size, Sense::click());

        // Start async initialization if needed
        if !self.initialized && self.init_thread.is_none() {
            self.start_async_init();
        }

        // Check if async init is complete
        if !self.initialized {
            self.check_init_complete();
        }

        // Update frame if playback requested (even if buffering), or try to get preview frame when Ready/Paused
        if self.scheduler.is_playback_requested() {
            self.update_frame();
        } else if matches!(self.state, VideoState::Ready | VideoState::Paused { .. }) {
            // Try to get preview frame from queue (non-blocking)
            self.try_get_preview_frame();
        }

        // Check for end of stream
        if self.frame_queue.is_eos() && self.frame_queue.is_empty() {
            tracing::debug!(
                "EOS condition met: loop_playback={}, state={:?}",
                self.loop_playback,
                self.state
            );
            if self.loop_playback {
                tracing::debug!(
                    "Loop triggered: eos={}, empty={}, state={:?}",
                    self.frame_queue.is_eos(),
                    self.frame_queue.is_empty(),
                    self.state
                );
                self.seek(Duration::ZERO);
                self.play();
            } else {
                self.state = VideoState::Ended;
            }
        }

        // Render the video frame, loading state, or error
        match &self.state {
            VideoState::Error(err) => {
                self.render_error(ui, rect, err);
            }
            VideoState::Loading => {
                // Show loading indicator while initializing
                self.render_loading(ui, rect);
            }
            _ => {
                self.render(ui, rect);

                // Show buffering overlay if buffering < 100%
                let buffering = self.buffering_percent();
                if buffering < 100 {
                    self.render_buffering_overlay(ui, rect, buffering);
                }
            }
        }

        // Show controls overlay and handle interactions
        let mut state_changed = false;
        let mut controls_response = VideoControlsResponse::default();

        if self.show_controls {
            let controls = VideoControls::new(&self.state, self.position(), self.duration())
                .with_config(self.controls_config.clone())
                .with_muted(self.audio_handle.is_muted());
            controls_response = controls.show(ui, rect);

            // Handle control interactions
            if controls_response.toggle_playback {
                self.toggle_playback();
                state_changed = true;
            }

            if let Some(seek_pos) = controls_response.seek_to {
                self.seek(seek_pos);
                state_changed = true;
            }

            if controls_response.toggle_mute {
                self.toggle_mute();
                state_changed = true;
            }
        }

        // Handle click on video area to toggle playback (only if not handled by controls)
        // Ignore click if any control was interacted with
        let control_was_used = controls_response.toggle_playback
            || controls_response.toggle_mute
            || controls_response.toggle_fullscreen
            || controls_response.seek_to.is_some()
            || controls_response.is_seeking;
        let clicked = response.clicked() && !control_was_used;
        if clicked {
            self.toggle_playback();
            state_changed = true;
        }

        // Request repaint if playing/buffering, loading, initializing, or have pending frame
        let is_initializing = self.init_thread.is_some();
        let has_pending_frame = self.pending_frame.lock().unwrap().frame.is_some();
        let is_buffering = self.buffering_percent() < 100;
        if self.scheduler.is_playback_requested()
            || is_initializing
            || has_pending_frame
            || is_buffering
            || matches!(
                self.state,
                VideoState::Loading | VideoState::Buffering { .. }
            )
        {
            ui.ctx().request_repaint();
        }

        VideoPlayerResponse {
            response,
            clicked,
            state_changed,
            toggle_fullscreen: controls_response.toggle_fullscreen,
        }
    }

    /// Updates the current frame from the decode queue.
    ///
    /// This stores the frame in pending_frame for the render callback to process.
    /// The actual texture creation and upload happens in the prepare callback
    /// which has access to VideoRenderResources.
    fn update_frame(&mut self) {
        // Get the next frame to display
        if let Some(frame) = self.scheduler.get_next_frame(&self.frame_queue) {
            // Update state with current position
            self.state = VideoState::Playing {
                position: frame.pts,
            };

            // Check if texture needs to be recreated
            let texture_guard = self.texture.lock().unwrap();
            let (width, height) = frame.dimensions();
            let format = frame.frame.format();

            let needs_recreate = texture_guard
                .as_ref()
                .map(|t| t.dimensions() != (width, height) || t.format() != format)
                .unwrap_or(true);
            drop(texture_guard);

            // Store the frame for the render callback to process
            if let Some(cpu_frame) = frame.frame.as_cpu() {
                let mut pending = self.pending_frame.lock().unwrap();
                pending.frame = Some(cpu_frame.clone());
                pending.needs_recreate = needs_recreate;
            }
        }
    }

    /// Tries to get a preview frame from the queue without advancing playback.
    ///
    /// This is used to display the first frame before playback starts.
    fn try_get_preview_frame(&mut self) {
        // Only try if we don't already have a pending frame
        if self.pending_frame.lock().unwrap().frame.is_some() {
            return;
        }

        // Peek at the queue - don't pop so we can play from the beginning
        let Some(frame) = self.frame_queue.peek() else {
            return;
        };

        let Some(cpu_frame) = frame.frame.as_cpu() else {
            return;
        };

        // Check if texture needs to be recreated
        let (width, height) = frame.dimensions();
        let format = frame.frame.format();
        let needs_recreate = self
            .texture
            .lock()
            .unwrap()
            .as_ref()
            .map(|t| t.dimensions() != (width, height) || t.format() != format)
            .unwrap_or(true);

        // Store the frame for the render callback to process
        let mut pending = self.pending_frame.lock().unwrap();
        pending.frame = Some(cpu_frame.clone());
        pending.needs_recreate = needs_recreate;
    }

    /// Sets the wgpu render state for this player.
    ///
    /// This must be called before the player can render frames if the player
    /// was created with `new()` instead of `with_wgpu()`.
    pub fn set_wgpu_state(&mut self, wgpu_render_state: &egui_wgpu::RenderState) {
        self.device = Some(wgpu_render_state.device.clone());
        self.queue = Some(wgpu_render_state.queue.clone());

        // Register video render resources if not already done
        {
            let renderer = wgpu_render_state.renderer.read();
            if renderer
                .callback_resources
                .get::<VideoRenderResources>()
                .is_none()
            {
                drop(renderer);
                VideoRenderResources::register(wgpu_render_state);
            }
        }
    }

    /// Renders an error overlay when video fails to load.
    fn render_error(&self, ui: &mut Ui, rect: egui::Rect, error: &VideoError) {
        use egui::{Align2, Color32, CornerRadius, FontId};

        // Draw dark background
        ui.painter()
            .rect_filled(rect, CornerRadius::ZERO, Color32::from_rgb(30, 30, 30));

        // Draw error icon (X)
        let center = rect.center();
        let icon_size = 40.0;
        let stroke = egui::Stroke::new(4.0, Color32::from_rgb(255, 100, 100));

        ui.painter().line_segment(
            [
                egui::pos2(center.x - icon_size / 2.0, center.y - icon_size / 2.0),
                egui::pos2(center.x + icon_size / 2.0, center.y + icon_size / 2.0),
            ],
            stroke,
        );
        ui.painter().line_segment(
            [
                egui::pos2(center.x + icon_size / 2.0, center.y - icon_size / 2.0),
                egui::pos2(center.x - icon_size / 2.0, center.y + icon_size / 2.0),
            ],
            stroke,
        );

        // Draw error message
        let error_text = format!("Video Error: {error}");
        ui.painter().text(
            egui::pos2(center.x, center.y + icon_size + 10.0),
            Align2::CENTER_TOP,
            error_text,
            FontId::proportional(12.0),
            Color32::from_rgb(200, 200, 200),
        );
    }

    /// Renders a loading indicator while video is initializing.
    fn render_loading(&self, ui: &mut Ui, rect: egui::Rect) {
        use egui::{Align2, Color32, CornerRadius, FontId};

        // Draw dark background
        ui.painter()
            .rect_filled(rect, CornerRadius::ZERO, Color32::from_rgb(30, 30, 30));

        let center = rect.center();

        // Draw animated loading spinner
        let time = ui.input(|i| i.time);
        let spinner_radius = 20.0;
        let num_dots = 8;
        let dot_radius = 4.0;

        for i in 0..num_dots {
            let angle = (i as f64 / num_dots as f64) * std::f64::consts::TAU + time * 2.0;
            let x = center.x + (angle.cos() * spinner_radius as f64) as f32;
            let y = center.y + (angle.sin() * spinner_radius as f64) as f32;

            // Fade dots based on position in rotation
            let alpha = ((i as f64 / num_dots as f64 + time * 2.0).fract() * 255.0) as u8;
            let color = Color32::from_rgba_unmultiplied(200, 200, 200, alpha);

            ui.painter()
                .circle_filled(egui::pos2(x, y), dot_radius, color);
        }

        // Draw "Loading..." text
        ui.painter().text(
            egui::pos2(center.x, center.y + spinner_radius + 20.0),
            Align2::CENTER_TOP,
            "Loading...",
            FontId::proportional(12.0),
            Color32::from_rgb(200, 200, 200),
        );
    }

    /// Renders a buffering progress indicator overlay.
    fn render_buffering_overlay(&self, ui: &mut Ui, rect: egui::Rect, percent: i32) {
        use egui::{Align2, Color32, CornerRadius, FontId, Stroke};

        let center = rect.center();

        // Semi-transparent dark overlay
        ui.painter().rect_filled(
            rect,
            CornerRadius::ZERO,
            Color32::from_rgba_unmultiplied(0, 0, 0, 160),
        );

        // Progress ring parameters
        let ring_radius = 30.0;
        let ring_thickness = 4.0;

        // Draw background ring (dark gray)
        ui.painter().circle_stroke(
            center,
            ring_radius,
            Stroke::new(ring_thickness, Color32::from_rgb(60, 60, 60)),
        );

        // Draw progress arc
        let progress = percent as f32 / 100.0;
        let num_segments = 32;
        let segments_to_draw = (num_segments as f32 * progress) as usize;

        if segments_to_draw > 0 {
            let start_angle = -std::f32::consts::FRAC_PI_2; // Start from top

            for i in 0..segments_to_draw {
                let angle1 = start_angle + (i as f32 / num_segments as f32) * std::f32::consts::TAU;
                let angle2 =
                    start_angle + ((i + 1) as f32 / num_segments as f32) * std::f32::consts::TAU;

                let p1 = egui::pos2(
                    center.x + angle1.cos() * ring_radius,
                    center.y + angle1.sin() * ring_radius,
                );
                let p2 = egui::pos2(
                    center.x + angle2.cos() * ring_radius,
                    center.y + angle2.sin() * ring_radius,
                );

                ui.painter().line_segment(
                    [p1, p2],
                    Stroke::new(ring_thickness, Color32::from_rgb(100, 180, 255)),
                );
            }
        }

        // Draw percentage text in center
        ui.painter().text(
            center,
            Align2::CENTER_CENTER,
            format!("{percent}%"),
            FontId::proportional(14.0),
            Color32::WHITE,
        );

        // Draw "Buffering" text below
        ui.painter().text(
            egui::pos2(center.x, center.y + ring_radius + 15.0),
            Align2::CENTER_TOP,
            "Buffering",
            FontId::proportional(12.0),
            Color32::from_rgb(200, 200, 200),
        );
    }

    /// Renders the video frame.
    fn render(&self, ui: &mut Ui, rect: egui::Rect) {
        // Get the current pixel format from pending frame or existing texture
        let format = {
            let pending = self.pending_frame.lock().unwrap();
            if let Some(ref frame) = pending.frame {
                frame.format
            } else {
                self.texture
                    .lock()
                    .unwrap()
                    .as_ref()
                    .map(|t| t.format())
                    .unwrap_or(PixelFormat::Yuv420p)
            }
        };

        // Create render callback
        let callback = VideoRenderCallback {
            texture: Arc::clone(&self.texture),
            pending_frame: Arc::clone(&self.pending_frame),
            format,
            rect,
        };

        // Add paint callback
        ui.painter()
            .add(egui_wgpu::Callback::new_paint_callback(rect, callback));
    }
}

impl Drop for VideoPlayer {
    fn drop(&mut self) {
        // Stop the decode thread
        if let Some(ref thread) = self.decode_thread {
            thread.stop();
        }

        // Stop the audio thread
        #[cfg(all(feature = "ffmpeg", not(target_os = "android")))]
        if let Some(ref audio_thread) = self.audio_thread {
            audio_thread.stop();
        }
    }
}

/// Response from showing a video player widget.
pub struct VideoPlayerResponse {
    /// The egui response from the widget allocation
    pub response: Response,
    /// Whether the video was clicked
    pub clicked: bool,
    /// Whether the playback state changed
    pub state_changed: bool,
    /// Whether fullscreen was toggled
    pub toggle_fullscreen: bool,
}

/// Extension trait for easily adding video players to egui.
pub trait VideoPlayerExt {
    /// Shows a video player for the given URL.
    fn video_player(&mut self, player: &mut VideoPlayer, size: Vec2) -> VideoPlayerResponse;
}

impl VideoPlayerExt for Ui {
    fn video_player(&mut self, player: &mut VideoPlayer, size: Vec2) -> VideoPlayerResponse {
        player.show(self, size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_player_creation() {
        let player = VideoPlayer::new("test.mp4");
        assert!(matches!(player.state(), VideoState::Loading));
        assert!(!player.is_playing());
    }

    #[test]
    fn test_video_player_autoplay() {
        let player = VideoPlayer::new("test.mp4").with_autoplay(true);
        assert!(player.autoplay);
    }
}
