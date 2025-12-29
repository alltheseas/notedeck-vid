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

use super::frame_queue::{DecodeThread, FrameQueue, FrameScheduler};
use super::video::{CpuFrame, PixelFormat, VideoDecoderBackend, VideoError, VideoMetadata, VideoState};
use super::audio::AudioHandle;
use super::video_controls::{VideoControls, VideoControlsConfig, VideoControlsResponse};
#[cfg(not(target_os = "macos"))]
use super::video_decoder::FfmpegDecoder;
#[cfg(target_os = "macos")]
use super::macos_video::MacOSVideoDecoder;
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
        }
    }

    /// Creates a new video player with wgpu render state.
    ///
    /// This is the preferred way to create a video player as it allows
    /// immediate texture creation and upload.
    pub fn with_wgpu(
        url: impl Into<String>,
        wgpu_render_state: &egui_wgpu::RenderState,
    ) -> Self {
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

    /// Initializes the video player.
    ///
    /// This opens the video file, starts the decode thread, and prepares
    /// for playback. Must be called before `show()`.
    pub fn initialize(&mut self) -> Result<(), VideoError> {
        if self.initialized {
            return Ok(());
        }

        // Open the video with platform-specific decoder
        #[cfg(target_os = "macos")]
        let decoder = MacOSVideoDecoder::new(&self.url)?;
        #[cfg(not(target_os = "macos"))]
        let decoder = FfmpegDecoder::new(&self.url)?;

        // Store metadata
        self.metadata = Some(decoder.metadata().clone());

        // Create and start the decode thread
        let frame_queue = Arc::clone(&self.frame_queue);
        let decode_thread = DecodeThread::new(decoder, frame_queue);

        self.decode_thread = Some(decode_thread);
        self.state = VideoState::Ready;
        self.initialized = true;

        // Start playback if autoplay is enabled
        if self.autoplay {
            self.play();
        }

        Ok(())
    }

    /// Starts or resumes playback.
    pub fn play(&mut self) {
        if let Some(ref thread) = self.decode_thread {
            thread.play();
            self.scheduler.start();
            self.state = VideoState::Playing {
                position: self.scheduler.position(),
            };
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
        if let Some(ref thread) = self.decode_thread {
            thread.seek(position);
            self.scheduler.seek(position);

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
        self.metadata.as_ref().and_then(|m| m.duration)
    }

    /// Returns true if the video is currently playing.
    pub fn is_playing(&self) -> bool {
        self.scheduler.is_playing()
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
    }

    /// Returns whether audio is muted.
    pub fn is_muted(&self) -> bool {
        self.audio_handle.is_muted()
    }

    /// Toggles the mute state.
    pub fn toggle_mute(&mut self) {
        self.audio_handle.toggle_mute();
    }

    /// Shows the video player widget.
    ///
    /// This renders the current video frame and handles user interactions.
    pub fn show(&mut self, ui: &mut Ui, size: Vec2) -> VideoPlayerResponse {
        // Allocate space for the video
        let (rect, response) = ui.allocate_exact_size(size, Sense::click());

        // Initialize if needed
        if !self.initialized {
            if let Err(e) = self.initialize() {
                self.state = VideoState::Error(e);
            }
        }

        // Update frame if playing
        if self.scheduler.is_playing() {
            self.update_frame();
        }

        // Check for end of stream
        if self.frame_queue.is_eos() && self.frame_queue.is_empty() {
            if self.loop_playback {
                self.seek(Duration::ZERO);
                self.play();
            } else {
                self.state = VideoState::Ended;
            }
        }

        // Render the video frame or error state
        match &self.state {
            VideoState::Error(err) => {
                // Draw error overlay
                self.render_error(ui, rect, err);
            }
            _ => {
                self.render(ui, rect);
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

        // Request repaint if playing or loading
        if self.scheduler.is_playing() || matches!(self.state, VideoState::Loading | VideoState::Buffering { .. }) {
            ui.ctx().request_repaint();
        }

        VideoPlayerResponse {
            response,
            clicked,
            state_changed,
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
            self.state = VideoState::Playing { position: frame.pts };

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
        use egui::{Align2, Color32, FontId, Rounding};

        // Draw dark background
        ui.painter().rect_filled(
            rect,
            Rounding::ZERO,
            Color32::from_rgb(30, 30, 30),
        );

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
        let error_text = format!("Video Error: {}", error);
        ui.painter().text(
            egui::pos2(center.x, center.y + icon_size + 10.0),
            Align2::CENTER_TOP,
            error_text,
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
