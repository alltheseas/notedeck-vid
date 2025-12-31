//! Video playback core types and state machine.
//!
//! This module provides the foundational types for hardware-accelerated video
//! playback across all platforms (macOS, Windows, Linux, Android).

use std::sync::Arc;
use std::time::Duration;

/// Represents the current state of video playback.
#[derive(Debug, Clone, PartialEq)]
pub enum VideoState {
    /// Video is being loaded/initialized
    Loading,
    /// Video is ready to play but not started
    Ready,
    /// Video is actively playing
    Playing {
        /// Current playback position
        position: Duration,
    },
    /// Video is paused
    Paused {
        /// Position when paused
        position: Duration,
    },
    /// Video is buffering (network stream)
    Buffering {
        /// Position where buffering started
        position: Duration,
    },
    /// An error occurred
    Error(VideoError),
    /// Playback completed
    Ended,
}

impl VideoState {
    /// Returns the current position if available.
    pub fn position(&self) -> Option<Duration> {
        match self {
            VideoState::Playing { position } => Some(*position),
            VideoState::Paused { position } => Some(*position),
            VideoState::Buffering { position } => Some(*position),
            _ => None,
        }
    }

    /// Returns true if video is currently playing.
    pub fn is_playing(&self) -> bool {
        matches!(self, VideoState::Playing { .. })
    }

    /// Returns true if video can be played (Ready or Paused).
    pub fn can_play(&self) -> bool {
        matches!(
            self,
            VideoState::Ready | VideoState::Paused { .. } | VideoState::Ended
        )
    }
}

/// Errors that can occur during video playback.
#[derive(Debug, Clone, PartialEq)]
pub enum VideoError {
    /// Failed to open the video source
    OpenFailed(String),
    /// Decoder initialization failed
    DecoderInit(String),
    /// Frame decoding error
    DecodeFailed(String),
    /// Seek operation failed
    SeekFailed(String),
    /// Unsupported codec or format
    UnsupportedFormat(String),
    /// Network error (for streaming)
    Network(String),
    /// Generic error
    Generic(String),
}

impl std::fmt::Display for VideoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VideoError::OpenFailed(msg) => write!(f, "Failed to open video: {}", msg),
            VideoError::DecoderInit(msg) => write!(f, "Decoder initialization failed: {}", msg),
            VideoError::DecodeFailed(msg) => write!(f, "Frame decode failed: {}", msg),
            VideoError::SeekFailed(msg) => write!(f, "Seek failed: {}", msg),
            VideoError::UnsupportedFormat(msg) => write!(f, "Unsupported format: {}", msg),
            VideoError::Network(msg) => write!(f, "Network error: {}", msg),
            VideoError::Generic(msg) => write!(f, "Video error: {}", msg),
        }
    }
}

impl std::error::Error for VideoError {}

/// Pixel format for decoded video frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// YUV 4:2:0 planar (most common video format)
    Yuv420p,
    /// NV12 (Y plane + interleaved UV, common for hardware decoders)
    Nv12,
    /// RGB 24-bit
    Rgb24,
    /// RGBA 32-bit
    Rgba,
    /// BGRA 32-bit (common on some platforms)
    Bgra,
}

impl PixelFormat {
    /// Returns the number of planes for this format.
    pub fn num_planes(&self) -> usize {
        match self {
            PixelFormat::Yuv420p => 3,
            PixelFormat::Nv12 => 2,
            PixelFormat::Rgb24 | PixelFormat::Rgba | PixelFormat::Bgra => 1,
        }
    }

    /// Returns true if this is a YUV-based format requiring conversion.
    pub fn is_yuv(&self) -> bool {
        matches!(self, PixelFormat::Yuv420p | PixelFormat::Nv12)
    }
}

/// A single plane of pixel data.
#[derive(Debug, Clone)]
pub struct Plane {
    /// Raw pixel data
    pub data: Vec<u8>,
    /// Stride (bytes per row, may include padding)
    pub stride: usize,
}

/// A decoded video frame with CPU-accessible pixel data.
#[derive(Debug, Clone)]
pub struct CpuFrame {
    /// Pixel format of the frame
    pub format: PixelFormat,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Pixel data planes
    pub planes: Vec<Plane>,
}

impl CpuFrame {
    /// Creates a new CpuFrame with the given parameters.
    pub fn new(format: PixelFormat, width: u32, height: u32, planes: Vec<Plane>) -> Self {
        Self {
            format,
            width,
            height,
            planes,
        }
    }

    /// Returns the Y plane for YUV formats, or the single plane for RGB formats.
    pub fn plane(&self, index: usize) -> Option<&Plane> {
        self.planes.get(index)
    }
}

/// A decoded video frame, either CPU-accessible or platform-specific GPU surface.
#[derive(Debug, Clone)]
pub enum DecodedFrame {
    /// CPU-accessible frame data (works on all platforms)
    Cpu(CpuFrame),
    // Platform-specific GPU surfaces for zero-copy rendering
    // These are feature-gated and will be implemented later
    // #[cfg(target_os = "macos")]
    // VideoToolbox(CVPixelBufferRef),
    // #[cfg(target_os = "windows")]
    // D3D11(ID3D11Texture2D),
    // #[cfg(target_os = "linux")]
    // VaSurface(VASurfaceID),
    // #[cfg(target_os = "android")]
    // HardwareBuffer(AHardwareBuffer),
}

impl DecodedFrame {
    /// Returns the frame dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        match self {
            DecodedFrame::Cpu(frame) => (frame.width, frame.height),
        }
    }

    /// Returns the pixel format.
    pub fn format(&self) -> PixelFormat {
        match self {
            DecodedFrame::Cpu(frame) => frame.format,
        }
    }

    /// Attempts to get a reference to the CPU frame data.
    pub fn as_cpu(&self) -> Option<&CpuFrame> {
        match self {
            DecodedFrame::Cpu(frame) => Some(frame),
        }
    }
}

/// A video frame with presentation timestamp.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Presentation timestamp (when this frame should be displayed)
    pub pts: Duration,
    /// The decoded frame data
    pub frame: DecodedFrame,
}

impl VideoFrame {
    /// Creates a new VideoFrame.
    pub fn new(pts: Duration, frame: DecodedFrame) -> Self {
        Self { pts, frame }
    }

    /// Returns the frame dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        self.frame.dimensions()
    }
}

/// Metadata about a video stream.
#[derive(Debug, Clone)]
pub struct VideoMetadata {
    /// Video width in pixels
    pub width: u32,
    /// Video height in pixels
    pub height: u32,
    /// Duration of the video (if known)
    pub duration: Option<Duration>,
    /// Frame rate (frames per second)
    pub frame_rate: f32,
    /// Codec name
    pub codec: String,
    /// Pixel aspect ratio (1.0 for square pixels)
    pub pixel_aspect_ratio: f32,
}

impl VideoMetadata {
    /// Returns the aspect ratio of the video.
    pub fn aspect_ratio(&self) -> f32 {
        if self.height == 0 {
            return 1.0;
        }
        (self.width as f32 / self.height as f32) * self.pixel_aspect_ratio
    }

    /// Returns the frame duration based on frame rate.
    pub fn frame_duration(&self) -> Duration {
        if self.frame_rate <= 0.0 {
            return Duration::from_millis(33); // Default to ~30fps
        }
        Duration::from_secs_f64(1.0 / self.frame_rate as f64)
    }
}

/// Hardware acceleration type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwAccelType {
    /// No hardware acceleration (software decode)
    None,
    /// macOS VideoToolbox
    VideoToolbox,
    /// Linux VAAPI
    Vaapi,
    /// Linux VDPAU (NVIDIA legacy)
    Vdpau,
    /// Windows D3D11VA
    D3d11va,
    /// Windows DXVA2
    Dxva2,
    /// Android MediaCodec
    MediaCodec,
}

impl HwAccelType {
    /// Returns the best hardware acceleration for the current platform.
    #[cfg(target_os = "macos")]
    pub fn platform_default() -> Self {
        HwAccelType::VideoToolbox
    }

    #[cfg(target_os = "windows")]
    pub fn platform_default() -> Self {
        HwAccelType::D3d11va
    }

    #[cfg(target_os = "linux")]
    pub fn platform_default() -> Self {
        // VAAPI is more widely supported
        HwAccelType::Vaapi
    }

    #[cfg(target_os = "android")]
    pub fn platform_default() -> Self {
        HwAccelType::MediaCodec
    }

    #[cfg(not(any(
        target_os = "macos",
        target_os = "windows",
        target_os = "linux",
        target_os = "android"
    )))]
    pub fn platform_default() -> Self {
        HwAccelType::None
    }
}

/// Trait for video decoder backends.
///
/// This trait abstracts the platform-specific video decoding implementations,
/// allowing the same video player code to work with FFmpeg on desktop and
/// ExoPlayer on Android.
pub trait VideoDecoderBackend: Send {
    /// Opens a video from a URL or file path.
    fn open(url: &str) -> Result<Self, VideoError>
    where
        Self: Sized;

    /// Decodes and returns the next video frame, or None if no more frames.
    fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError>;

    /// Seeks to a specific position in the video.
    fn seek(&mut self, position: Duration) -> Result<(), VideoError>;

    /// Returns the video metadata.
    fn metadata(&self) -> &VideoMetadata;

    /// Pauses playback.
    ///
    /// For decoders with their own playback control (like ExoPlayer on Android),
    /// this actually pauses the underlying player. For decoders like FFmpeg that
    /// don't have playback state, this is a no-op (the decode thread handles pausing).
    fn pause(&mut self) -> Result<(), VideoError> {
        Ok(()) // Default no-op for decoders without playback control
    }

    /// Resumes playback.
    ///
    /// For decoders with their own playback control (like ExoPlayer on Android),
    /// this actually resumes the underlying player. For decoders like FFmpeg that
    /// don't have playback state, this is a no-op (the decode thread handles resuming).
    fn resume(&mut self) -> Result<(), VideoError> {
        Ok(()) // Default no-op for decoders without playback control
    }

    /// Returns the total duration if known.
    fn duration(&self) -> Option<Duration> {
        self.metadata().duration
    }

    /// Sets the muted state for audio playback.
    ///
    /// For decoders with integrated audio (like ExoPlayer on Android),
    /// this mutes/unmutes the audio. For decoders without integrated audio,
    /// this is a no-op (audio is handled separately).
    fn set_muted(&mut self, _muted: bool) -> Result<(), VideoError> {
        Ok(()) // Default no-op
    }

    /// Sets the volume for audio playback.
    ///
    /// For decoders with integrated audio (like ExoPlayer on Android),
    /// this sets the volume. For decoders without integrated audio,
    /// this is a no-op (audio is handled separately).
    fn set_volume(&mut self, _volume: f32) -> Result<(), VideoError> {
        Ok(()) // Default no-op
    }

    /// Returns the video dimensions.
    fn dimensions(&self) -> (u32, u32) {
        let meta = self.metadata();
        (meta.width, meta.height)
    }

    /// Returns true if the decoder has reached end of stream.
    ///
    /// This is more reliable than counting None results from decode_next(),
    /// as it reflects the actual decoder state rather than buffering timeouts.
    fn is_eof(&self) -> bool {
        false // Default - most decoders signal EOF via decode_next returning None
    }

    /// Returns the current buffering percentage (0-100).
    ///
    /// For network streams, this indicates how much data has been buffered.
    /// Returns 100 for local files or when buffering state is unknown.
    fn buffering_percent(&self) -> i32 {
        100 // Default - assume fully buffered
    }

    /// Returns the current hardware acceleration type.
    fn hw_accel_type(&self) -> HwAccelType {
        HwAccelType::None
    }
}

/// Implementation for boxed trait objects to enable decoder fallback patterns.
impl VideoDecoderBackend for Box<dyn VideoDecoderBackend + Send> {
    fn open(_url: &str) -> Result<Self, VideoError>
    where
        Self: Sized,
    {
        // Not supported on boxed trait objects - use concrete types for open
        Err(VideoError::DecoderInit(
            "Cannot call open() on boxed trait object".to_string(),
        ))
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        (**self).decode_next()
    }

    fn seek(&mut self, position: Duration) -> Result<(), VideoError> {
        (**self).seek(position)
    }

    fn metadata(&self) -> &VideoMetadata {
        (**self).metadata()
    }

    fn pause(&mut self) -> Result<(), VideoError> {
        (**self).pause()
    }

    fn resume(&mut self) -> Result<(), VideoError> {
        (**self).resume()
    }

    fn set_muted(&mut self, muted: bool) -> Result<(), VideoError> {
        (**self).set_muted(muted)
    }

    fn set_volume(&mut self, volume: f32) -> Result<(), VideoError> {
        (**self).set_volume(volume)
    }

    fn duration(&self) -> Option<Duration> {
        (**self).duration()
    }

    fn dimensions(&self) -> (u32, u32) {
        (**self).dimensions()
    }

    fn is_eof(&self) -> bool {
        (**self).is_eof()
    }

    fn buffering_percent(&self) -> i32 {
        (**self).buffering_percent()
    }

    fn hw_accel_type(&self) -> HwAccelType {
        (**self).hw_accel_type()
    }
}

/// Handle to a video player instance for use in the job system.
#[derive(Clone)]
pub struct VideoPlayerHandle {
    /// Shared reference to the internal player state
    inner: Arc<std::sync::Mutex<VideoPlayerInner>>,
}

struct VideoPlayerInner {
    state: VideoState,
    metadata: Option<VideoMetadata>,
}

impl VideoPlayerHandle {
    /// Creates a new video player handle.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(std::sync::Mutex::new(VideoPlayerInner {
                state: VideoState::Loading,
                metadata: None,
            })),
        }
    }

    /// Returns the current playback state.
    pub fn state(&self) -> VideoState {
        self.inner.lock().unwrap().state.clone()
    }

    /// Sets the playback state.
    pub fn set_state(&self, state: VideoState) {
        self.inner.lock().unwrap().state = state;
    }

    /// Returns the video metadata if available.
    pub fn metadata(&self) -> Option<VideoMetadata> {
        self.inner.lock().unwrap().metadata.clone()
    }

    /// Sets the video metadata.
    pub fn set_metadata(&self, metadata: VideoMetadata) {
        self.inner.lock().unwrap().metadata = Some(metadata);
    }
}

impl Default for VideoPlayerHandle {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_state_position() {
        let playing = VideoState::Playing {
            position: Duration::from_secs(10),
        };
        assert_eq!(playing.position(), Some(Duration::from_secs(10)));

        let loading = VideoState::Loading;
        assert_eq!(loading.position(), None);
    }

    #[test]
    fn test_video_metadata_aspect_ratio() {
        let meta = VideoMetadata {
            width: 1920,
            height: 1080,
            duration: Some(Duration::from_secs(120)),
            frame_rate: 30.0,
            codec: "h264".to_string(),
            pixel_aspect_ratio: 1.0,
        };
        assert!((meta.aspect_ratio() - 1.777).abs() < 0.01);
    }

    #[test]
    fn test_pixel_format_planes() {
        assert_eq!(PixelFormat::Yuv420p.num_planes(), 3);
        assert_eq!(PixelFormat::Nv12.num_planes(), 2);
        assert_eq!(PixelFormat::Rgba.num_planes(), 1);
    }
}
