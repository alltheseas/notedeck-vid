//! egui-vid: Cross-platform video playback for egui with hardware acceleration
//!
//! This crate provides hardware-accelerated video playback for egui applications
//! using **native platform media frameworks** - no FFmpeg required by default.
//!
//! # Native Platform Support
//!
//! Each platform uses its native media stack for optimal performance:
//!
//! | Platform | Native Framework | Hardware Acceleration |
//! |----------|------------------|----------------------|
//! | macOS | AVFoundation + VideoToolbox | Apple Silicon / Intel QuickSync |
//! | Linux | GStreamer | VA-API, NVDEC |
//! | Windows | Media Foundation | DXVA2, D3D11VA |
//! | Android | MediaCodec | Device hardware codecs |
//!
//! # Example
//!
//! ```ignore
//! use egui_vid::{VideoPlayer, VideoPlayerExt};
//!
//! // In your egui update() function:
//! let player = VideoPlayer::new("https://example.com/video.mp4");
//! player.show(ui, available_size);
//! ```
//!
//! # Feature Flags
//!
//! Native decoders (recommended - one per platform):
//! - `macos-native-video`: AVFoundation + VideoToolbox on macOS
//! - `linux-gstreamer-video`: GStreamer + VA-API on Linux
//! - `windows-native-video`: Media Foundation + DXVA2 on Windows
//!
//! Optional fallback:
//! - `ffmpeg`: FFmpeg decoder (cross-platform fallback, requires FFmpeg installation)

#![deny(clippy::disallowed_methods)]

pub mod media;

// Re-export main video types for convenience
pub use media::{
    AudioConfig, AudioHandle, AudioPlayer, AudioSamples, AudioState, AudioSync, CpuFrame,
    DecodedFrame, HwAccelType, PixelFormat, Plane, VideoControls, VideoControlsConfig,
    VideoControlsResponse, VideoDecoderBackend, VideoError, VideoFrame, VideoMetadata, VideoPlayer,
    VideoPlayerExt, VideoPlayerHandle, VideoPlayerResponse, VideoState,
};

#[cfg(feature = "ffmpeg")]
pub use media::{FfmpegDecoder, FfmpegDecoderBuilder, HwAccelConfig};

#[cfg(target_os = "android")]
pub use media::AndroidVideoDecoder;
