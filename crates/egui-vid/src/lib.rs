//! egui-vid: Cross-platform video playback for egui with hardware acceleration
//!
//! This crate provides hardware-accelerated video playback for egui applications,
//! with native decoder support for each platform:
//!
//! - **macOS**: Native AVFoundation + VideoToolbox (recommended)
//! - **Linux**: GStreamer with VA-API/NVDEC acceleration
//! - **Windows**: FFmpeg with DXVA2/D3D11VA acceleration
//! - **Android**: MediaCodec via JNI
//!
//! # Example
//!
//! ```no_run
//! use egui_vid::{VideoPlayer, VideoPlayerExt};
//!
//! // In your egui app:
//! let player = VideoPlayer::new("https://example.com/video.mp4");
//! player.show(ui, available_size);
//! ```
//!
//! # Features
//!
//! - `macos-native-video`: Native AVFoundation + VideoToolbox on macOS (recommended)
//! - `linux-gstreamer-video`: GStreamer on Linux
//! - `ffmpeg`: FFmpeg fallback (works on all platforms, requires FFmpeg installation)

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
