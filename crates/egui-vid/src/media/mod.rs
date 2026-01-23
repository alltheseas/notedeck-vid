//! Video and audio playback modules for egui-vid.
//!
//! This module provides cross-platform hardware-accelerated video playback:
//!
//! - [`VideoPlayer`] - Main video player widget for egui
//! - [`VideoControls`] - Play/pause, seek, volume controls
//! - [`video`] - Core video types and decoder traits
//! - [`audio`] - Audio playback and synchronization
//!
//! # Platform Support
//!
//! | Platform | Decoder | Hardware Acceleration |
//! |----------|---------|----------------------|
//! | macOS | AVFoundation | VideoToolbox |
//! | Linux | GStreamer | VA-API, NVDEC |
//! | Windows | Media Foundation | DXVA2, D3D11VA |
//! | Android | MediaCodec | Hardware codecs |
//!
//! # Known Issues
//!
//! ## macOS objc2 Version Coexistence
//!
//! The native macOS decoder uses `objc2 0.6.x` for AVFoundation bindings,
//! while `winit` (used by egui) uses `objc2 0.5.x`. These versions coexist
//! safely because they bind to different Objective-C classes:
//!
//! - `objc2 0.5.x`: winit's window management classes
//! - `objc2 0.6.x`: AVFoundation media classes
//!
//! This is a known working configuration. If you encounter issues, ensure
//! you're using the damus egui/eframe fork which has been tested with this
//! setup.

#[cfg(target_os = "android")]
pub mod android_video;
pub mod audio;
#[cfg(feature = "ffmpeg")]
pub mod audio_decoder;
pub mod frame_queue;
#[cfg(all(target_os = "linux", feature = "linux-gstreamer-video"))]
pub mod linux_video_gst;
#[cfg(all(target_os = "macos", feature = "macos-native-video"))]
pub mod macos_video;
pub mod network;
pub mod triple_buffer;
pub mod video;
pub mod video_controls;
#[cfg(feature = "ffmpeg")]
pub mod video_decoder;
pub mod video_player;
pub mod video_texture;
#[cfg(all(target_os = "windows", feature = "windows-native-video"))]
pub mod windows_audio;
#[cfg(all(target_os = "windows", feature = "windows-native-video"))]
pub mod windows_video;

// Re-export main types
pub use audio::{AudioConfig, AudioHandle, AudioPlayer, AudioSamples, AudioState, AudioSync};
pub use video::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata, VideoPlayerHandle, VideoState,
};
pub use video_controls::{VideoControls, VideoControlsConfig, VideoControlsResponse};
#[cfg(feature = "ffmpeg")]
pub use video_decoder::{FfmpegDecoder, FfmpegDecoderBuilder, HwAccelConfig};
pub use video_player::{VideoPlayer, VideoPlayerExt, VideoPlayerResponse};

#[cfg(target_os = "android")]
pub use android_video::AndroidVideoDecoder;

#[cfg(all(target_os = "macos", feature = "macos-native-video"))]
pub use macos_video::MacOSVideoDecoder;

#[cfg(all(target_os = "linux", feature = "linux-gstreamer-video"))]
pub use linux_video_gst::GStreamerDecoder;

#[cfg(all(target_os = "windows", feature = "windows-native-video"))]
pub use windows_video::WindowsVideoDecoder;

/// Maximum texture size wgpu can handle without panicking.
pub const MAX_SIZE_WGPU: usize = 8192;
