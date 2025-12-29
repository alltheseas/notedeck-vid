//! FFmpeg-based video decoder implementation.
//!
//! This module provides video decoding using FFmpeg (rust-ffmpeg) with support
//! for hardware acceleration on multiple platforms:
//!
//! - **macOS**: VideoToolbox (H.264, HEVC, VP9, AV1 on Apple Silicon)
//! - **Linux**: VAAPI (Intel/AMD GPUs)
//! - **Windows**: D3D11VA
//!
//! The decoder automatically attempts hardware acceleration and falls back
//! to software decoding if hardware acceleration is unavailable.

use std::time::Duration;

use super::video::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata,
};

/// Configuration for hardware acceleration.
#[derive(Debug, Clone)]
pub struct HwAccelConfig {
    /// The type of hardware acceleration to use
    pub hw_type: HwAccelType,
    /// Whether to fall back to software if hardware fails
    pub fallback_to_software: bool,
    /// Preferred output pixel format (None = let FFmpeg decide)
    pub preferred_output_format: Option<PixelFormat>,
}

impl Default for HwAccelConfig {
    fn default() -> Self {
        Self {
            hw_type: HwAccelType::platform_default(),
            fallback_to_software: true,
            preferred_output_format: None,
        }
    }
}

impl HwAccelConfig {
    /// Creates a config for software-only decoding.
    pub fn software_only() -> Self {
        Self {
            hw_type: HwAccelType::None,
            fallback_to_software: false,
            preferred_output_format: None,
        }
    }

    /// Creates a config for the specified hardware acceleration type.
    pub fn with_hw_type(hw_type: HwAccelType) -> Self {
        Self {
            hw_type,
            fallback_to_software: true,
            preferred_output_format: None,
        }
    }
}

/// FFmpeg-based video decoder with hardware acceleration support.
///
/// This decoder uses FFmpeg for video decoding with automatic hardware
/// acceleration on supported platforms:
///
/// - **macOS**: VideoToolbox
/// - **Linux**: VAAPI (with fallback to software)
/// - **Windows**: D3D11VA
///
/// # Example
///
/// ```ignore
/// // Create decoder with automatic hardware acceleration
/// let decoder = FfmpegDecoder::new("video.mp4")?;
///
/// // Or explicitly configure hardware acceleration
/// let decoder = FfmpegDecoderBuilder::new("video.mp4")
///     .with_hw_accel(HwAccelType::VideoToolbox)
///     .build()?;
/// ```
pub struct FfmpegDecoder {
    /// Video metadata (dimensions, duration, codec, etc.)
    metadata: VideoMetadata,
    /// The URL or path being decoded
    url: String,
    /// Hardware acceleration configuration
    hw_config: HwAccelConfig,
    /// The actual hardware acceleration type in use (may differ from config if fallback occurred)
    active_hw_type: HwAccelType,
    /// Current presentation timestamp
    current_pts: Duration,
    /// Whether end-of-file has been reached
    eof_reached: bool,

    // FFmpeg state (will be populated when FFmpeg integration is complete)
    // format_ctx: ffmpeg::format::context::Input,
    // video_stream_index: usize,
    // decoder: ffmpeg::codec::decoder::Video,
    // hw_device_ctx: Option<ffmpeg::device::Context>,
    // scaler: Option<ffmpeg::software::scaling::Context>,
}

impl FfmpegDecoder {
    /// Creates a new FFmpeg decoder for the given URL or file path.
    ///
    /// This will:
    /// 1. Open the media file/stream
    /// 2. Find the video stream
    /// 3. Attempt hardware acceleration (platform-specific)
    /// 4. Initialize the decoder (with software fallback if HW fails)
    /// 5. Extract metadata (duration, dimensions, framerate, etc.)
    pub fn new(url: &str) -> Result<Self, VideoError> {
        Self::new_with_config(url, HwAccelConfig::default())
    }

    /// Creates a new FFmpeg decoder with explicit hardware acceleration configuration.
    pub fn new_with_config(url: &str, hw_config: HwAccelConfig) -> Result<Self, VideoError> {
        // Try to initialize hardware acceleration (may fail if required but unavailable)
        let active_hw_type = Self::try_init_hw_accel(&hw_config)?;

        tracing::info!(
            "FfmpegDecoder: Opening {} with HW accel: {:?} (requested: {:?})",
            url,
            active_hw_type,
            hw_config.hw_type
        );

        // TODO: Integrate with actual FFmpeg once dependencies are configured
        //
        // ffmpeg::init().map_err(|e| VideoError::DecoderInit(e.to_string()))?;
        //
        // let format_ctx = ffmpeg::format::input(&url)
        //     .map_err(|e| VideoError::OpenFailed(e.to_string()))?;
        //
        // let video_stream = format_ctx
        //     .streams()
        //     .best(ffmpeg::media::Type::Video)
        //     .ok_or_else(|| VideoError::OpenFailed("No video stream found".to_string()))?;
        //
        // let video_stream_index = video_stream.index();
        //
        // // Set up hardware acceleration if available
        // let hw_device_ctx = Self::create_hw_device_context(active_hw_type)?;
        //
        // let codec_ctx = ffmpeg::codec::context::Context::from_parameters(video_stream.parameters())
        //     .map_err(|e| VideoError::DecoderInit(e.to_string()))?;
        //
        // // Configure decoder with hardware context
        // if let Some(ref hw_ctx) = hw_device_ctx {
        //     codec_ctx.set_hw_device_ctx(hw_ctx);
        // }
        //
        // let decoder = codec_ctx.decoder().video()
        //     .map_err(|e| VideoError::DecoderInit(e.to_string()))?;

        // For now, create a placeholder implementation
        let metadata = VideoMetadata {
            width: 1920,
            height: 1080,
            duration: Some(Duration::from_secs(60)),
            frame_rate: 30.0,
            codec: "h264".to_string(),
            pixel_aspect_ratio: 1.0,
        };

        Ok(Self {
            metadata,
            url: url.to_string(),
            hw_config,
            active_hw_type,
            current_pts: Duration::ZERO,
            eof_reached: false,
        })
    }

    /// Attempts to initialize hardware acceleration for the given configuration.
    ///
    /// Returns Ok(HwAccelType) with the active acceleration type, or Err if
    /// hardware acceleration was required (fallback_to_software = false) but failed.
    fn try_init_hw_accel(config: &HwAccelConfig) -> Result<HwAccelType, VideoError> {
        if config.hw_type == HwAccelType::None {
            return Ok(HwAccelType::None);
        }

        // Platform-specific hardware acceleration initialization
        let result = Self::init_hw_accel_platform(config.hw_type);

        match result {
            Ok(hw_type) => {
                tracing::info!("Hardware acceleration initialized: {:?}", hw_type);
                Ok(hw_type)
            }
            Err(e) => {
                if config.fallback_to_software {
                    tracing::warn!(
                        "Hardware acceleration {:?} failed: {}. Falling back to software decoding.",
                        config.hw_type,
                        e
                    );
                    Ok(HwAccelType::None)
                } else {
                    tracing::error!(
                        "Hardware acceleration {:?} failed: {}. No fallback configured.",
                        config.hw_type,
                        e
                    );
                    Err(VideoError::DecoderInit(format!(
                        "Hardware acceleration {:?} unavailable and fallback disabled: {}",
                        config.hw_type, e
                    )))
                }
            }
        }
    }

    /// Platform-specific hardware acceleration initialization.
    #[cfg(target_os = "macos")]
    fn init_hw_accel_platform(hw_type: HwAccelType) -> Result<HwAccelType, String> {
        match hw_type {
            HwAccelType::VideoToolbox => {
                // TODO: Initialize VideoToolbox
                //
                // VideoToolbox initialization on macOS:
                // 1. Check if VideoToolbox framework is available
                // 2. Create hardware device context:
                //    let hw_device_ctx = ffmpeg::device::context::Context::new(
                //        ffmpeg::device::Type::VideoToolbox
                //    )?;
                // 3. Verify codec support (H.264, HEVC, VP9 on Apple Silicon, AV1 on M3+)
                //
                // FFmpeg command equivalent:
                // ffmpeg -hwaccel videotoolbox -i input.mp4 output.mp4
                //
                // For rust-ffmpeg:
                // - Use av_hwdevice_ctx_create with AV_HWDEVICE_TYPE_VIDEOTOOLBOX
                // - Set decoder's hw_device_ctx
                // - Handle CVPixelBuffer output for zero-copy rendering
                tracing::debug!("VideoToolbox: Checking availability...");

                // Placeholder: VideoToolbox is generally always available on macOS 10.8+
                Ok(HwAccelType::VideoToolbox)
            }
            _ => Err(format!("{:?} not supported on macOS", hw_type)),
        }
    }

    #[cfg(target_os = "linux")]
    fn init_hw_accel_platform(hw_type: HwAccelType) -> Result<HwAccelType, String> {
        match hw_type {
            HwAccelType::Vaapi => {
                // TODO: Initialize VAAPI
                //
                // VAAPI initialization on Linux:
                // 1. Open DRM render node (typically /dev/dri/renderD128)
                // 2. Create VA display from DRM fd
                // 3. Initialize VA-API
                // 4. Create FFmpeg hardware device context:
                //    let hw_device_ctx = ffmpeg::device::context::Context::new_with_opts(
                //        ffmpeg::device::Type::Vaapi,
                //        "/dev/dri/renderD128"
                //    )?;
                //
                // FFmpeg command equivalent:
                // ffmpeg -hwaccel vaapi -hwaccel_device /dev/dri/renderD128 -i input.mp4 output.mp4
                //
                // Common issues:
                // - Need appropriate permissions on /dev/dri/renderD128
                // - Intel: requires intel-media-driver or i965-va-driver
                // - AMD: requires libva-mesa-driver
                tracing::debug!("VAAPI: Checking render node availability...");

                // Check if render node exists
                if std::path::Path::new("/dev/dri/renderD128").exists() {
                    Ok(HwAccelType::Vaapi)
                } else {
                    Err("VAAPI: No render node found at /dev/dri/renderD128".to_string())
                }
            }
            HwAccelType::Vdpau => {
                // TODO: Initialize VDPAU (NVIDIA legacy)
                //
                // VDPAU is primarily for older NVIDIA GPUs
                // Modern NVIDIA GPUs should use NVDEC via CUDA
                tracing::debug!("VDPAU: Checking availability...");
                Err("VDPAU support not yet implemented".to_string())
            }
            _ => Err(format!("{:?} not supported on Linux", hw_type)),
        }
    }

    #[cfg(target_os = "windows")]
    fn init_hw_accel_platform(hw_type: HwAccelType) -> Result<HwAccelType, String> {
        match hw_type {
            HwAccelType::D3d11va => {
                // TODO: Initialize D3D11VA
                //
                // D3D11VA initialization on Windows:
                // 1. Create D3D11 device
                // 2. Create FFmpeg hardware device context:
                //    let hw_device_ctx = ffmpeg::device::context::Context::new(
                //        ffmpeg::device::Type::D3d11va
                //    )?;
                //
                // FFmpeg command equivalent:
                // ffmpeg -hwaccel d3d11va -i input.mp4 output.mp4
                //
                // For wgpu integration:
                // - Can potentially share ID3D11Texture2D directly with wgpu
                // - Requires careful synchronization
                tracing::debug!("D3D11VA: Checking availability...");

                // D3D11VA is available on Windows 8+ with compatible GPU
                Ok(HwAccelType::D3d11va)
            }
            HwAccelType::Dxva2 => {
                // DXVA2 is legacy (Windows Vista/7), prefer D3D11VA
                tracing::debug!("DXVA2: Legacy API, prefer D3D11VA");
                Err("DXVA2 is deprecated, use D3D11VA instead".to_string())
            }
            _ => Err(format!("{:?} not supported on Windows", hw_type)),
        }
    }

    #[cfg(target_os = "android")]
    fn init_hw_accel_platform(hw_type: HwAccelType) -> Result<HwAccelType, String> {
        match hw_type {
            HwAccelType::MediaCodec => {
                // NOTE: On Android, we don't use FFmpeg for decoding
                // Instead, we use ExoPlayer via JNI (see android_video.rs)
                // This code path is here for completeness but won't be used
                tracing::warn!("MediaCodec: Use ExoPlayer JNI interface instead of FFmpeg");
                Err("Use ExoPlayer for Android video decoding".to_string())
            }
            _ => Err(format!("{:?} not supported on Android", hw_type)),
        }
    }

    #[cfg(not(any(
        target_os = "macos",
        target_os = "windows",
        target_os = "linux",
        target_os = "android"
    )))]
    fn init_hw_accel_platform(hw_type: HwAccelType) -> Result<HwAccelType, String> {
        Err(format!(
            "Hardware acceleration not supported on this platform: {:?}",
            hw_type
        ))
    }

    /// Returns the hardware acceleration configuration.
    pub fn hw_config(&self) -> &HwAccelConfig {
        &self.hw_config
    }

    /// Returns true if hardware acceleration is currently active.
    pub fn is_hw_accel_active(&self) -> bool {
        self.active_hw_type != HwAccelType::None
    }

    /// Converts an FFmpeg pixel format to our PixelFormat enum.
    fn convert_pixel_format(ffmpeg_format: i32) -> PixelFormat {
        // FFmpeg pixel format constants (from libavutil/pixfmt.h)
        // AV_PIX_FMT_YUV420P = 0
        // AV_PIX_FMT_NV12 = 23 (varies by version)
        // AV_PIX_FMT_RGB24 = 2
        // AV_PIX_FMT_RGBA = 26
        // AV_PIX_FMT_BGRA = 28
        match ffmpeg_format {
            0 => PixelFormat::Yuv420p,
            23 => PixelFormat::Nv12,
            2 => PixelFormat::Rgb24,
            26 => PixelFormat::Rgba,
            28 => PixelFormat::Bgra,
            _ => PixelFormat::Yuv420p, // Default fallback
        }
    }

    /// Creates a CpuFrame from FFmpeg frame data.
    ///
    /// This extracts the pixel data from all planes of the FFmpeg frame
    /// and copies it into our CpuFrame structure.
    fn frame_to_cpu_frame(
        format: PixelFormat,
        width: u32,
        height: u32,
        _frame_data: &[&[u8]],
        _strides: &[usize],
    ) -> CpuFrame {
        // TODO: Implement actual frame extraction from FFmpeg
        //
        // For YUV420P format:
        // - Plane 0 (Y): width * height bytes
        // - Plane 1 (U): (width/2) * (height/2) bytes
        // - Plane 2 (V): (width/2) * (height/2) bytes
        //
        // let planes = match format {
        //     PixelFormat::Yuv420p => {
        //         vec![
        //             Plane { data: frame.data(0).to_vec(), stride: frame.stride(0) },
        //             Plane { data: frame.data(1).to_vec(), stride: frame.stride(1) },
        //             Plane { data: frame.data(2).to_vec(), stride: frame.stride(2) },
        //         ]
        //     }
        //     PixelFormat::Nv12 => {
        //         vec![
        //             Plane { data: frame.data(0).to_vec(), stride: frame.stride(0) },
        //             Plane { data: frame.data(1).to_vec(), stride: frame.stride(1) },
        //         ]
        //     }
        //     PixelFormat::Rgba | PixelFormat::Bgra | PixelFormat::Rgb24 => {
        //         vec![
        //             Plane { data: frame.data(0).to_vec(), stride: frame.stride(0) },
        //         ]
        //     }
        // };

        // Placeholder: create empty planes
        let planes = match format {
            PixelFormat::Yuv420p => {
                let y_size = (width * height) as usize;
                let uv_size = ((width / 2) * (height / 2)) as usize;
                vec![
                    Plane {
                        data: vec![128; y_size], // Gray Y plane
                        stride: width as usize,
                    },
                    Plane {
                        data: vec![128; uv_size], // Neutral U plane
                        stride: (width / 2) as usize,
                    },
                    Plane {
                        data: vec![128; uv_size], // Neutral V plane
                        stride: (width / 2) as usize,
                    },
                ]
            }
            PixelFormat::Nv12 => {
                let y_size = (width * height) as usize;
                let uv_size = ((width / 2) * (height / 2) * 2) as usize;
                vec![
                    Plane {
                        data: vec![128; y_size],
                        stride: width as usize,
                    },
                    Plane {
                        data: vec![128; uv_size],
                        stride: width as usize,
                    },
                ]
            }
            PixelFormat::Rgba | PixelFormat::Bgra => {
                let size = (width * height * 4) as usize;
                vec![Plane {
                    data: vec![128; size],
                    stride: (width * 4) as usize,
                }]
            }
            PixelFormat::Rgb24 => {
                let size = (width * height * 3) as usize;
                vec![Plane {
                    data: vec![128; size],
                    stride: (width * 3) as usize,
                }]
            }
        };

        CpuFrame::new(format, width, height, planes)
    }

    /// Converts PTS (presentation timestamp) from stream time base to Duration.
    fn pts_to_duration(pts: i64, time_base_num: i32, time_base_den: i32) -> Duration {
        if pts < 0 || time_base_den == 0 {
            return Duration::ZERO;
        }

        let seconds = (pts as f64) * (time_base_num as f64) / (time_base_den as f64);
        Duration::from_secs_f64(seconds.max(0.0))
    }

    /// Converts Duration to PTS for seeking.
    fn duration_to_pts(duration: Duration, time_base_num: i32, time_base_den: i32) -> i64 {
        if time_base_num == 0 {
            return 0;
        }

        let seconds = duration.as_secs_f64();
        ((seconds * time_base_den as f64) / time_base_num as f64) as i64
    }
}

impl VideoDecoderBackend for FfmpegDecoder {
    fn open(url: &str) -> Result<Self, VideoError>
    where
        Self: Sized,
    {
        Self::new(url)
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        if self.eof_reached {
            return Ok(None);
        }

        // TODO: Implement actual frame decoding
        //
        // The implementation will look like:
        //
        // let mut frame = ffmpeg::util::frame::Video::empty();
        //
        // loop {
        //     match self.decoder.receive_frame(&mut frame) {
        //         Ok(()) => {
        //             // Got a frame
        //             let pts = Self::pts_to_duration(
        //                 frame.pts().unwrap_or(0),
        //                 self.time_base.0,
        //                 self.time_base.1
        //             );
        //
        //             let cpu_frame = Self::frame_to_cpu_frame(
        //                 Self::convert_pixel_format(frame.format() as i32),
        //                 frame.width(),
        //                 frame.height(),
        //                 &[frame.data(0), frame.data(1), frame.data(2)],
        //                 &[frame.stride(0), frame.stride(1), frame.stride(2)],
        //             );
        //
        //             return Ok(Some(VideoFrame::new(pts, DecodedFrame::Cpu(cpu_frame))));
        //         }
        //         Err(ffmpeg::Error::Eof) => {
        //             self.eof_reached = true;
        //             return Ok(None);
        //         }
        //         Err(ffmpeg::Error::Other { errno: EAGAIN }) => {
        //             // Need more packets, read next packet
        //             match self.format_ctx.packets().next() {
        //                 Some((stream, packet)) => {
        //                     if stream.index() == self.video_stream_index {
        //                         self.decoder.send_packet(&packet)?;
        //                     }
        //                 }
        //                 None => {
        //                     self.decoder.send_eof()?;
        //                 }
        //             }
        //         }
        //         Err(e) => return Err(VideoError::DecodeFailed(e.to_string())),
        //     }
        // }

        // Placeholder: simulate frame generation
        let frame_duration = self.metadata.frame_duration();

        if let Some(duration) = self.metadata.duration {
            if self.current_pts >= duration {
                self.eof_reached = true;
                return Ok(None);
            }
        }

        let pts = self.current_pts;
        self.current_pts += frame_duration;

        let cpu_frame = Self::frame_to_cpu_frame(
            PixelFormat::Yuv420p,
            self.metadata.width,
            self.metadata.height,
            &[],
            &[],
        );

        Ok(Some(VideoFrame::new(pts, DecodedFrame::Cpu(cpu_frame))))
    }

    fn seek(&mut self, position: Duration) -> Result<(), VideoError> {
        // TODO: Implement actual seeking
        //
        // let pts = Self::duration_to_pts(position, self.time_base.0, self.time_base.1);
        //
        // self.format_ctx.seek(pts, ..pts)?;
        // self.decoder.flush();

        // Placeholder: just update current position
        self.current_pts = position;
        self.eof_reached = false;

        Ok(())
    }

    fn metadata(&self) -> &VideoMetadata {
        &self.metadata
    }

    fn hw_accel_type(&self) -> HwAccelType {
        self.active_hw_type
    }
}

/// Builder for FfmpegDecoder with configuration options.
pub struct FfmpegDecoderBuilder {
    url: String,
    hw_config: HwAccelConfig,
}

impl FfmpegDecoderBuilder {
    /// Creates a new builder for the given URL.
    ///
    /// By default, uses the platform's preferred hardware acceleration
    /// with software fallback enabled.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            hw_config: HwAccelConfig::default(),
        }
    }

    /// Sets the hardware acceleration type to use.
    pub fn with_hw_accel(mut self, hw_type: HwAccelType) -> Self {
        self.hw_config.hw_type = hw_type;
        self
    }

    /// Disables hardware acceleration (software only).
    pub fn software_only(mut self) -> Self {
        self.hw_config = HwAccelConfig::software_only();
        self
    }

    /// Sets whether to fall back to software decoding if hardware fails.
    pub fn with_fallback(mut self, fallback: bool) -> Self {
        self.hw_config.fallback_to_software = fallback;
        self
    }

    /// Sets the preferred output pixel format.
    pub fn with_output_format(mut self, format: PixelFormat) -> Self {
        self.hw_config.preferred_output_format = Some(format);
        self
    }

    /// Sets the complete hardware acceleration configuration.
    pub fn with_hw_config(mut self, config: HwAccelConfig) -> Self {
        self.hw_config = config;
        self
    }

    /// Builds the decoder with the configured options.
    pub fn build(self) -> Result<FfmpegDecoder, VideoError> {
        FfmpegDecoder::new_with_config(&self.url, self.hw_config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let decoder = FfmpegDecoder::new("test.mp4");
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_decoder_with_software_only() {
        let decoder = FfmpegDecoder::new_with_config("test.mp4", HwAccelConfig::software_only());
        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.hw_accel_type(), HwAccelType::None);
        assert!(!decoder.is_hw_accel_active());
    }

    #[test]
    fn test_decoder_builder() {
        let decoder = FfmpegDecoderBuilder::new("test.mp4")
            .software_only()
            .build();
        assert!(decoder.is_ok());
        assert_eq!(decoder.unwrap().hw_accel_type(), HwAccelType::None);
    }

    #[test]
    fn test_decoder_builder_with_hw_accel() {
        // This will use platform default HW accel
        let decoder = FfmpegDecoderBuilder::new("test.mp4")
            .with_fallback(true)
            .build();
        assert!(decoder.is_ok());

        // The actual HW type depends on platform
        let decoder = decoder.unwrap();
        let hw_type = decoder.hw_accel_type();

        #[cfg(target_os = "macos")]
        assert_eq!(hw_type, HwAccelType::VideoToolbox);

        #[cfg(target_os = "linux")]
        {
            // May be VAAPI if render node exists, otherwise None
            assert!(hw_type == HwAccelType::Vaapi || hw_type == HwAccelType::None);
        }

        #[cfg(target_os = "windows")]
        assert_eq!(hw_type, HwAccelType::D3d11va);
    }

    #[test]
    fn test_hw_accel_config_default() {
        let config = HwAccelConfig::default();
        assert!(config.fallback_to_software);
        assert!(config.preferred_output_format.is_none());

        #[cfg(target_os = "macos")]
        assert_eq!(config.hw_type, HwAccelType::VideoToolbox);

        #[cfg(target_os = "linux")]
        assert_eq!(config.hw_type, HwAccelType::Vaapi);

        #[cfg(target_os = "windows")]
        assert_eq!(config.hw_type, HwAccelType::D3d11va);
    }

    #[test]
    fn test_metadata() {
        let decoder = FfmpegDecoder::new("test.mp4").unwrap();
        let metadata = decoder.metadata();
        assert_eq!(metadata.width, 1920);
        assert_eq!(metadata.height, 1080);
    }

    #[test]
    fn test_pts_conversion() {
        // 30 fps video, time_base = 1/30
        let pts = 90; // 3 seconds worth of frames
        let duration = FfmpegDecoder::pts_to_duration(pts, 1, 30);
        assert_eq!(duration, Duration::from_secs(3));
    }

    #[test]
    fn test_duration_to_pts() {
        let duration = Duration::from_secs(3);
        let pts = FfmpegDecoder::duration_to_pts(duration, 1, 30);
        assert_eq!(pts, 90);
    }
}
