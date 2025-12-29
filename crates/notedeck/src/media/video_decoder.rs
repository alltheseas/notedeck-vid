//! FFmpeg-based video decoder implementation.
//!
//! This module provides the software video decoder using FFmpeg (rust-ffmpeg).
//! Hardware acceleration support (VideoToolbox, VAAPI, D3D11VA) will be added
//! in a future milestone.

use std::time::Duration;

use super::video::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata,
};

/// FFmpeg-based video decoder.
///
/// This decoder uses the rust-ffmpeg crate to decode video files and streams.
/// Currently implements software decoding; hardware acceleration will be added
/// in Milestone 2.
pub struct FfmpegDecoder {
    // These will be populated once we integrate with the actual FFmpeg crate
    metadata: VideoMetadata,
    url: String,
    // Placeholder fields - will be replaced with actual FFmpeg types
    // format_ctx: ffmpeg::format::context::Input,
    // video_stream_index: usize,
    // decoder: ffmpeg::codec::decoder::Video,
    // scaler: Option<ffmpeg::software::scaling::Context>,
    current_pts: Duration,
    eof_reached: bool,
}

impl FfmpegDecoder {
    /// Creates a new FFmpeg decoder for the given URL or file path.
    ///
    /// This will:
    /// 1. Open the media file/stream
    /// 2. Find the video stream
    /// 3. Initialize the decoder
    /// 4. Extract metadata (duration, dimensions, framerate, etc.)
    pub fn new(url: &str) -> Result<Self, VideoError> {
        // TODO: Integrate with actual FFmpeg once dependencies are configured
        //
        // The implementation will look like:
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
        // let codec_ctx = ffmpeg::codec::context::Context::from_parameters(video_stream.parameters())
        //     .map_err(|e| VideoError::DecoderInit(e.to_string()))?;
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
            current_pts: Duration::ZERO,
            eof_reached: false,
        })
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
        // Software decoding for now
        HwAccelType::None
    }
}

/// Builder for FfmpegDecoder with configuration options.
pub struct FfmpegDecoderBuilder {
    url: String,
    hw_accel: Option<HwAccelType>,
    // Additional options can be added here
}

impl FfmpegDecoderBuilder {
    /// Creates a new builder for the given URL.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            hw_accel: None,
        }
    }

    /// Enables hardware acceleration (if available).
    pub fn with_hw_accel(mut self, hw_accel: HwAccelType) -> Self {
        self.hw_accel = Some(hw_accel);
        self
    }

    /// Builds the decoder.
    pub fn build(self) -> Result<FfmpegDecoder, VideoError> {
        // TODO: Apply hardware acceleration settings during decoder creation
        FfmpegDecoder::new(&self.url)
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
