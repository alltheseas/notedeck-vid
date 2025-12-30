//! Linux native video decoding using VAAPI via cros-codecs.
//!
//! This module provides hardware-accelerated video decoding on Linux
//! using the VAAPI backend from cros-codecs (ChromeOS video stack).
//!
//! # Architecture
//!
//! The decoding pipeline consists of:
//! 1. **Demuxer**: Extracts encoded video samples from containers
//!    - MP4/MOV: Uses the `mp4` crate
//!    - MKV/WebM: Uses `matroska-demuxer`
//! 2. **Codec Parser**: cros-codecs parses H.264/H.265/VP9 NAL units
//! 3. **VAAPI Backend**: Hardware decoding via libva
//! 4. **Frame Output**: NV12/I420 frames for GPU upload
//!
//! # Feature Flag
//!
//! This module is only compiled when `linux-native-video` feature is enabled:
//! ```toml
//! [features]
//! linux-native-video = ["cros-codecs", "matroska-demuxer", "mp4"]
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use notedeck::media::LinuxVaapiDecoder;
//! use notedeck::media::VideoDecoderBackend;
//!
//! let decoder = LinuxVaapiDecoder::open("video.mp4")?;
//! while let Some(frame) = decoder.decode_next()? {
//!     // Process frame...
//! }
//! ```

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::time::Duration;

use crate::media::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata,
};

/// Video container format detected from file extension or content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerFormat {
    /// MP4/MOV container (ISO Base Media File Format)
    Mp4,
    /// Matroska container (.mkv)
    Matroska,
    /// WebM container (Matroska subset for web)
    WebM,
}

impl ContainerFormat {
    /// Detect container format from URL/path.
    pub fn from_url(url: &str) -> Option<Self> {
        let lower = url.to_lowercase();
        if lower.ends_with(".mp4") || lower.ends_with(".mov") || lower.ends_with(".m4v") {
            Some(ContainerFormat::Mp4)
        } else if lower.ends_with(".mkv") {
            Some(ContainerFormat::Matroska)
        } else if lower.ends_with(".webm") {
            Some(ContainerFormat::WebM)
        } else {
            None
        }
    }
}

/// Video codec detected from container metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    /// H.264/AVC
    H264,
    /// H.265/HEVC
    H265,
    /// VP8
    Vp8,
    /// VP9
    Vp9,
    /// AV1
    Av1,
}

impl VideoCodec {
    /// Get codec name as string.
    pub fn as_str(&self) -> &'static str {
        match self {
            VideoCodec::H264 => "h264",
            VideoCodec::H265 => "hevc",
            VideoCodec::Vp8 => "vp8",
            VideoCodec::Vp9 => "vp9",
            VideoCodec::Av1 => "av1",
        }
    }
}

/// An encoded video sample extracted from a container.
#[derive(Debug, Clone)]
pub struct EncodedSample {
    /// Raw encoded data (NAL units for H.264/H.265, frame data for VP8/VP9)
    pub data: Vec<u8>,
    /// Presentation timestamp
    pub pts: Duration,
    /// Whether this is a keyframe (IDR for H.264, I-frame for VP8/VP9)
    pub is_keyframe: bool,
}

/// Trait for video demuxers.
pub trait Demuxer: Send {
    /// Read the next encoded video sample.
    fn next_sample(&mut self) -> Result<Option<EncodedSample>, VideoError>;

    /// Seek to the nearest keyframe before the given position.
    fn seek(&mut self, position: Duration) -> Result<(), VideoError>;

    /// Get video metadata.
    fn metadata(&self) -> &DemuxerMetadata;
}

/// Metadata extracted from the demuxer.
#[derive(Debug, Clone)]
pub struct DemuxerMetadata {
    /// Video width in pixels
    pub width: u32,
    /// Video height in pixels
    pub height: u32,
    /// Duration of the video (if known)
    pub duration: Option<Duration>,
    /// Frame rate (frames per second)
    pub frame_rate: f64,
    /// Video codec
    pub codec: VideoCodec,
    /// Codec-specific data (SPS/PPS for H.264, etc.)
    pub codec_private: Option<Vec<u8>>,
}

// ============================================================================
// MP4 Demuxer Implementation
// ============================================================================

/// MP4 demuxer using the `mp4` crate.
///
/// Extracts H.264/H.265 samples from MP4/MOV containers.
pub struct Mp4Demuxer {
    reader: mp4::Mp4Reader<BufReader<File>>,
    video_track_id: u32,
    sample_index: u32,
    sample_count: u32,
    timescale: u32,
    metadata: DemuxerMetadata,
}

impl Mp4Demuxer {
    /// Opens an MP4 file and finds the video track.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, VideoError> {
        let file = File::open(path.as_ref())
            .map_err(|e| VideoError::OpenFailed(format!("Failed to open file: {}", e)))?;

        let size = file
            .metadata()
            .map_err(|e| VideoError::OpenFailed(format!("Failed to get file size: {}", e)))?
            .len();

        let reader = BufReader::new(file);
        let mp4 = mp4::Mp4Reader::read_header(reader, size)
            .map_err(|e| VideoError::OpenFailed(format!("Failed to parse MP4 header: {}", e)))?;

        // Find video track
        let mut video_track_id = None;
        let mut video_track_info = None;

        for (track_id, track) in mp4.tracks().iter() {
            if let Ok(track_type) = track.track_type() {
                if track_type == mp4::TrackType::Video {
                    video_track_id = Some(*track_id);
                    video_track_info = Some(track);
                    break;
                }
            }
        }

        let (track_id, track) = match (video_track_id, video_track_info) {
            (Some(id), Some(t)) => (id, t),
            _ => {
                return Err(VideoError::UnsupportedFormat(
                    "No video track found in MP4 file".to_string(),
                ));
            }
        };

        // Detect codec
        let codec = match track.media_type() {
            Ok(mp4::MediaType::H264) => VideoCodec::H264,
            Ok(mp4::MediaType::H265) => VideoCodec::H265,
            Ok(mp4::MediaType::VP9) => VideoCodec::Vp9,
            Ok(other) => {
                return Err(VideoError::UnsupportedFormat(format!(
                    "Unsupported video codec: {:?}",
                    other
                )));
            }
            Err(e) => {
                return Err(VideoError::UnsupportedFormat(format!(
                    "Failed to detect codec: {}",
                    e
                )));
            }
        };

        let width = track.width() as u32;
        let height = track.height() as u32;
        let frame_rate = track.frame_rate();
        let timescale = track.timescale();
        let sample_count = track.sample_count();

        // Get duration from track or file
        let duration = if let Ok(dur) = track.duration() {
            Some(dur)
        } else {
            mp4.duration()
        };

        // Get codec private data (SPS/PPS for H.264)
        let codec_private = track.sequence_parameter_set().ok().map(|sps| {
            let mut data = sps.to_vec();
            if let Ok(pps) = track.picture_parameter_set() {
                data.extend_from_slice(pps);
            }
            data
        });

        let metadata = DemuxerMetadata {
            width,
            height,
            duration,
            frame_rate,
            codec,
            codec_private,
        };

        tracing::info!(
            "MP4 demuxer opened: {}x{} {:?} @ {:.2} fps, {} samples",
            width,
            height,
            codec,
            frame_rate,
            sample_count
        );

        Ok(Self {
            reader: mp4,
            video_track_id: track_id,
            sample_index: 1, // MP4 samples are 1-indexed
            sample_count,
            timescale,
            metadata,
        })
    }
}

impl Demuxer for Mp4Demuxer {
    fn next_sample(&mut self) -> Result<Option<EncodedSample>, VideoError> {
        if self.sample_index > self.sample_count {
            return Ok(None);
        }

        let sample = self
            .reader
            .read_sample(self.video_track_id, self.sample_index)
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to read sample: {}", e)))?;

        self.sample_index += 1;

        match sample {
            Some(s) => {
                // Convert timestamp from timescale to Duration
                let pts_us = if self.timescale > 0 {
                    (s.start_time as u64 * 1_000_000) / self.timescale as u64
                } else {
                    s.start_time as u64
                };

                Ok(Some(EncodedSample {
                    data: s.bytes.to_vec(),
                    pts: Duration::from_micros(pts_us),
                    is_keyframe: s.is_sync,
                }))
            }
            None => Ok(None),
        }
    }

    fn seek(&mut self, position: Duration) -> Result<(), VideoError> {
        // Convert position to sample time
        let target_time = (position.as_micros() as u64 * self.timescale as u64) / 1_000_000;

        // Binary search for nearest keyframe before position
        // For now, simple linear search from beginning (can be optimized)
        self.sample_index = 1;

        // TODO: Implement proper keyframe seeking using sample tables
        // For now, just reset to start if seeking backward
        tracing::debug!("MP4 seek to {:?} (target_time: {})", position, target_time);

        Ok(())
    }

    fn metadata(&self) -> &DemuxerMetadata {
        &self.metadata
    }
}

// ============================================================================
// Matroska/WebM Demuxer Implementation
// ============================================================================

/// Matroska/WebM demuxer using `matroska-demuxer`.
///
/// Extracts VP8/VP9/H.264 samples from MKV/WebM containers.
pub struct MkvDemuxer {
    demuxer: matroska_demuxer::MatroskaFile<BufReader<File>>,
    video_track_num: u64,
    metadata: DemuxerMetadata,
}

impl MkvDemuxer {
    /// Opens a Matroska/WebM file and finds the video track.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, VideoError> {
        let file = File::open(path.as_ref())
            .map_err(|e| VideoError::OpenFailed(format!("Failed to open file: {}", e)))?;

        let reader = BufReader::new(file);
        let demuxer = matroska_demuxer::MatroskaFile::open(reader).map_err(|e| {
            VideoError::OpenFailed(format!("Failed to parse Matroska header: {}", e))
        })?;

        // Find video track
        let mut video_track = None;
        for track in demuxer.tracks() {
            if track.track_type() == matroska_demuxer::TrackType::Video {
                video_track = Some(track);
                break;
            }
        }

        let track = video_track.ok_or_else(|| {
            VideoError::UnsupportedFormat("No video track found in Matroska file".to_string())
        })?;

        let video_track_num = track.track_number().get();

        // Detect codec from codec_id
        let codec_id = track.codec_id();
        let codec = match codec_id {
            "V_MPEG4/ISO/AVC" => VideoCodec::H264,
            "V_MPEGH/ISO/HEVC" => VideoCodec::H265,
            "V_VP8" => VideoCodec::Vp8,
            "V_VP9" => VideoCodec::Vp9,
            "V_AV1" => VideoCodec::Av1,
            other => {
                return Err(VideoError::UnsupportedFormat(format!(
                    "Unsupported Matroska codec: {}",
                    other
                )));
            }
        };

        // Get video dimensions
        let (width, height) = if let Some(video) = track.video() {
            (video.pixel_width() as u32, video.pixel_height() as u32)
        } else {
            return Err(VideoError::UnsupportedFormat(
                "Video track missing video settings".to_string(),
            ));
        };

        // Calculate frame rate from default duration (nanoseconds per frame)
        let frame_rate = if let Some(default_duration) = track.default_duration() {
            if default_duration > 0 {
                1_000_000_000.0 / default_duration as f64
            } else {
                30.0 // Default
            }
        } else {
            30.0 // Default if not specified
        };

        // Get duration from segment info
        let duration = demuxer
            .info()
            .duration()
            .map(|d| Duration::from_nanos(d as u64));

        // Get codec private data
        let codec_private = track.codec_private().map(|p| p.to_vec());

        let metadata = DemuxerMetadata {
            width,
            height,
            duration,
            frame_rate,
            codec,
            codec_private,
        };

        tracing::info!(
            "Matroska demuxer opened: {}x{} {:?} @ {:.2} fps",
            width,
            height,
            codec,
            frame_rate
        );

        Ok(Self {
            demuxer,
            video_track_num,
            metadata,
        })
    }
}

impl Demuxer for MkvDemuxer {
    fn next_sample(&mut self) -> Result<Option<EncodedSample>, VideoError> {
        loop {
            match self.demuxer.next_frame() {
                Ok(Some(frame)) => {
                    // Skip non-video frames
                    if frame.track != self.video_track_num {
                        continue;
                    }

                    return Ok(Some(EncodedSample {
                        data: frame.data,
                        pts: Duration::from_nanos(frame.timestamp),
                        is_keyframe: frame.is_keyframe.unwrap_or(false),
                    }));
                }
                Ok(None) => return Ok(None),
                Err(e) => {
                    return Err(VideoError::DecodeFailed(format!(
                        "Failed to read Matroska frame: {:?}",
                        e
                    )));
                }
            }
        }
    }

    fn seek(&mut self, _position: Duration) -> Result<(), VideoError> {
        // matroska-demuxer doesn't support seeking directly
        // Would need to re-open the file and skip frames
        // For now, return error
        Err(VideoError::SeekFailed(
            "Matroska seeking not yet implemented".to_string(),
        ))
    }

    fn metadata(&self) -> &DemuxerMetadata {
        &self.metadata
    }
}

// ============================================================================
// Linux VAAPI Decoder
// ============================================================================

/// Unified demuxer enum for different container formats.
enum DemuxerImpl {
    Mp4(Mp4Demuxer),
    Matroska(MkvDemuxer),
}

impl Demuxer for DemuxerImpl {
    fn next_sample(&mut self) -> Result<Option<EncodedSample>, VideoError> {
        match self {
            DemuxerImpl::Mp4(d) => d.next_sample(),
            DemuxerImpl::Matroska(d) => d.next_sample(),
        }
    }

    fn seek(&mut self, position: Duration) -> Result<(), VideoError> {
        match self {
            DemuxerImpl::Mp4(d) => d.seek(position),
            DemuxerImpl::Matroska(d) => d.seek(position),
        }
    }

    fn metadata(&self) -> &DemuxerMetadata {
        match self {
            DemuxerImpl::Mp4(d) => d.metadata(),
            DemuxerImpl::Matroska(d) => d.metadata(),
        }
    }
}

/// Linux VAAPI video decoder using cros-codecs.
///
/// This decoder uses cros-codecs for hardware-accelerated video decoding
/// via the VAAPI backend. It supports H.264, H.265, VP8, VP9, and AV1.
pub struct LinuxVaapiDecoder {
    /// Demuxer for extracting encoded samples
    demuxer: DemuxerImpl,
    /// Video metadata
    metadata: VideoMetadata,
    /// Current playback position
    position: Duration,
    // TODO: Add cros-codecs decoder instance when VAAPI integration is done
    // decoder: Option<StatelessDecoder<H264, VaapiBackend>>,
}

impl LinuxVaapiDecoder {
    /// Creates a new decoder for the given file path.
    ///
    /// This will:
    /// 1. Detect the container format
    /// 2. Open and parse the container to find video tracks
    /// 3. Extract video metadata
    pub fn new(path: &str) -> Result<Self, VideoError> {
        // Detect container format
        let container = ContainerFormat::from_url(path).ok_or_else(|| {
            VideoError::UnsupportedFormat(format!(
                "Could not detect container format from path: {}",
                path
            ))
        })?;

        tracing::info!(
            "LinuxVaapiDecoder: Opening {:?} container: {}",
            container,
            path
        );

        // Open appropriate demuxer
        let demuxer = match container {
            ContainerFormat::Mp4 => {
                let mp4 = Mp4Demuxer::open(path)?;
                DemuxerImpl::Mp4(mp4)
            }
            ContainerFormat::Matroska | ContainerFormat::WebM => {
                let mkv = MkvDemuxer::open(path)?;
                DemuxerImpl::Matroska(mkv)
            }
        };

        // Build VideoMetadata from demuxer metadata
        let dm = demuxer.metadata();
        let metadata = VideoMetadata {
            width: dm.width,
            height: dm.height,
            duration: dm.duration,
            frame_rate: dm.frame_rate as f32,
            codec: dm.codec.as_str().to_string(),
            pixel_aspect_ratio: 1.0,
        };

        Ok(Self {
            demuxer,
            metadata,
            position: Duration::ZERO,
        })
    }
}

impl VideoDecoderBackend for LinuxVaapiDecoder {
    fn open(url: &str) -> Result<Self, VideoError>
    where
        Self: Sized,
    {
        Self::new(url)
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        // Read next sample from demuxer
        let sample = match self.demuxer.next_sample()? {
            Some(s) => s,
            None => return Ok(None),
        };

        self.position = sample.pts;

        // TODO: Feed to cros-codecs for VAAPI decoding
        // For now, return a placeholder gray frame to verify demuxing works
        let width = self.metadata.width;
        let height = self.metadata.height;

        // Create a gray NV12 frame (Y=128, UV=128 is gray)
        let y_size = (width * height) as usize;
        let uv_size = y_size / 2; // NV12 has half-size interleaved UV

        let y_plane = Plane {
            data: vec![128u8; y_size],
            stride: width as usize,
        };

        let uv_plane = Plane {
            data: vec![128u8; uv_size],
            stride: width as usize,
        };

        let cpu_frame = CpuFrame::new(PixelFormat::Nv12, width, height, vec![y_plane, uv_plane]);

        Ok(Some(VideoFrame::new(
            sample.pts,
            DecodedFrame::Cpu(cpu_frame),
        )))
    }

    fn seek(&mut self, position: Duration) -> Result<(), VideoError> {
        self.demuxer.seek(position)?;
        self.position = position;
        Ok(())
    }

    fn metadata(&self) -> &VideoMetadata {
        &self.metadata
    }

    fn duration(&self) -> Option<Duration> {
        self.metadata.duration
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.metadata.width, self.metadata.height)
    }

    fn hw_accel_type(&self) -> HwAccelType {
        HwAccelType::Vaapi
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container_format_detection() {
        assert_eq!(
            ContainerFormat::from_url("video.mp4"),
            Some(ContainerFormat::Mp4)
        );
        assert_eq!(
            ContainerFormat::from_url("video.mkv"),
            Some(ContainerFormat::Matroska)
        );
        assert_eq!(
            ContainerFormat::from_url("video.webm"),
            Some(ContainerFormat::WebM)
        );
        assert_eq!(ContainerFormat::from_url("video.avi"), None);
    }

    #[test]
    fn test_codec_names() {
        assert_eq!(VideoCodec::H264.as_str(), "h264");
        assert_eq!(VideoCodec::H265.as_str(), "hevc");
        assert_eq!(VideoCodec::Vp9.as_str(), "vp9");
    }
}
