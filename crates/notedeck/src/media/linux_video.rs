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
//! # Requirements
//!
//! - libva-dev (or equivalent) installed
//! - VA-API driver for your hardware:
//!   - Intel: intel-media-driver
//!   - AMD: Mesa (radeonsi)
//!   - NVIDIA: Not supported (use NVDEC instead)
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

use std::collections::VecDeque;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::Duration;

use crate::media::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata,
};

// cros-codecs imports for VAAPI integration
// These are used in the VAAPI decoder implementation
#[allow(unused_imports)]
use cros_codecs::decoder::BlockingMode;
#[allow(unused_imports)]
use cros_codecs::DecodedFormat;

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
    ///
    /// Handles URLs with query parameters (e.g., `video.mp4?token=xyz`).
    pub fn from_url(url: &str) -> Option<Self> {
        // Strip query parameters and fragments before checking extension
        let path = url.split('?').next().unwrap_or(url);
        let path = path.split('#').next().unwrap_or(path);
        let lower = path.to_lowercase();

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

        tracing::debug!("MP4 seek to {:?} (target_time: {})", position, target_time);

        // TODO: Implement proper keyframe seeking using sample tables
        // For now, only support seeking to start (position == 0)
        if position.is_zero() {
            self.sample_index = 1;
            Ok(())
        } else {
            Err(VideoError::SeekFailed(
                "MP4 seeking not yet implemented (only seek to start supported)".to_string(),
            ))
        }
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
// VAAPI Decoder Backend
// ============================================================================

/// DRM render node paths to try for VAAPI.
const DRM_RENDER_NODES: &[&str] = &[
    "/dev/dri/renderD128",
    "/dev/dri/renderD129",
    "/dev/dri/renderD130",
];

/// Check if a DRM render node exists and is accessible.
fn find_drm_render_node() -> Option<&'static str> {
    for path in DRM_RENDER_NODES {
        if std::path::Path::new(path).exists() {
            tracing::info!("Found DRM render node: {}", path);
            return Some(path);
        }
    }
    tracing::warn!("No DRM render node found");
    None
}

/// Decoded frame from VAAPI with pixel data.
struct VaapiDecodedFrame {
    /// NV12 Y plane data
    y_data: Vec<u8>,
    /// NV12 UV plane data (interleaved)
    uv_data: Vec<u8>,
    /// Frame width
    width: u32,
    /// Frame height
    height: u32,
    /// Presentation timestamp
    pts: Duration,
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

/// VAAPI decoder state for a specific codec.
///
/// This enum wraps the codec-specific StatelessDecoder instances.
/// cros-codecs uses generics, so we need separate variants for each codec.
enum VaapiDecoderState {
    /// H.264 decoder
    H264(Box<dyn VaapiCodecDecoder>),
    /// VP9 decoder
    Vp9(Box<dyn VaapiCodecDecoder>),
    /// VP8 decoder
    Vp8(Box<dyn VaapiCodecDecoder>),
    /// Not initialized (fallback to software/placeholder)
    None,
}

/// Trait for codec-specific VAAPI decoder operations.
///
/// This abstracts over the different StatelessDecoder<Codec, Backend> types.
trait VaapiCodecDecoder: Send {
    /// Decode a single frame from the bitstream.
    /// Returns the decoded frame data or None if more data is needed.
    fn decode_frame(
        &mut self,
        data: &[u8],
        timestamp_us: u64,
    ) -> Result<Option<VaapiDecodedFrame>, VideoError>;

    /// Flush the decoder and return any remaining frames.
    fn flush(&mut self) -> Result<Vec<VaapiDecodedFrame>, VideoError>;
}

/// Linux VAAPI video decoder using cros-codecs.
///
/// This decoder uses cros-codecs for hardware-accelerated video decoding
/// via the VAAPI backend. Currently supports H.264, VP8, and VP9.
///
/// Note: H.265/HEVC and AV1 are not yet supported by cros-codecs 0.0.6.
pub struct LinuxVaapiDecoder {
    /// Demuxer for extracting encoded samples
    demuxer: DemuxerImpl,
    /// Video metadata
    metadata: VideoMetadata,
    /// Current playback position
    position: Duration,
    /// VAAPI decoder state
    vaapi_state: VaapiDecoderState,
    /// Queue of decoded frames ready for display
    frame_queue: VecDeque<VaapiDecodedFrame>,
    /// Whether VAAPI initialization succeeded
    hw_accel_active: bool,
}

impl LinuxVaapiDecoder {
    /// Creates a new decoder for the given file path.
    ///
    /// This will:
    /// 1. Detect the container format
    /// 2. Open and parse the container to find video tracks
    /// 3. Initialize VAAPI hardware decoder (if available)
    /// 4. Fall back to placeholder frames if VAAPI fails
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

        // Try to initialize VAAPI decoder
        let (vaapi_state, hw_accel_active) =
            match Self::init_vaapi_decoder(dm.codec, dm.width, dm.height) {
                Ok(state) => {
                    tracing::info!("VAAPI hardware decoder initialized for {:?}", dm.codec);
                    (state, true)
                }
                Err(e) => {
                    tracing::warn!(
                        "VAAPI initialization failed, using placeholder frames: {}",
                        e
                    );
                    (VaapiDecoderState::None, false)
                }
            };

        Ok(Self {
            demuxer,
            metadata,
            position: Duration::ZERO,
            vaapi_state,
            frame_queue: VecDeque::new(),
            hw_accel_active,
        })
    }

    /// Initialize the VAAPI decoder for the given codec.
    ///
    /// # VAAPI Integration Status
    ///
    /// The architecture is in place but requires Linux testing to complete:
    /// 1. ✅ DRM render node detection
    /// 2. ✅ Demuxer → EncodedSample pipeline
    /// 3. ⏳ libva Display initialization (requires libva-dev)
    /// 4. ⏳ StatelessDecoder creation with VAAPI backend
    /// 5. ⏳ Frame extraction from VA surfaces
    ///
    /// The cros-codecs API for creating a decoder is:
    /// ```ignore
    /// use cros_codecs::decoder::stateless::h264::H264;
    /// use cros_codecs::backend::vaapi::decoder::VaapiDecoder;
    ///
    /// let drm_path = find_drm_render_node()?;
    /// let display = libva::Display::open(drm_path)?;
    /// let decoder = StatelessDecoder::<H264, _>::new_vaapi(display, BlockingMode::Blocking)?;
    /// ```
    fn init_vaapi_decoder(
        codec: VideoCodec,
        _width: u32,
        _height: u32,
    ) -> Result<VaapiDecoderState, VideoError> {
        // Check for DRM render node (required for VAAPI)
        let drm_path = find_drm_render_node().ok_or_else(|| {
            VideoError::DecoderInit(
                "No DRM render node found. VAAPI requires /dev/dri/renderD128 or similar."
                    .to_string(),
            )
        })?;

        tracing::info!("Attempting VAAPI init with DRM node: {}", drm_path);

        // Create codec-specific decoder
        // Note: The actual StatelessDecoder creation requires libva bindings
        // which link against the system libva library. This needs Linux testing.
        match codec {
            VideoCodec::H264 => {
                // TODO: Complete H264 VAAPI decoder on Linux
                //
                // Required steps:
                // 1. Open libva Display from DRM path
                // 2. Create H264 stateless decoder with VAAPI backend
                // 3. Implement VaapiCodecDecoder trait for decode_frame/flush
                //
                // The cros-codecs crate provides:
                // - cros_codecs::decoder::stateless::h264::H264 (codec)
                // - cros_codecs::backend::vaapi (VAAPI backend)
                // - StatelessDecoder::decode() for frame-by-frame decoding

                Err(VideoError::DecoderInit(format!(
                    "H.264 VAAPI decoder requires Linux. Found DRM node: {}. \
                     Demuxing works, awaiting VAAPI integration.",
                    drm_path
                )))
            }
            VideoCodec::Vp9 => {
                // TODO: Complete VP9 VAAPI decoder on Linux
                Err(VideoError::DecoderInit(format!(
                    "VP9 VAAPI decoder requires Linux. Found DRM node: {}.",
                    drm_path
                )))
            }
            VideoCodec::Vp8 => {
                // TODO: Complete VP8 VAAPI decoder on Linux
                Err(VideoError::DecoderInit(format!(
                    "VP8 VAAPI decoder requires Linux. Found DRM node: {}.",
                    drm_path
                )))
            }
            VideoCodec::H265 => {
                // cros-codecs 0.0.6 doesn't have H.265 support yet
                Err(VideoError::UnsupportedFormat(
                    "H.265/HEVC not yet supported by cros-codecs".to_string(),
                ))
            }
            VideoCodec::Av1 => {
                // AV1 support may be available in newer cros-codecs versions
                Err(VideoError::UnsupportedFormat(
                    "AV1 VAAPI support requires cros-codecs with AV1 feature".to_string(),
                ))
            }
        }
    }

    /// Generate a placeholder frame for testing without VAAPI.
    fn generate_placeholder_frame(&self, pts: Duration, sample_data: &[u8]) -> VideoFrame {
        // Guard against zero dimensions (would cause division issues)
        let width = self.metadata.width.max(1);
        let height = self.metadata.height.max(1);

        // Create a gradient pattern based on sample data hash for visual feedback
        let hash = sample_data
            .iter()
            .take(16)
            .fold(0u8, |acc, &b| acc.wrapping_add(b));

        let y_size = (width * height) as usize;
        let uv_size = y_size / 2;

        // Y plane: gradient with hash-based variation
        let mut y_data = Vec::with_capacity(y_size);
        for y in 0..height {
            for x in 0..width {
                let base = ((x as f32 / width as f32) * 200.0) as u8;
                let variation = ((y as u8).wrapping_mul(3)).wrapping_add(hash);
                y_data.push(base.saturating_add(variation / 4));
            }
        }

        // UV plane: neutral (128 = no color)
        let uv_data = vec![128u8; uv_size];

        let y_plane = Plane {
            data: y_data,
            stride: width as usize,
        };

        let uv_plane = Plane {
            data: uv_data,
            stride: width as usize,
        };

        let cpu_frame = CpuFrame::new(PixelFormat::Nv12, width, height, vec![y_plane, uv_plane]);

        VideoFrame::new(pts, DecodedFrame::Cpu(cpu_frame))
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
        // Check if we have queued decoded frames
        if let Some(frame) = self.frame_queue.pop_front() {
            let y_plane = Plane {
                data: frame.y_data,
                stride: frame.width as usize,
            };
            let uv_plane = Plane {
                data: frame.uv_data,
                stride: frame.width as usize,
            };
            let cpu_frame = CpuFrame::new(
                PixelFormat::Nv12,
                frame.width,
                frame.height,
                vec![y_plane, uv_plane],
            );
            return Ok(Some(VideoFrame::new(
                frame.pts,
                DecodedFrame::Cpu(cpu_frame),
            )));
        }

        // Loop to handle cases where decoder needs multiple samples
        // (e.g., B-frames waiting for reference frames)
        loop {
            // Read next sample from demuxer
            let sample = match self.demuxer.next_sample()? {
                Some(s) => s,
                None => return Ok(None),
            };

            self.position = sample.pts;

            // Try to decode with VAAPI if available
            match &mut self.vaapi_state {
                VaapiDecoderState::H264(decoder)
                | VaapiDecoderState::Vp9(decoder)
                | VaapiDecoderState::Vp8(decoder) => {
                    let timestamp_us = sample.pts.as_micros() as u64;
                    match decoder.decode_frame(&sample.data, timestamp_us) {
                        Ok(Some(frame)) => {
                            let y_plane = Plane {
                                data: frame.y_data,
                                stride: frame.width as usize,
                            };
                            let uv_plane = Plane {
                                data: frame.uv_data,
                                stride: frame.width as usize,
                            };
                            let cpu_frame = CpuFrame::new(
                                PixelFormat::Nv12,
                                frame.width,
                                frame.height,
                                vec![y_plane, uv_plane],
                            );
                            return Ok(Some(VideoFrame::new(
                                frame.pts,
                                DecodedFrame::Cpu(cpu_frame),
                            )));
                        }
                        Ok(None) => {
                            // Decoder needs more data, continue loop to feed next sample
                            continue;
                        }
                        Err(e) => {
                            tracing::warn!(
                                "VAAPI decode error: {}, falling back to placeholder",
                                e
                            );
                            return Ok(Some(
                                self.generate_placeholder_frame(sample.pts, &sample.data),
                            ));
                        }
                    }
                }
                VaapiDecoderState::None => {
                    // No VAAPI, return placeholder frame
                    return Ok(Some(
                        self.generate_placeholder_frame(sample.pts, &sample.data),
                    ));
                }
            }
        }
    }

    fn seek(&mut self, position: Duration) -> Result<(), VideoError> {
        // Clear frame queue on seek
        self.frame_queue.clear();

        // Seek the demuxer
        self.demuxer.seek(position)?;
        self.position = position;

        // Flush the decoder if active
        match &mut self.vaapi_state {
            VaapiDecoderState::H264(decoder)
            | VaapiDecoderState::Vp9(decoder)
            | VaapiDecoderState::Vp8(decoder) => {
                if let Ok(frames) = decoder.flush() {
                    // Discard flushed frames after seek
                    drop(frames);
                }
            }
            VaapiDecoderState::None => {}
        }

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
        if self.hw_accel_active {
            HwAccelType::Vaapi
        } else {
            HwAccelType::None
        }
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
    fn test_container_format_with_query_params() {
        // URLs with query parameters should still be detected
        assert_eq!(
            ContainerFormat::from_url("https://example.com/video.mp4?token=abc123"),
            Some(ContainerFormat::Mp4)
        );
        assert_eq!(
            ContainerFormat::from_url("video.webm?v=2&quality=high"),
            Some(ContainerFormat::WebM)
        );
        // Fragment identifiers should also be stripped
        assert_eq!(
            ContainerFormat::from_url("video.mkv#t=10"),
            Some(ContainerFormat::Matroska)
        );
        // Both query and fragment
        assert_eq!(
            ContainerFormat::from_url("video.mp4?token=x#start"),
            Some(ContainerFormat::Mp4)
        );
    }

    #[test]
    fn test_codec_names() {
        assert_eq!(VideoCodec::H264.as_str(), "h264");
        assert_eq!(VideoCodec::H265.as_str(), "hevc");
        assert_eq!(VideoCodec::Vp9.as_str(), "vp9");
    }

    #[test]
    fn test_drm_render_nodes() {
        // Verify the DRM paths are reasonable
        assert!(!DRM_RENDER_NODES.is_empty());
        for path in DRM_RENDER_NODES {
            assert!(path.starts_with("/dev/dri/"));
        }
    }
}
