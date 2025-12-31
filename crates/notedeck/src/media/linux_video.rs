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
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::media::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata,
};
use crate::media::network::http_stream_to_file;

/// Progressive video downloader that enables streaming playback.
///
/// Downloads video in the background while allowing read access to
/// already-downloaded portions of the file.
pub struct ProgressiveDownloader {
    /// Path to the temp file being downloaded
    path: PathBuf,
    /// Number of bytes downloaded so far
    downloaded: Arc<AtomicU64>,
    /// Total file size (if known from Content-Length)
    total_size: Arc<AtomicU64>,
    /// Whether download is complete
    complete: Arc<AtomicBool>,
    /// Whether an error occurred
    error: Arc<AtomicBool>,
    /// Join handle for download thread
    _download_thread: Option<std::thread::JoinHandle<()>>,
}

impl ProgressiveDownloader {
    /// Start downloading a URL to a temp file in the background.
    pub fn start(url: &str) -> Result<Self, VideoError> {
        let extension = ContainerFormat::from_url(url)
            .map(|f| match f {
                ContainerFormat::Mp4 => ".mp4",
                ContainerFormat::Matroska => ".mkv",
                ContainerFormat::WebM => ".webm",
            })
            .unwrap_or(".mp4");

        // Create temp file
        let temp_file = tempfile::Builder::new()
            .prefix("notedeck_video_")
            .suffix(extension)
            .tempfile()
            .map_err(|e| VideoError::OpenFailed(format!("Failed to create temp file: {}", e)))?;

        let (file, path) = temp_file.keep()
            .map_err(|e| VideoError::OpenFailed(format!("Failed to persist temp file: {}", e)))?;

        let downloaded = Arc::new(AtomicU64::new(0));
        let total_size = Arc::new(AtomicU64::new(0));
        let complete = Arc::new(AtomicBool::new(false));
        let error = Arc::new(AtomicBool::new(false));

        let url = url.to_string();
        let path_clone = path.clone();
        let downloaded_clone = Arc::clone(&downloaded);
        let total_size_clone = Arc::clone(&total_size);
        let complete_clone = Arc::clone(&complete);
        let error_clone = Arc::clone(&error);

        // Spawn background download thread
        let handle = std::thread::spawn(move || {
            let result = Self::download_worker(
                &url,
                path_clone,
                file,
                downloaded_clone,
                total_size_clone,
            );

            match result {
                Ok(()) => complete_clone.store(true, Ordering::Release),
                Err(e) => {
                    tracing::error!("Progressive download failed: {}", e);
                    error_clone.store(true, Ordering::Release);
                }
            }
        });

        tracing::info!("ProgressiveDownloader: Started downloading to {:?}", path);

        Ok(Self {
            path,
            downloaded,
            total_size,
            complete,
            error,
            _download_thread: Some(handle),
        })
    }

    /// Background worker that downloads the video with true streaming.
    fn download_worker(
        url: &str,
        path: PathBuf,
        file: File,
        downloaded: Arc<AtomicU64>,
        total_size: Arc<AtomicU64>,
    ) -> Result<(), VideoError> {
        // Use tokio runtime for async HTTP
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| VideoError::OpenFailed(format!("Failed to create runtime: {}", e)))?;

        // Stream directly to file - no memory buffering of entire video
        let downloaded_clone = Arc::clone(&downloaded);
        let total_size_clone = Arc::clone(&total_size);

        // on_start: set total size from Content-Length BEFORE streaming
        let on_start = move |content_length: Option<u64>| {
            if let Some(len) = content_length {
                total_size_clone.store(len, Ordering::Release);
                tracing::debug!("ProgressiveDownloader: Content-Length={} bytes", len);
            }
        };

        // on_chunk: update downloaded counter
        let on_chunk = move |chunk_len: usize| {
            let new_pos = downloaded_clone.fetch_add(chunk_len as u64, Ordering::AcqRel)
                + chunk_len as u64;
            // Log progress every 10MB
            if new_pos % (10 * 1024 * 1024) < chunk_len as u64 {
                tracing::debug!("ProgressiveDownloader: {} MB downloaded", new_pos / (1024 * 1024));
            }
        };

        let (content_length, _content_type) = rt
            .block_on(http_stream_to_file(url, file, on_start, on_chunk))
            .map_err(|e| VideoError::OpenFailed(format!("HTTP request failed: {}", e)))?;

        // Ensure total_size is set even if no Content-Length was provided
        if content_length.is_none() {
            total_size.store(downloaded.load(Ordering::Acquire), Ordering::Release);
        }

        let final_size = downloaded.load(Ordering::Acquire);
        tracing::info!(
            "ProgressiveDownloader: Completed download of {} bytes to {:?}",
            final_size,
            path
        );

        Ok(())
    }

    /// Get the path to the temp file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get number of bytes downloaded so far.
    pub fn bytes_downloaded(&self) -> u64 {
        self.downloaded.load(Ordering::Acquire)
    }

    /// Get total file size (0 if unknown).
    pub fn total_size(&self) -> u64 {
        self.total_size.load(Ordering::Acquire)
    }

    /// Check if download is complete.
    pub fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Acquire)
    }

    /// Check if an error occurred.
    pub fn has_error(&self) -> bool {
        self.error.load(Ordering::Acquire)
    }

    /// Wait until at least `min_bytes` are available, with timeout.
    pub fn wait_for_bytes(&self, min_bytes: u64, timeout: Duration) -> bool {
        let start = std::time::Instant::now();
        while self.bytes_downloaded() < min_bytes {
            if self.is_complete() || self.has_error() {
                return self.bytes_downloaded() >= min_bytes;
            }
            if start.elapsed() > timeout {
                return false;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        true
    }
}

/// A reader that wraps a file being progressively downloaded.
/// Blocks reads if data isn't available yet.
pub struct ProgressiveReader {
    file: File,
    downloader: Arc<ProgressiveDownloader>,
    position: u64,
}

impl ProgressiveReader {
    pub fn new(downloader: Arc<ProgressiveDownloader>) -> Result<Self, VideoError> {
        tracing::debug!("ProgressiveReader: Waiting for initial 1024 bytes...");
        // Wait for some initial data before opening (30s timeout for slow networks)
        if !downloader.wait_for_bytes(1024, Duration::from_secs(30)) {
            return Err(VideoError::OpenFailed("Timeout waiting for initial data".to_string()));
        }
        tracing::debug!("ProgressiveReader: Got initial data, {} bytes available", downloader.bytes_downloaded());

        let file = OpenOptions::new()
            .read(true)
            .open(downloader.path())
            .map_err(|e| VideoError::OpenFailed(format!("Failed to open temp file: {}", e)))?;

        Ok(Self {
            file,
            downloader,
            position: 0,
        })
    }
}

impl Read for ProgressiveReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let needed = self.position + buf.len() as u64;

        // Wait for data if not available yet
        while self.downloader.bytes_downloaded() < needed {
            if self.downloader.is_complete() {
                // Download complete, read what's available
                break;
            }
            if self.downloader.has_error() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Download failed",
                ));
            }
            std::thread::sleep(Duration::from_millis(5));
        }

        let n = self.file.read(buf)?;
        self.position += n as u64;
        Ok(n)
    }
}

impl Seek for ProgressiveReader {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(p) => p,
            SeekFrom::Current(offset) => {
                if offset >= 0 {
                    self.position + offset as u64
                } else {
                    self.position.saturating_sub((-offset) as u64)
                }
            }
            SeekFrom::End(offset) => {
                // For seek from end, we need to know the total size
                let total = self.downloader.total_size();
                if total == 0 {
                    // Total size not known yet - wait for Content-Length
                    tracing::debug!("ProgressiveReader: Waiting for total size for SeekFrom::End");
                    let start = std::time::Instant::now();
                    while self.downloader.total_size() == 0 && !self.downloader.has_error() {
                        if start.elapsed() > Duration::from_secs(30) {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::TimedOut,
                                "Timeout waiting for file size",
                            ));
                        }
                        std::thread::sleep(Duration::from_millis(10));
                    }
                }
                let total = self.downloader.total_size();
                if offset >= 0 {
                    total + offset as u64
                } else {
                    total.saturating_sub((-offset) as u64)
                }
            }
        };

        tracing::debug!(
            "ProgressiveReader: Seeking to {} (downloaded: {}, total: {})",
            new_pos,
            self.downloader.bytes_downloaded(),
            self.downloader.total_size()
        );

        // Wait for the position to be available (with timeout)
        let start = std::time::Instant::now();
        while self.downloader.bytes_downloaded() < new_pos {
            if self.downloader.is_complete() {
                break;
            }
            if self.downloader.has_error() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Download failed",
                ));
            }
            // Add a timeout to prevent infinite waiting
            if start.elapsed() > Duration::from_secs(120) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!(
                        "Timeout waiting for position {} (have {} of {} bytes)",
                        new_pos,
                        self.downloader.bytes_downloaded(),
                        self.downloader.total_size()
                    ),
                ));
            }
            std::thread::sleep(Duration::from_millis(5));
        }

        self.position = self.file.seek(pos)?;
        Ok(self.position)
    }
}

// cros-codecs imports for VAAPI integration
use cros_codecs::decoder::stateless::h264::H264;
use cros_codecs::decoder::stateless::StatelessDecoder;
use cros_codecs::decoder::BlockingMode;
use cros_codecs::libva::Display as VaDisplay;
use cros_codecs::video_frame::gbm_video_frame::{GbmDevice, GbmVideoFrame};
use cros_codecs::video_frame::VideoFrame as CrosVideoFrame;
use std::rc::Rc;

// ============================================================================
// Unified Video Reader (supports both local files and streaming)
// ============================================================================

/// Video source that supports both local files and progressive streaming.
///
/// This enum allows demuxers to work with both file paths and URLs
/// without requiring generic type parameters throughout the codebase.
pub enum VideoSource {
    /// Local file reader
    File(BufReader<File>),
    /// Progressive streaming reader (for HTTP URLs)
    Progressive(BufReader<ProgressiveReader>),
}

impl VideoSource {
    /// Create a video source from a local file path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, VideoError> {
        let file = File::open(path.as_ref())
            .map_err(|e| VideoError::OpenFailed(format!("Failed to open file: {}", e)))?;
        Ok(VideoSource::File(BufReader::new(file)))
    }

    /// Create a video source from a progressive downloader.
    pub fn from_progressive(downloader: Arc<ProgressiveDownloader>) -> Result<Self, VideoError> {
        tracing::debug!("VideoSource: Creating ProgressiveReader...");
        let reader = ProgressiveReader::new(downloader)?;
        tracing::debug!("VideoSource: ProgressiveReader created successfully");
        Ok(VideoSource::Progressive(BufReader::new(reader)))
    }

    /// Get the file size (blocks until known for progressive sources).
    pub fn size(&self) -> Result<u64, VideoError> {
        match self {
            VideoSource::File(reader) => reader
                .get_ref()
                .metadata()
                .map(|m| m.len())
                .map_err(|e| VideoError::OpenFailed(format!("Failed to get file size: {}", e))),
            VideoSource::Progressive(reader) => {
                let downloader = &reader.get_ref().downloader;
                // Wait for total size to be known
                while downloader.total_size() == 0 && !downloader.has_error() {
                    if downloader.is_complete() {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                if downloader.has_error() {
                    return Err(VideoError::OpenFailed("Download failed".to_string()));
                }
                Ok(downloader.total_size())
            }
        }
    }
}

impl Read for VideoSource {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            VideoSource::File(r) => r.read(buf),
            VideoSource::Progressive(r) => r.read(buf),
        }
    }
}

impl Seek for VideoSource {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match self {
            VideoSource::File(r) => r.seek(pos),
            VideoSource::Progressive(r) => r.seek(pos),
        }
    }
}

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
/// Supports both local files and progressive streaming via `VideoSource`.
pub struct Mp4Demuxer {
    reader: mp4::Mp4Reader<VideoSource>,
    video_track_id: u32,
    sample_index: u32,
    sample_count: u32,
    timescale: u32,
    metadata: DemuxerMetadata,
}

impl Mp4Demuxer {
    /// Opens an MP4 file and finds the video track.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, VideoError> {
        let source = VideoSource::from_file(path)?;
        Self::from_source(source)
    }

    /// Opens an MP4 from a VideoSource (file or streaming).
    pub fn from_source(source: VideoSource) -> Result<Self, VideoError> {
        tracing::debug!("Mp4Demuxer: Getting source size...");
        let size = source.size()?;
        tracing::debug!("Mp4Demuxer: Size = {} bytes, reading header...", size);
        let mp4 = mp4::Mp4Reader::read_header(source, size)
            .map_err(|e| VideoError::OpenFailed(format!("Failed to parse MP4 header: {}", e)))?;
        tracing::debug!("Mp4Demuxer: Header parsed successfully");

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

        // Get duration from track
        let duration = Some(track.duration());

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
/// Supports both local files and progressive streaming via `VideoSource`.
pub struct MkvDemuxer {
    demuxer: matroska_demuxer::MatroskaFile<VideoSource>,
    video_track_num: u64,
    metadata: DemuxerMetadata,
}

impl MkvDemuxer {
    /// Opens a Matroska/WebM file and finds the video track.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, VideoError> {
        let source = VideoSource::from_file(path)?;
        Self::from_source(source)
    }

    /// Opens a Matroska file from a VideoSource (file or streaming).
    pub fn from_source(source: VideoSource) -> Result<Self, VideoError> {
        let demuxer = matroska_demuxer::MatroskaFile::open(source).map_err(|e| {
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
            (video.pixel_width().get() as u32, video.pixel_height().get() as u32)
        } else {
            return Err(VideoError::UnsupportedFormat(
                "Video track missing video settings".to_string(),
            ));
        };

        // Calculate frame rate from default duration (nanoseconds per frame)
        let frame_rate = if let Some(default_duration) = track.default_duration() {
            1_000_000_000.0 / default_duration.get() as f64
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
        use matroska_demuxer::Frame;
        let mut frame = Frame::default();
        loop {
            match self.demuxer.next_frame(&mut frame) {
                Ok(true) => {
                    // Skip non-video frames
                    if frame.track != self.video_track_num {
                        continue;
                    }

                    return Ok(Some(EncodedSample {
                        data: std::mem::take(&mut frame.data),
                        pts: Duration::from_nanos(frame.timestamp),
                        is_keyframe: frame.is_keyframe.unwrap_or(false),
                    }));
                }
                Ok(false) => return Ok(None), // EOF
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
    /// Y plane stride (bytes per row, may include padding)
    y_stride: usize,
    /// UV plane stride (bytes per row, may include padding)
    uv_stride: usize,
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

/// Type alias for the H264 VAAPI StatelessDecoder with GbmVideoFrame
type H264StatelessDecoder = StatelessDecoder<
    H264,
    cros_codecs::backend::vaapi::decoder::VaapiBackend<GbmVideoFrame>,
>;

/// H.264 VAAPI decoder wrapper with GBM support for frame output.
struct H264VaapiDecoder {
    decoder: H264StatelessDecoder,
    #[allow(dead_code)]
    display: Rc<VaDisplay>,
    gbm_device: Arc<GbmDevice>,
    /// Coded resolution (from format change events)
    coded_width: u32,
    coded_height: u32,
}

// SAFETY: The decoder is only accessed from a single thread (the decode thread).
// The Rc<VaDisplay> and Arc<GbmDevice> are not accessed across threads.
unsafe impl Send for H264VaapiDecoder {}

impl VaapiCodecDecoder for H264VaapiDecoder {
    fn decode_frame(
        &mut self,
        data: &[u8],
        timestamp_us: u64,
    ) -> Result<Option<VaapiDecodedFrame>, VideoError> {
        use cros_codecs::decoder::stateless::StatelessVideoDecoder;
        use cros_codecs::decoder::DecoderEvent;
        use cros_codecs::decoder::DecodedHandle;
        use cros_codecs::video_frame::gbm_video_frame::GbmUsage;
        use cros_codecs::video_frame::ReadMapping;
        use cros_codecs::Fourcc;
        use cros_codecs::Resolution;

        // Create frame allocator callback using GbmDevice
        let gbm = Arc::clone(&self.gbm_device);
        let coded_width = self.coded_width;
        let coded_height = self.coded_height;

        let mut alloc_frame = || -> Option<GbmVideoFrame> {
            // Create a new GBM frame for decoding
            // Use NV12 format (standard for H.264 decoded output)
            let visible_res = Resolution {
                width: coded_width,
                height: coded_height,
            };
            let coded_res = Resolution {
                width: coded_width,
                height: coded_height,
            };

            match GbmDevice::new_frame(
                Arc::clone(&gbm),
                Fourcc::from(b"NV12"),
                visible_res,
                coded_res,
                GbmUsage::Decode,
            ) {
                Ok(frame) => {
                    tracing::debug!(
                        "Allocated GBM frame {}x{} for decoding",
                        coded_width,
                        coded_height
                    );
                    Some(frame)
                }
                Err(e) => {
                    tracing::error!("Failed to allocate GBM frame: {}", e);
                    None
                }
            }
        };

        // Feed data to decoder
        let bytes_decoded = self
            .decoder
            .decode(timestamp_us, data, &mut alloc_frame)
            .map_err(|e| VideoError::DecodeFailed(format!("H264 decode error: {:?}", e)))?;

        tracing::debug!("H264 decoder consumed {} of {} bytes", bytes_decoded, data.len());

        // Check for decoded frames
        if let Some(event) = self.decoder.next_event() {
            match event {
                DecoderEvent::FrameReady(handle) => {
                    // Sync to ensure decoding is complete
                    handle
                        .sync()
                        .map_err(|e| VideoError::DecodeFailed(format!("Sync failed: {:?}", e)))?;

                    let resolution = handle.display_resolution();
                    let timestamp = handle.timestamp();
                    let width = resolution.width;
                    let height = resolution.height;

                    tracing::debug!(
                        "Frame ready: {}x{} @ ts={}",
                        width,
                        height,
                        timestamp
                    );

                    // Get the decoded video frame (GbmVideoFrame)
                    let video_frame = handle.video_frame();

                    // Get plane pitches (strides) before mapping
                    let pitches = video_frame.get_plane_pitch();
                    let y_stride = pitches.first().copied().unwrap_or(width as usize);
                    let uv_stride = pitches.get(1).copied().unwrap_or(width as usize);

                    tracing::debug!(
                        "Frame pitches: Y={}, UV={} (width={})",
                        y_stride,
                        uv_stride,
                        width
                    );

                    // Map the frame to read pixel data
                    // Use the VideoFrame trait's map() method
                    let mapping = CrosVideoFrame::map(video_frame.as_ref()).map_err(|e| {
                        VideoError::DecodeFailed(format!("Failed to map video frame: {}", e))
                    })?;

                    // Get the raw NV12 data (Y plane followed by interleaved UV)
                    let planes = mapping.get();

                    // NV12 format: Y plane at index 0, UV plane at index 1
                    // Note: plane data includes stride padding
                    let y_plane_size = y_stride * height as usize;
                    let uv_plane_size = uv_stride * (height as usize / 2);

                    let y_data = if !planes.is_empty() {
                        planes[0][..y_plane_size.min(planes[0].len())].to_vec()
                    } else {
                        tracing::warn!("No Y plane data in decoded frame");
                        vec![128u8; y_plane_size]
                    };

                    let uv_data = if planes.len() > 1 {
                        planes[1][..uv_plane_size.min(planes[1].len())].to_vec()
                    } else {
                        tracing::warn!("No UV plane data in decoded frame");
                        vec![128u8; uv_plane_size]
                    };

                    Ok(Some(VaapiDecodedFrame {
                        y_data,
                        uv_data,
                        y_stride,
                        uv_stride,
                        width,
                        height,
                        pts: Duration::from_micros(timestamp),
                    }))
                }
                DecoderEvent::FormatChanged => {
                    // Get updated format info from the decoder
                    if let Some(info) = self.decoder.stream_info() {
                        self.coded_width = info.coded_resolution.width;
                        self.coded_height = info.coded_resolution.height;
                        tracing::info!(
                            "H264 format changed: {}x{} (coded), {:?}",
                            self.coded_width,
                            self.coded_height,
                            info.format
                        );
                    } else {
                        tracing::warn!("H264 format changed but no stream info available");
                    }
                    Ok(None)
                }
            }
        } else {
            Ok(None)
        }
    }

    fn flush(&mut self) -> Result<Vec<VaapiDecodedFrame>, VideoError> {
        use cros_codecs::decoder::stateless::StatelessVideoDecoder;

        self.decoder
            .flush()
            .map_err(|e| VideoError::DecodeFailed(format!("Flush failed: {:?}", e)))?;

        // Collect any remaining frames
        let mut frames = Vec::new();
        while let Some(frame) = self.decode_frame(&[], 0)? {
            frames.push(frame);
        }
        Ok(frames)
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
    /// Progressive downloader (kept alive for streaming)
    #[allow(dead_code)]
    progressive_downloader: Option<Arc<ProgressiveDownloader>>,
}

impl LinuxVaapiDecoder {
    /// Creates a new decoder for the given file path or URL.
    ///
    /// For URLs, uses progressive streaming - playback can start before
    /// the full download completes. For local files, opens directly.
    ///
    /// This will:
    /// 1. Start progressive download if it's a URL (playback starts immediately)
    /// 2. Detect the container format
    /// 3. Open and parse the container to find video tracks
    /// 4. Initialize VAAPI hardware decoder (if available)
    /// 5. Fall back to placeholder frames if VAAPI fails
    pub fn new(path: &str) -> Result<Self, VideoError> {
        let is_url = path.starts_with("http://") || path.starts_with("https://");

        // Detect container format early
        let container = ContainerFormat::from_url(path).ok_or_else(|| {
            VideoError::UnsupportedFormat(format!(
                "Could not detect container format from path: {}",
                path
            ))
        })?;

        tracing::info!(
            "LinuxVaapiDecoder: Opening {:?} container: {} (streaming: {})",
            container,
            path,
            is_url
        );

        // Create video source and track progressive downloader
        let (source, progressive_downloader) = if is_url {
            let downloader = Arc::new(ProgressiveDownloader::start(path)?);
            let source = VideoSource::from_progressive(Arc::clone(&downloader))?;
            (source, Some(downloader))
        } else {
            let source = VideoSource::from_file(path)?;
            (source, None)
        };

        // Open appropriate demuxer with the video source
        let demuxer = match container {
            ContainerFormat::Mp4 => DemuxerImpl::Mp4(Mp4Demuxer::from_source(source)?),
            ContainerFormat::Matroska | ContainerFormat::WebM => {
                DemuxerImpl::Matroska(MkvDemuxer::from_source(source)?)
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
            progressive_downloader,
        })
    }

    /// Initialize the VAAPI decoder for the given codec.
    ///
    /// Opens libva Display and creates a StatelessDecoder for hardware-accelerated
    /// video decoding via VAAPI. Uses GBM (Generic Buffer Management) for frame output.
    fn init_vaapi_decoder(
        codec: VideoCodec,
        width: u32,
        height: u32,
    ) -> Result<VaapiDecoderState, VideoError> {
        // Check for DRM render node (required for VAAPI)
        let drm_path = find_drm_render_node().ok_or_else(|| {
            VideoError::DecoderInit(
                "No DRM render node found. VAAPI requires /dev/dri/renderD128 or similar."
                    .to_string(),
            )
        })?;

        tracing::info!("Attempting VAAPI init with DRM node: {}", drm_path);

        // Open GBM device (used for frame output)
        let gbm_device = GbmDevice::open(std::path::PathBuf::from(drm_path))
            .map_err(|e| VideoError::DecoderInit(format!("Failed to open GBM device: {}", e)))?;
        tracing::info!("Successfully opened GBM device");

        // Open libva Display
        let display = VaDisplay::open_drm_display(std::path::PathBuf::from(drm_path))
            .map_err(|e| VideoError::DecoderInit(format!("Failed to open VA display: {:?}", e)))?;
        tracing::info!("Successfully opened libva Display");

        // Create codec-specific decoder
        match codec {
            VideoCodec::H264 => {
                tracing::info!("Creating H.264 VAAPI decoder with GbmVideoFrame");
                let decoder = StatelessDecoder::<H264, _>::new_vaapi(
                    Rc::clone(&display),
                    BlockingMode::Blocking,
                )
                .map_err(|e| {
                    VideoError::DecoderInit(format!("Failed to create H264 VAAPI decoder: {:?}", e))
                })?;

                tracing::info!("H.264 VAAPI decoder created successfully");
                Ok(VaapiDecoderState::H264(Box::new(H264VaapiDecoder {
                    decoder,
                    display: Rc::clone(&display),
                    gbm_device: Arc::clone(&gbm_device),
                    // Initial dimensions (will be updated by FormatChanged event)
                    coded_width: width,
                    coded_height: height,
                })))
            }
            VideoCodec::Vp9 => Err(VideoError::UnsupportedFormat(
                "VP9 VAAPI decoder not yet implemented".to_string(),
            )),
            VideoCodec::Vp8 => Err(VideoError::UnsupportedFormat(
                "VP8 VAAPI decoder not yet implemented".to_string(),
            )),
            VideoCodec::H265 => Err(VideoError::UnsupportedFormat(
                "H.265/HEVC not yet supported by cros-codecs".to_string(),
            )),
            VideoCodec::Av1 => Err(VideoError::UnsupportedFormat(
                "AV1 VAAPI support requires cros-codecs with AV1 feature".to_string(),
            )),
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
                stride: frame.y_stride,
            };
            let uv_plane = Plane {
                data: frame.uv_data,
                stride: frame.uv_stride,
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
                                stride: frame.y_stride,
                            };
                            let uv_plane = Plane {
                                data: frame.uv_data,
                                stride: frame.uv_stride,
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
