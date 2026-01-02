//! macOS hardware-accelerated video decoder using AVFoundation + VideoToolbox.
//!
//! This module provides zero-dependency video decoding on macOS using native Apple frameworks:
//! - **AVFoundation**: For streaming playback via AVPlayer
//! - **VideoToolbox**: For hardware-accelerated H.264/HEVC/VP9 decoding
//! - **CoreVideo**: For efficient pixel buffer handling
//!
//! VideoToolbox automatically uses the Apple GPU for decoding, providing excellent
//! performance and power efficiency on all Apple Silicon and Intel Macs.
//!
//! # Streaming Support
//!
//! This implementation uses AVPlayer + AVPlayerItemVideoOutput instead of AVAssetReader.
//! AVPlayer handles buffering automatically and supports streaming from remote URLs.
//! AVAssetReader is designed for offline processing and fails with "Operation Stopped"
//! for remote assets that aren't fully downloaded.
//!
//! # Thread Safety
//!
//! AVPlayer must be created on the main thread. This decoder checks for main thread
//! during initialization and fails if called from a background thread. The video_player.rs
//! module handles this by initializing macOS decoders synchronously on the main thread.
//! Frame polling via AVPlayerItemVideoOutput is thread-safe and works from any thread.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::MainThreadMarker;
use objc2_av_foundation::{
    AVMediaTypeVideo, AVPlayer, AVPlayerItem, AVPlayerItemStatus, AVPlayerItemVideoOutput,
};
use objc2_core_media::{CMTime, CMTimeFlags};
use objc2_core_video::{
    kCVPixelBufferPixelFormatTypeKey, kCVPixelFormatType_32BGRA, CVPixelBufferGetBaseAddress,
    CVPixelBufferGetBytesPerRow, CVPixelBufferGetHeight, CVPixelBufferGetPixelFormatType,
    CVPixelBufferGetWidth, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
    CVPixelBufferUnlockBaseAddress,
};
use objc2_foundation::{NSCopying, NSMutableDictionary, NSNumber, NSString, NSURL};

use super::video::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata,
};

/// macOS video decoder using AVPlayer and AVPlayerItemVideoOutput.
///
/// This decoder provides hardware-accelerated video decoding with streaming support
/// and automatic buffering for remote URLs.
///
/// # Thread Requirements
///
/// - `new()` MUST be called from the main thread (will fail otherwise)
/// - `decode_next()` can be called from any thread (frame polling is thread-safe)
/// - `seek()` can be called from any thread
pub struct MacOSVideoDecoder {
    /// AVPlayer for playback control
    player: Retained<AVPlayer>,
    /// Player item (kept alive for output and status)
    player_item: Retained<AVPlayerItem>,
    /// Video output for frame extraction (thread-safe)
    video_output: Retained<AVPlayerItemVideoOutput>,
    /// Cached metadata (updated once when ready, then immutable)
    /// Using UnsafeCell because the trait requires &VideoMetadata return
    /// Safety: Only written once during init, reads are safe after metadata_ready is true
    metadata: UnsafeCell<VideoMetadata>,
    /// Duration in seconds (updated when ready)
    duration_secs: Mutex<f64>,
    /// Whether EOF has been reached
    eof_reached: AtomicBool,
    /// Whether we've successfully extracted metadata
    metadata_ready: AtomicBool,
    /// Whether preview extraction is done (first pause marks end of preview)
    /// After preview, resume() will unmute audio
    preview_done: AtomicBool,
    /// Whether we're seeking and waiting for new frames (triggers buffering UI)
    seeking: AtomicBool,
}

// AVPlayerItemVideoOutput's copyPixelBuffer methods are thread-safe
// The player/player_item are only accessed from main thread operations
// We use atomic bools and mutex for cross-thread state
unsafe impl Send for MacOSVideoDecoder {}
unsafe impl Sync for MacOSVideoDecoder {}

impl MacOSVideoDecoder {
    /// Creates a new macOS video decoder for the given URL.
    ///
    /// # Thread Safety
    ///
    /// This method MUST be called from the main thread. It will return an error
    /// if called from a background thread. The video_player module handles this
    /// by initializing macOS decoders synchronously before spawning decode threads.
    ///
    /// # Non-blocking
    ///
    /// This method returns immediately without waiting for the video to be ready.
    /// The AVPlayer will buffer in the background. Frame polling in decode_next()
    /// will return None until frames are available.
    pub fn new(url: &str) -> Result<Self, VideoError> {
        tracing::info!("MacOSVideoDecoder: Opening {}", url);

        // Check that we're on the main thread
        let mtm = MainThreadMarker::new().ok_or_else(|| {
            VideoError::DecoderInit(
                "MacOSVideoDecoder must be initialized on the main thread. \
                 This is required by AVPlayer. The video player should call \
                 this synchronously before spawning the decode thread."
                    .to_string(),
            )
        })?;

        Self::init_on_main_thread(url, mtm)
    }

    /// Initialize AVPlayer on the main thread (non-blocking).
    ///
    /// Creates AVPlayer, AVPlayerItem, and AVPlayerItemVideoOutput with BGRA pixel format.
    /// Returns immediately without waiting for the player to become ready - buffering
    /// happens in the background and metadata is extracted lazily via `try_update_metadata`.
    fn init_on_main_thread(url: &str, mtm: MainThreadMarker) -> Result<Self, VideoError> {
        // Create NSURL
        let ns_url: Retained<NSURL> = if url.starts_with("http://") || url.starts_with("https://") {
            let ns_string = NSString::from_str(url);
            NSURL::URLWithString(&ns_string)
                .ok_or_else(|| VideoError::DecoderInit(format!("Invalid URL: {}", url)))?
        } else {
            let path = url.strip_prefix("file://").unwrap_or(url);
            let ns_string = NSString::from_str(path);
            NSURL::fileURLWithPath(&ns_string)
        };

        // Create AVPlayerItem
        let player_item = unsafe { AVPlayerItem::playerItemWithURL(&ns_url, mtm) };

        // Create video output with BGRA settings
        let output_settings = Self::create_output_settings();
        let settings_ptr = Retained::as_ptr(&output_settings)
            as *const objc2_foundation::NSDictionary<NSString, AnyObject>;
        let settings: &objc2_foundation::NSDictionary<NSString, AnyObject> =
            unsafe { &*settings_ptr };

        let video_output = unsafe {
            use objc2::AllocAnyThread;
            AVPlayerItemVideoOutput::initWithPixelBufferAttributes(
                AVPlayerItemVideoOutput::alloc(),
                Some(settings),
            )
        };

        // Add output to player item
        unsafe { player_item.addOutput(&video_output) };

        // Create player
        let player = unsafe { AVPlayer::playerWithPlayerItem(Some(&player_item), mtm) };

        // Mute initially to prevent audio during preview extraction
        // Will be unmuted when user clicks play
        unsafe { player.setMuted(true) };

        // Use placeholder metadata - will be updated when video is ready
        let metadata = VideoMetadata {
            width: 1920,
            height: 1080,
            duration: None,
            frame_rate: 30.0,
            codec: "videotoolbox".to_string(),
            pixel_aspect_ratio: 1.0,
        };

        tracing::info!("MacOSVideoDecoder: Created player (paused, waiting for play)");

        Ok(Self {
            player,
            player_item,
            video_output,
            metadata: UnsafeCell::new(metadata),
            duration_secs: Mutex::new(0.0),
            eof_reached: AtomicBool::new(false),
            metadata_ready: AtomicBool::new(false),
            preview_done: AtomicBool::new(false),
            seeking: AtomicBool::new(false),
        })
    }

    /// Try to extract metadata from the player item when it becomes ready.
    ///
    /// Called on each `decode_next()` until metadata is available. When the player
    /// reaches `ReadyToPlay` status, extracts video dimensions, frame rate, and duration.
    /// Uses atomic ordering to ensure metadata writes are visible to other threads.
    fn try_update_metadata(&self) {
        if self.metadata_ready.load(Ordering::Relaxed) {
            return;
        }

        let status = unsafe { self.player_item.status() };
        match status {
            AVPlayerItemStatus::ReadyToPlay => {
                // Extract real metadata now
                let duration_cm = unsafe { self.player_item.duration() };
                let duration = cmtime_to_duration(duration_cm);
                let duration_secs = cmtime_to_seconds(duration_cm);

                let asset = unsafe { self.player_item.asset() };
                let media_type = match unsafe { AVMediaTypeVideo } {
                    Some(mt) => mt,
                    None => return,
                };

                #[allow(deprecated)]
                let video_tracks = unsafe { asset.tracksWithMediaType(media_type) };

                if !video_tracks.is_empty() {
                    let video_track = video_tracks.objectAtIndex(0);
                    let natural_size = unsafe { video_track.naturalSize() };
                    let w = natural_size.width as u32;
                    let h = natural_size.height as u32;
                    let fps = unsafe { video_track.nominalFrameRate() };
                    let fps = if fps <= 0.0 { 30.0 } else { fps };

                    if w > 0 && h > 0 {
                        // Safety: Only called once, and metadata_ready acts as a barrier
                        // After this write completes and metadata_ready is set, no more writes occur
                        unsafe {
                            let meta = &mut *self.metadata.get();
                            meta.width = w;
                            meta.height = h;
                            meta.duration = duration;
                            meta.frame_rate = fps;
                        }

                        *self.duration_secs.lock().unwrap() = duration_secs;
                        self.metadata_ready.store(true, Ordering::Release);

                        tracing::info!(
                            "MacOSVideoDecoder: Video ready {}x{} @ {:.2}fps, duration: {:?}",
                            w,
                            h,
                            fps,
                            duration
                        );
                    }
                }
            }
            AVPlayerItemStatus::Failed => {
                let error = unsafe { self.player_item.error() };
                let error_msg = error
                    .map(|e| e.localizedDescription().to_string())
                    .unwrap_or_else(|| "Unknown error".to_string());
                tracing::error!("MacOSVideoDecoder: Player item failed: {}", error_msg);
            }
            _ => {
                // Still loading, do nothing
            }
        }
    }

    /// Create pixel buffer attribute dictionary for AVPlayerItemVideoOutput.
    ///
    /// Configures output to use 32-bit BGRA pixel format, which is then converted
    /// to RGBA during frame processing for compatibility with egui textures.
    fn create_output_settings() -> Retained<NSMutableDictionary<NSString, AnyObject>> {
        unsafe {
            let dict: Retained<NSMutableDictionary<NSString, AnyObject>> =
                NSMutableDictionary::new();

            let key_cfstring = kCVPixelBufferPixelFormatTypeKey;
            let pixel_format = NSNumber::numberWithUnsignedInt(kCVPixelFormatType_32BGRA);

            let key_ptr = key_cfstring as *const _ as *const NSString;
            let key: &NSString = &*key_ptr;
            let key_copying: &ProtocolObject<dyn NSCopying> = ProtocolObject::from_ref(key);

            let value_ptr = Retained::as_ptr(&pixel_format) as *mut AnyObject;
            let value: &AnyObject = &*value_ptr;

            dict.setObject_forKey(value, key_copying);
            dict
        }
    }
}

impl Drop for MacOSVideoDecoder {
    fn drop(&mut self) {
        // Pause playback before dropping
        unsafe { self.player.pause() };
    }
}

impl VideoDecoderBackend for MacOSVideoDecoder {
    fn open(url: &str) -> Result<Self, VideoError>
    where
        Self: Sized,
    {
        Self::new(url)
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        if self.eof_reached.load(Ordering::Relaxed) {
            return Ok(None);
        }

        // Try to update metadata if not ready yet
        self.try_update_metadata();

        // Get current playback time
        let current_time = unsafe { self.player.currentTime() };

        // Check for new frame (thread-safe operation)
        let has_new = unsafe { self.video_output.hasNewPixelBufferForItemTime(current_time) };

        if !has_new {
            // Check for EOF
            let duration_secs = *self.duration_secs.lock().unwrap();
            let current_secs = cmtime_to_seconds(current_time);
            if duration_secs > 0.0 && current_secs >= duration_secs - 0.1 {
                self.eof_reached.store(true, Ordering::Relaxed);
                return Ok(None);
            }
            // No frame ready yet (still buffering or between frames)
            return Ok(None);
        }

        // Copy pixel buffer (thread-safe operation)
        let mut actual_time = current_time;
        let pixel_buffer = unsafe {
            self.video_output
                .copyPixelBufferForItemTime_itemTimeForDisplay(
                    current_time,
                    &mut actual_time as *mut CMTime,
                )
        };

        let Some(pixel_buffer) = pixel_buffer else {
            return Ok(None);
        };

        // Process pixel buffer
        let pts = cmtime_to_duration(actual_time).unwrap_or(Duration::ZERO);

        let pixel_format = CVPixelBufferGetPixelFormatType(&pixel_buffer);
        if pixel_format != kCVPixelFormatType_32BGRA {
            return Err(VideoError::DecodeFailed(format!(
                "Unexpected pixel format: 0x{:08X}",
                pixel_format
            )));
        }

        let width = CVPixelBufferGetWidth(&pixel_buffer);
        let height = CVPixelBufferGetHeight(&pixel_buffer);

        // Check for overflow before allocation
        let pixel_count = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| VideoError::DecodeFailed("Dimensions too large".to_string()))?;

        let lock_result = unsafe {
            CVPixelBufferLockBaseAddress(&pixel_buffer, CVPixelBufferLockFlags::ReadOnly)
        };
        if lock_result != 0 {
            return Err(VideoError::DecodeFailed(format!(
                "Failed to lock pixel buffer: {}",
                lock_result
            )));
        }

        let bytes_per_row = CVPixelBufferGetBytesPerRow(&pixel_buffer);
        let base_address = CVPixelBufferGetBaseAddress(&pixel_buffer);

        if base_address.is_null() {
            unsafe {
                CVPixelBufferUnlockBaseAddress(&pixel_buffer, CVPixelBufferLockFlags::ReadOnly);
            }
            return Err(VideoError::DecodeFailed(
                "Null pixel buffer base address".to_string(),
            ));
        }

        // Convert BGRA to RGBA
        let data_size = bytes_per_row * height;
        let bgra_data = unsafe { std::slice::from_raw_parts(base_address as *const u8, data_size) };

        let mut rgba_data = Vec::with_capacity(pixel_count);
        for y in 0..height {
            let row_start = y * bytes_per_row;
            for x in 0..width {
                let pixel_start = row_start + x * 4;
                let b = bgra_data[pixel_start];
                let g = bgra_data[pixel_start + 1];
                let r = bgra_data[pixel_start + 2];
                let a = bgra_data[pixel_start + 3];
                rgba_data.push(r);
                rgba_data.push(g);
                rgba_data.push(b);
                rgba_data.push(a);
            }
        }

        unsafe {
            CVPixelBufferUnlockBaseAddress(&pixel_buffer, CVPixelBufferLockFlags::ReadOnly);
        }

        let cpu_frame = CpuFrame::new(
            PixelFormat::Rgba,
            width as u32,
            height as u32,
            vec![Plane {
                data: rgba_data,
                stride: width * 4,
            }],
        );

        // Clear seeking flag - we have a frame, buffering is done
        self.seeking.store(false, Ordering::Relaxed);

        Ok(Some(VideoFrame::new(pts, DecodedFrame::Cpu(cpu_frame))))
    }

    fn seek(&mut self, position: Duration) -> Result<(), VideoError> {
        // Mark as seeking to trigger buffering UI until frames arrive
        self.seeking.store(true, Ordering::Relaxed);
        let seek_time = duration_to_cmtime(position);
        unsafe { self.player.seekToTime(seek_time) };
        self.eof_reached.store(false, Ordering::Relaxed);
        tracing::debug!("MacOSVideoDecoder: seeking to {:?}", position);
        Ok(())
    }

    /// Pause AVPlayer playback.
    ///
    /// The first pause marks the end of preview extraction phase.
    fn pause(&mut self) -> Result<(), VideoError> {
        unsafe { self.player.pause() };
        // First pause marks end of preview - subsequent resumes will unmute
        self.preview_done.store(true, Ordering::Relaxed);
        tracing::debug!("MacOSVideoDecoder: paused");
        Ok(())
    }

    /// Resume AVPlayer playback.
    ///
    /// Unmutes audio only after preview is done (first pause has occurred).
    fn resume(&mut self) -> Result<(), VideoError> {
        // Only unmute after preview phase (first pause marks end of preview)
        if self.preview_done.load(Ordering::Relaxed) {
            unsafe { self.player.setMuted(false) };
            tracing::debug!("MacOSVideoDecoder: unmuted for playback");
        }
        unsafe { self.player.play() };
        tracing::debug!("MacOSVideoDecoder: resumed/playing");
        Ok(())
    }

    /// Set muted state for AVPlayer audio.
    fn set_muted(&mut self, muted: bool) -> Result<(), VideoError> {
        unsafe { self.player.setMuted(muted) };
        tracing::debug!("MacOSVideoDecoder: muted={}", muted);
        Ok(())
    }

    /// Returns buffering percentage based on player state.
    ///
    /// Returns 0 when:
    /// - Waiting for player to become ready (initial load)
    /// - Seeking and waiting for new frames
    ///
    /// Returns 100 when ready to play.
    fn buffering_percent(&self) -> i32 {
        // Show buffering during initial load or after seek
        if !self.metadata_ready.load(Ordering::Relaxed) || self.seeking.load(Ordering::Relaxed) {
            0
        } else {
            100
        }
    }

    fn metadata(&self) -> &VideoMetadata {
        // Safety: metadata is only written once during try_update_metadata()
        // The metadata_ready atomic with Release/Acquire ordering ensures
        // that reads after metadata_ready is true see the complete write
        unsafe { &*self.metadata.get() }
    }

    fn hw_accel_type(&self) -> HwAccelType {
        HwAccelType::VideoToolbox
    }
}

fn cmtime_to_duration(time: CMTime) -> Option<Duration> {
    if time.timescale <= 0 {
        return None;
    }
    let seconds = time.value as f64 / time.timescale as f64;
    if seconds < 0.0 {
        return None;
    }
    Some(Duration::from_secs_f64(seconds))
}

fn cmtime_to_seconds(time: CMTime) -> f64 {
    if time.timescale <= 0 {
        return 0.0;
    }
    time.value as f64 / time.timescale as f64
}

fn duration_to_cmtime(duration: Duration) -> CMTime {
    let timescale: i32 = 600;
    let value = (duration.as_secs_f64() * timescale as f64) as i64;
    CMTime {
        value,
        timescale,
        flags: CMTimeFlags::Valid,
        epoch: 0,
    }
}
