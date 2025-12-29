//! macOS hardware-accelerated video decoder using AVFoundation + VideoToolbox.
//!
//! This module provides zero-dependency video decoding on macOS using native Apple frameworks:
//! - **AVFoundation**: For demuxing (reading video containers like MP4, MOV)
//! - **VideoToolbox**: For hardware-accelerated H.264/HEVC/VP9 decoding
//! - **CoreVideo**: For efficient pixel buffer handling
//!
//! VideoToolbox automatically uses the Apple GPU for decoding, providing excellent
//! performance and power efficiency on all Apple Silicon and Intel Macs.
//!
//! # Thread Safety
//!
//! AVFoundation objects are not thread-safe. This module uses a dedicated decode thread
//! that owns all AVFoundation objects and communicates via channels, making the public
//! `MacOSVideoDecoder` struct `Send` + `Sync`.

use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use objc2::rc::Retained;
use objc2_av_foundation::{
    AVAssetReader, AVAssetReaderStatus, AVAssetReaderTrackOutput, AVMediaTypeVideo, AVURLAsset,
};
use objc2_core_media::{CMSampleBufferGetPresentationTimeStamp, CMTime, CMTimeFlags};
use objc2_core_video::{
    CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow, CVPixelBufferGetHeight,
    CVPixelBufferGetWidth, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
    CVPixelBufferUnlockBaseAddress,
};
use objc2_foundation::{NSString, NSURL};

use super::video::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata,
};

/// Commands sent to the decode thread.
enum DecodeCommand {
    /// Request next frame
    DecodeNext,
    /// Seek to position
    Seek(Duration),
    /// Shutdown the decode thread
    Shutdown,
}

/// Response from the decode thread.
enum DecodeResponse {
    /// Successfully decoded a frame
    Frame(VideoFrame),
    /// End of stream reached
    EndOfStream,
    /// Error occurred
    Error(VideoError),
    /// Seek completed (success or error)
    SeekComplete(Result<(), VideoError>),
    /// Metadata available
    Metadata(VideoMetadata),
}

/// macOS video decoder using AVFoundation and VideoToolbox.
///
/// This decoder provides hardware-accelerated video decoding with zero external
/// dependencies on macOS. The actual decoding runs on a dedicated thread to
/// handle AVFoundation's thread-safety requirements.
pub struct MacOSVideoDecoder {
    /// Channel to send commands to decode thread
    cmd_tx: Sender<DecodeCommand>,
    /// Channel to receive responses from decode thread
    resp_rx: Receiver<DecodeResponse>,
    /// Handle to the decode thread
    thread_handle: Option<JoinHandle<()>>,
    /// Cached metadata
    metadata: VideoMetadata,
    /// Whether EOF has been reached
    eof_reached: bool,
}

impl MacOSVideoDecoder {
    /// Creates a new macOS video decoder for the given URL.
    pub fn new(url: &str) -> Result<Self, VideoError> {
        let (cmd_tx, cmd_rx) = mpsc::channel::<DecodeCommand>();
        let (resp_tx, resp_rx) = mpsc::channel::<DecodeResponse>();

        let url_owned = url.to_string();

        // Spawn decode thread
        let thread_handle = thread::spawn(move || {
            decode_thread_main(url_owned, cmd_rx, resp_tx);
        });

        // Wait for metadata response
        let metadata = match resp_rx.recv() {
            Ok(DecodeResponse::Metadata(m)) => m,
            Ok(DecodeResponse::Error(e)) => return Err(e),
            Ok(_) => return Err(VideoError::DecoderInit("Unexpected response".to_string())),
            Err(_) => return Err(VideoError::DecoderInit("Decode thread died".to_string())),
        };

        Ok(Self {
            cmd_tx,
            resp_rx,
            thread_handle: Some(thread_handle),
            metadata,
            eof_reached: false,
        })
    }
}

impl Drop for MacOSVideoDecoder {
    fn drop(&mut self) {
        // Send shutdown command
        let _ = self.cmd_tx.send(DecodeCommand::Shutdown);
        // Wait for thread to finish
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
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
        if self.eof_reached {
            return Ok(None);
        }

        // Send decode command
        self.cmd_tx
            .send(DecodeCommand::DecodeNext)
            .map_err(|_| VideoError::DecodeFailed("Decode thread died".to_string()))?;

        // Wait for response
        match self.resp_rx.recv() {
            Ok(DecodeResponse::Frame(frame)) => Ok(Some(frame)),
            Ok(DecodeResponse::EndOfStream) => {
                self.eof_reached = true;
                Ok(None)
            }
            Ok(DecodeResponse::Error(e)) => Err(e),
            Ok(_) => Err(VideoError::DecodeFailed("Unexpected response".to_string())),
            Err(_) => Err(VideoError::DecodeFailed("Decode thread died".to_string())),
        }
    }

    fn seek(&mut self, position: Duration) -> Result<(), VideoError> {
        // Send seek command
        self.cmd_tx
            .send(DecodeCommand::Seek(position))
            .map_err(|_| VideoError::SeekFailed("Decode thread died".to_string()))?;

        // Wait for response
        match self.resp_rx.recv() {
            Ok(DecodeResponse::SeekComplete(result)) => {
                if result.is_ok() {
                    self.eof_reached = false;
                }
                result
            }
            Ok(DecodeResponse::Error(e)) => Err(e),
            Ok(_) => Err(VideoError::SeekFailed("Unexpected response".to_string())),
            Err(_) => Err(VideoError::SeekFailed("Decode thread died".to_string())),
        }
    }

    fn metadata(&self) -> &VideoMetadata {
        &self.metadata
    }

    fn hw_accel_type(&self) -> HwAccelType {
        HwAccelType::VideoToolbox
    }
}

// ============================================================================
// Decode thread implementation
// ============================================================================

fn decode_thread_main(
    url: String,
    cmd_rx: Receiver<DecodeCommand>,
    resp_tx: Sender<DecodeResponse>,
) {
    // Initialize decoder
    let decoder = match InnerDecoder::new(&url) {
        Ok(d) => d,
        Err(e) => {
            let _ = resp_tx.send(DecodeResponse::Error(e));
            return;
        }
    };

    // Send metadata
    if resp_tx
        .send(DecodeResponse::Metadata(decoder.metadata.clone()))
        .is_err()
    {
        return;
    }

    // Main decode loop
    let mut decoder = decoder;
    loop {
        match cmd_rx.recv() {
            Ok(DecodeCommand::DecodeNext) => {
                let response = match decoder.decode_next() {
                    Ok(Some(frame)) => DecodeResponse::Frame(frame),
                    Ok(None) => DecodeResponse::EndOfStream,
                    Err(e) => DecodeResponse::Error(e),
                };
                if resp_tx.send(response).is_err() {
                    break;
                }
            }
            Ok(DecodeCommand::Seek(pos)) => {
                let result = decoder.seek(pos);
                if resp_tx.send(DecodeResponse::SeekComplete(result)).is_err() {
                    break;
                }
            }
            Ok(DecodeCommand::Shutdown) | Err(_) => break,
        }
    }
}

// ============================================================================
// Inner decoder - AVFoundation integration
// ============================================================================

/// Inner decoder (lives on decode thread, not Send).
struct InnerDecoder {
    metadata: VideoMetadata,
    asset: Retained<AVURLAsset>,
    reader: Option<Retained<AVAssetReader>>,
    track_output: Option<Retained<AVAssetReaderTrackOutput>>,
    eof_reached: bool,
    current_pts: Duration,
    #[allow(dead_code)]
    url: String,
}

impl InnerDecoder {
    fn new(url: &str) -> Result<Self, VideoError> {
        tracing::info!("MacOSVideoDecoder: Opening {}", url);

        // Create NSURL from the URL string
        let ns_url: Retained<NSURL> = if url.starts_with("http://") || url.starts_with("https://") {
            unsafe {
                let ns_string = NSString::from_str(url);
                NSURL::URLWithString(&ns_string)
                    .ok_or_else(|| VideoError::DecoderInit(format!("Invalid URL: {}", url)))?
            }
        } else {
            // File path
            let path = if url.starts_with("file://") {
                &url[7..]
            } else {
                url
            };
            unsafe {
                let ns_string = NSString::from_str(path);
                NSURL::fileURLWithPath(&ns_string)
            }
        };

        // Create AVURLAsset
        let asset = unsafe { AVURLAsset::URLAssetWithURL_options(&ns_url, None) };

        // Get video tracks - AVMediaTypeVideo is Option<&NSString>
        let media_type = unsafe { AVMediaTypeVideo }
            .ok_or_else(|| VideoError::DecoderInit("AVMediaTypeVideo not available".to_string()))?;
        let video_tracks = unsafe { asset.tracksWithMediaType(media_type) };

        if video_tracks.is_empty() {
            return Err(VideoError::DecoderInit(
                "No video track found in asset".to_string(),
            ));
        }

        // Get the first video track using objectAtIndex
        let video_track = video_tracks.objectAtIndex(0);

        // Extract metadata from the track
        let natural_size = unsafe { video_track.naturalSize() };
        let width = natural_size.width as u32;
        let height = natural_size.height as u32;

        // Get frame rate
        let frame_rate = unsafe { video_track.nominalFrameRate() };
        let frame_rate = if frame_rate <= 0.0 { 30.0 } else { frame_rate };

        // Get duration from asset
        let duration_cm = unsafe { asset.duration() };
        let duration = cmtime_to_duration(duration_cm);

        // Get codec info (simplified - we report VideoToolbox as the decoder)
        let codec = "videotoolbox".to_string();

        let metadata = VideoMetadata {
            width,
            height,
            duration,
            frame_rate,
            codec,
            pixel_aspect_ratio: 1.0,
        };

        tracing::info!(
            "MacOSVideoDecoder: Video {}x{} @ {:.2}fps, duration: {:?}",
            width,
            height,
            frame_rate,
            duration
        );

        let mut decoder = Self {
            metadata,
            asset,
            reader: None,
            track_output: None,
            eof_reached: false,
            current_pts: Duration::ZERO,
            url: url.to_string(),
        };

        // Initialize the reader
        decoder.setup_reader(Duration::ZERO)?;

        Ok(decoder)
    }

    /// Set up AVAssetReader starting from the given position
    fn setup_reader(&mut self, start_time: Duration) -> Result<(), VideoError> {
        // Get video tracks
        let media_type = unsafe { AVMediaTypeVideo }
            .ok_or_else(|| VideoError::DecoderInit("AVMediaTypeVideo not available".to_string()))?;
        let video_tracks = unsafe { self.asset.tracksWithMediaType(media_type) };
        if video_tracks.is_empty() {
            return Err(VideoError::DecoderInit(
                "No video track found in asset".to_string(),
            ));
        }
        let video_track = video_tracks.objectAtIndex(0);

        // Create track output with nil settings - VideoToolbox will use native pixel format
        // which is typically BGRA or NV12 depending on the source
        let track_output = unsafe {
            AVAssetReaderTrackOutput::assetReaderTrackOutputWithTrack_outputSettings(
                &video_track,
                None,
            )
        };

        // Create AVAssetReader
        let reader = unsafe {
            AVAssetReader::assetReaderWithAsset_error(&self.asset).map_err(|e| {
                VideoError::DecoderInit(format!(
                    "Failed to create AVAssetReader: {:?}",
                    e.localizedDescription()
                ))
            })?
        };

        // Set time range if seeking
        if start_time > Duration::ZERO {
            let start_cm = duration_to_cmtime(start_time);
            let duration_cm = unsafe { self.asset.duration() };

            // Calculate remaining duration
            let remaining_secs = cmtime_to_seconds(duration_cm) - start_time.as_secs_f64();
            let remaining_cm = CMTime {
                value: (remaining_secs * duration_cm.timescale as f64) as i64,
                timescale: duration_cm.timescale,
                flags: duration_cm.flags,
                epoch: duration_cm.epoch,
            };

            let time_range = objc2_core_media::CMTimeRange {
                start: start_cm,
                duration: remaining_cm,
            };
            unsafe { reader.setTimeRange(time_range) };
        }

        // Add output to reader
        let can_add = unsafe { reader.canAddOutput(&track_output) };
        if !can_add {
            return Err(VideoError::DecoderInit(
                "Cannot add track output to reader".to_string(),
            ));
        }
        unsafe { reader.addOutput(&track_output) };

        // Start reading
        let started = unsafe { reader.startReading() };
        if !started {
            let error = unsafe { reader.error() };
            let error_msg = error
                .map(|e| e.localizedDescription().to_string())
                .unwrap_or_else(|| "Unknown error".to_string());
            return Err(VideoError::DecoderInit(format!(
                "Failed to start reading: {}",
                error_msg
            )));
        }

        self.reader = Some(reader);
        self.track_output = Some(track_output);
        self.current_pts = start_time;
        self.eof_reached = false;

        Ok(())
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        if self.eof_reached {
            return Ok(None);
        }

        let track_output = self.track_output.as_ref().ok_or_else(|| {
            VideoError::DecodeFailed("Track output not initialized".to_string())
        })?;

        let reader = self.reader.as_ref().ok_or_else(|| {
            VideoError::DecodeFailed("Reader not initialized".to_string())
        })?;

        // Check reader status
        let status = unsafe { reader.status() };
        if status == AVAssetReaderStatus::Failed {
            let error = unsafe { reader.error() };
            let error_msg = error
                .map(|e| e.localizedDescription().to_string())
                .unwrap_or_else(|| "Unknown error".to_string());
            return Err(VideoError::DecodeFailed(format!(
                "Reader failed: {}",
                error_msg
            )));
        }

        if status == AVAssetReaderStatus::Completed {
            self.eof_reached = true;
            return Ok(None);
        }

        if status != AVAssetReaderStatus::Reading {
            return Err(VideoError::DecodeFailed(format!(
                "Unexpected reader status: {:?}",
                status
            )));
        }

        // Get next sample buffer
        let sample_buffer = unsafe { track_output.copyNextSampleBuffer() };

        let sample_buffer = match sample_buffer {
            Some(sb) => sb,
            None => {
                // Check if we're done or if there was an error
                let status = unsafe { reader.status() };
                if status == AVAssetReaderStatus::Completed {
                    self.eof_reached = true;
                    return Ok(None);
                } else if status == AVAssetReaderStatus::Failed {
                    let error = unsafe { reader.error() };
                    let error_msg = error
                        .map(|e| e.localizedDescription().to_string())
                        .unwrap_or_else(|| "Unknown error".to_string());
                    return Err(VideoError::DecodeFailed(error_msg));
                }
                self.eof_reached = true;
                return Ok(None);
            }
        };

        // Get presentation time using the free function
        let pts_cm = unsafe { CMSampleBufferGetPresentationTimeStamp(&sample_buffer) };
        let pts = cmtime_to_duration(pts_cm).unwrap_or(self.current_pts);

        // Get the image buffer (CVPixelBuffer)
        let image_buffer = unsafe { sample_buffer.image_buffer() };
        let pixel_buffer = match image_buffer {
            Some(buf) => buf,
            None => {
                return Err(VideoError::DecodeFailed(
                    "No image buffer in sample".to_string(),
                ));
            }
        };

        // Lock the pixel buffer for CPU access
        let lock_result = unsafe {
            CVPixelBufferLockBaseAddress(&pixel_buffer, CVPixelBufferLockFlags::ReadOnly)
        };
        if lock_result != 0 {
            return Err(VideoError::DecodeFailed(format!(
                "Failed to lock pixel buffer: {}",
                lock_result
            )));
        }

        // Get pixel buffer properties
        let width = unsafe { CVPixelBufferGetWidth(&pixel_buffer) };
        let height = unsafe { CVPixelBufferGetHeight(&pixel_buffer) };
        let bytes_per_row = unsafe { CVPixelBufferGetBytesPerRow(&pixel_buffer) };
        let base_address = unsafe { CVPixelBufferGetBaseAddress(&pixel_buffer) };

        if base_address.is_null() {
            unsafe {
                CVPixelBufferUnlockBaseAddress(&pixel_buffer, CVPixelBufferLockFlags::ReadOnly);
            }
            return Err(VideoError::DecodeFailed(
                "Null base address in pixel buffer".to_string(),
            ));
        }

        // Copy pixel data (BGRA format)
        let data_size = bytes_per_row * height;
        let bgra_data =
            unsafe { std::slice::from_raw_parts(base_address as *const u8, data_size) };

        // Convert BGRA to RGBA
        let mut rgba_data = Vec::with_capacity(width * height * 4);
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

        // Unlock the pixel buffer
        unsafe {
            CVPixelBufferUnlockBaseAddress(&pixel_buffer, CVPixelBufferLockFlags::ReadOnly);
        }

        self.current_pts = pts;

        // Create CPU frame
        let cpu_frame = CpuFrame::new(
            PixelFormat::Rgba,
            width as u32,
            height as u32,
            vec![Plane {
                data: rgba_data,
                stride: width * 4,
            }],
        );

        Ok(Some(VideoFrame::new(pts, DecodedFrame::Cpu(cpu_frame))))
    }

    fn seek(&mut self, position: Duration) -> Result<(), VideoError> {
        // AVAssetReader doesn't support seeking directly.
        // We need to recreate the reader with a new time range.
        self.reader = None;
        self.track_output = None;
        self.setup_reader(position)?;
        Ok(())
    }
}

// ============================================================================
// CMTime conversion helpers
// ============================================================================

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
    let timescale: i32 = 600; // Common video timescale
    let value = (duration.as_secs_f64() * timescale as f64) as i64;
    CMTime {
        value,
        timescale,
        flags: CMTimeFlags::Valid,
        epoch: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmtime_conversion() {
        let dur = Duration::from_secs(5);
        let cm = duration_to_cmtime(dur);
        let back = cmtime_to_duration(cm);
        assert!(back.is_some());
        let diff = (back.unwrap().as_secs_f64() - dur.as_secs_f64()).abs();
        assert!(diff < 0.001);
    }
}
