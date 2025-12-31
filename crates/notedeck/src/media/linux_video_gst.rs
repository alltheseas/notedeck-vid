//! GStreamer-based video decoder for Linux.
//!
//! This module provides hardware-accelerated video decoding using GStreamer,
//! which handles codec edge cases (frame_num gaps, broken streams) much better
//! than the cros-codecs approach.
//!
//! GStreamer automatically selects the best decoder (VA-API, software fallback)
//! and handles all the complexity of H.264/VP8/VP9/AV1 decoding.

use std::time::Duration;

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;

use crate::media::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata,
};

/// GStreamer-based video decoder for Linux.
///
/// Uses a GStreamer pipeline:
/// `urisourcebin ! decodebin ! videoconvert ! video/x-raw,format=NV12 ! appsink`
///
/// This handles:
/// - HTTP/HTTPS streaming
/// - All common codecs (H.264, VP8, VP9, AV1)
/// - Hardware acceleration via VA-API (automatic)
/// - Edge cases that break other decoders
pub struct GStreamerDecoder {
    pipeline: gst::Pipeline,
    appsink: gst_app::AppSink,
    metadata: VideoMetadata,
    position: Duration,
    eof: bool,
    /// True if we just seeked and are waiting for first frame
    seeking: bool,
}

impl GStreamerDecoder {
    /// Creates a new GStreamer decoder for the given URL.
    pub fn new(url: &str) -> Result<Self, VideoError> {
        // Initialize GStreamer (safe to call multiple times)
        gst::init().map_err(|e| VideoError::DecoderInit(format!("GStreamer init failed: {}", e)))?;

        // Build the pipeline
        let pipeline = gst::Pipeline::new();

        // Source element - handles HTTP, HTTPS, file://
        let source = gst::ElementFactory::make("uridecodebin")
            .property("uri", url)
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create uridecodebin: {}", e)))?;

        // Video convert to ensure we get a format we can handle
        let convert = gst::ElementFactory::make("videoconvert")
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create videoconvert: {}", e)))?;

        // App sink to pull frames
        let appsink = gst_app::AppSink::builder()
            .caps(
                &gst_video::VideoCapsBuilder::new()
                    .format(gst_video::VideoFormat::Nv12)
                    .build(),
            )
            .build();

        // Add elements to pipeline
        pipeline
            .add_many([&source, &convert, appsink.upcast_ref()])
            .map_err(|e| VideoError::DecoderInit(format!("Failed to add elements: {}", e)))?;

        // Link convert -> appsink (source links dynamically via pad-added)
        convert
            .link(&appsink)
            .map_err(|e| VideoError::DecoderInit(format!("Failed to link elements: {}", e)))?;

        // Handle dynamic pad creation from uridecodebin
        let convert_weak = convert.downgrade();
        source.connect_pad_added(move |_src, src_pad| {
            let Some(convert) = convert_weak.upgrade() else {
                return;
            };

            // Only link video pads
            let caps = src_pad.current_caps().unwrap_or_else(|| src_pad.query_caps(None));
            let structure = caps.structure(0).unwrap();
            let name = structure.name();

            if name.starts_with("video/") {
                let sink_pad = convert.static_pad("sink").unwrap();
                if !sink_pad.is_linked() {
                    if let Err(e) = src_pad.link(&sink_pad) {
                        tracing::warn!("Failed to link video pad: {:?}", e);
                    } else {
                        tracing::info!("Linked video pad: {}", name);
                    }
                }
            }
        });

        // Start the pipeline
        pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| VideoError::DecoderInit(format!("Failed to start pipeline: {:?}", e)))?;

        // Wait for pipeline to reach playing state or error
        let bus = pipeline.bus().unwrap();
        let mut width = 0u32;
        let mut height = 0u32;
        let mut duration = None;

        // Wait for async state change and get metadata
        for msg in bus.iter_timed(gst::ClockTime::from_seconds(10)) {
            match msg.view() {
                gst::MessageView::AsyncDone(_) => {
                    // Query duration
                    if let Some(dur) = pipeline.query_duration::<gst::ClockTime>() {
                        duration = Some(Duration::from_nanos(dur.nseconds()));
                    }
                    break;
                }
                gst::MessageView::Error(err) => {
                    return Err(VideoError::DecoderInit(format!(
                        "Pipeline error: {} ({:?})",
                        err.error(),
                        err.debug()
                    )));
                }
                gst::MessageView::StateChanged(state) => {
                    if state.src().map(|s| s == pipeline.upcast_ref::<gst::Object>()).unwrap_or(false) {
                        tracing::debug!(
                            "Pipeline state: {:?} -> {:?}",
                            state.old(),
                            state.current()
                        );
                    }
                }
                _ => {}
            }
        }

        // Get video dimensions from appsink caps
        if let Some(caps) = appsink.sink_pads().first().and_then(|p| p.current_caps()) {
            if let Some(s) = caps.structure(0) {
                width = s.get::<i32>("width").unwrap_or(0) as u32;
                height = s.get::<i32>("height").unwrap_or(0) as u32;
            }
        }

        // If we couldn't get dimensions yet, try pulling a frame
        if width == 0 || height == 0 {
            // Pull a sample to get dimensions
            if let Some(sample) = appsink.try_pull_preroll(gst::ClockTime::from_seconds(5)) {
                if let Some(caps) = sample.caps() {
                    if let Some(s) = caps.structure(0) {
                        width = s.get::<i32>("width").unwrap_or(0) as u32;
                        height = s.get::<i32>("height").unwrap_or(0) as u32;
                    }
                }
            }
        }

        if width == 0 || height == 0 {
            return Err(VideoError::DecoderInit(
                "Could not determine video dimensions".to_string(),
            ));
        }

        tracing::info!(
            "GStreamer decoder initialized: {}x{}, duration: {:?}",
            width,
            height,
            duration
        );

        let metadata = VideoMetadata {
            width,
            height,
            duration,
            frame_rate: 30.0, // GStreamer doesn't expose this easily; default to 30fps
            codec: "unknown".to_string(), // GStreamer handles codec internally
            pixel_aspect_ratio: 1.0,
        };

        Ok(Self {
            pipeline,
            appsink,
            metadata,
            position: Duration::ZERO,
            eof: false,
            seeking: false,
        })
    }

    /// Converts a GStreamer sample to our VideoFrame format.
    fn sample_to_frame(&self, sample: gst::Sample) -> Result<VideoFrame, VideoError> {
        let buffer = sample
            .buffer()
            .ok_or_else(|| VideoError::DecodeFailed("Sample has no buffer".to_string()))?;

        let caps = sample
            .caps()
            .ok_or_else(|| VideoError::DecodeFailed("Sample has no caps".to_string()))?;

        let video_info = gst_video::VideoInfo::from_caps(caps)
            .map_err(|e| VideoError::DecodeFailed(format!("Invalid video caps: {}", e)))?;

        let pts = buffer
            .pts()
            .map(|t| Duration::from_nanos(t.nseconds()))
            .unwrap_or(self.position);

        // Map the buffer for reading
        let map = buffer
            .map_readable()
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to map buffer: {}", e)))?;

        let width = video_info.width();
        let height = video_info.height();

        // For NV12: Y plane followed by interleaved UV plane
        let y_stride = video_info.stride()[0] as usize;
        let uv_stride = video_info.stride()[1] as usize;
        let y_offset = video_info.offset()[0] as usize;
        let uv_offset = video_info.offset()[1] as usize;

        let y_size = y_stride * height as usize;
        let uv_size = uv_stride * (height as usize / 2);

        let data = map.as_slice();

        // Extract Y plane
        let y_data = if y_offset + y_size <= data.len() {
            data[y_offset..y_offset + y_size].to_vec()
        } else {
            return Err(VideoError::DecodeFailed("Y plane out of bounds".to_string()));
        };

        // Extract UV plane
        let uv_data = if uv_offset + uv_size <= data.len() {
            data[uv_offset..uv_offset + uv_size].to_vec()
        } else {
            return Err(VideoError::DecodeFailed("UV plane out of bounds".to_string()));
        };

        let y_plane = Plane {
            data: y_data,
            stride: y_stride,
        };

        let uv_plane = Plane {
            data: uv_data,
            stride: uv_stride,
        };

        let cpu_frame = CpuFrame::new(PixelFormat::Nv12, width, height, vec![y_plane, uv_plane]);

        Ok(VideoFrame::new(pts, DecodedFrame::Cpu(cpu_frame)))
    }
}

impl Drop for GStreamerDecoder {
    fn drop(&mut self) {
        // Must set to Null state and wait for completion to properly clean up
        if let Err(e) = self.pipeline.set_state(gst::State::Null) {
            tracing::warn!("Failed to set pipeline to Null state: {:?}", e);
        }
        // Wait for state change to complete
        let _ = self.pipeline.state(gst::ClockTime::from_seconds(1));
    }
}

// GStreamer is thread-safe
unsafe impl Send for GStreamerDecoder {}

impl VideoDecoderBackend for GStreamerDecoder {
    fn open(url: &str) -> Result<Self, VideoError>
    where
        Self: Sized,
    {
        Self::new(url)
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        if self.eof {
            return Ok(None);
        }

        // Check for pipeline errors
        if let Some(bus) = self.pipeline.bus() {
            while let Some(msg) = bus.pop() {
                match msg.view() {
                    gst::MessageView::Error(err) => {
                        return Err(VideoError::DecodeFailed(format!(
                            "Pipeline error: {}",
                            err.error()
                        )));
                    }
                    gst::MessageView::Eos(_) => {
                        self.eof = true;
                        return Ok(None);
                    }
                    _ => {}
                }
            }
        }

        // Use longer timeout after seek to allow for rebuffering
        let timeout_ms = if self.seeking { 1000 } else { 100 };

        match self.appsink.try_pull_sample(gst::ClockTime::from_mseconds(timeout_ms)) {
            Some(sample) => {
                let frame = self.sample_to_frame(sample)?;
                self.position = frame.pts;
                self.seeking = false; // Got a frame, no longer seeking
                Ok(Some(frame))
            }
            None => {
                // Check if truly at EOS
                if self.appsink.is_eos() {
                    self.eof = true;
                    self.seeking = false;
                }
                Ok(None)
            }
        }
    }

    fn seek(&mut self, position: Duration) -> Result<(), VideoError> {
        let position_ns = position.as_nanos() as u64;

        // Mark that we're seeking - decode_next will use longer timeout
        self.seeking = true;

        // Use FLUSH to clear buffers, KEY_UNIT for fast seeking to nearest keyframe
        self.pipeline
            .seek_simple(
                gst::SeekFlags::FLUSH | gst::SeekFlags::KEY_UNIT,
                gst::ClockTime::from_nseconds(position_ns),
            )
            .map_err(|e| VideoError::SeekFailed(format!("Seek failed: {:?}", e)))?;

        // Wait for seek to complete (ASYNC_DONE message)
        if let Some(bus) = self.pipeline.bus() {
            for msg in bus.iter_timed(gst::ClockTime::from_seconds(5)) {
                match msg.view() {
                    gst::MessageView::AsyncDone(_) => {
                        tracing::debug!("Seek completed to {:?}", position);
                        break;
                    }
                    gst::MessageView::Error(err) => {
                        self.seeking = false;
                        return Err(VideoError::SeekFailed(format!(
                            "Seek error: {} ({:?})",
                            err.error(),
                            err.debug()
                        )));
                    }
                    _ => {}
                }
            }
        }

        self.position = position;
        self.eof = false;

        Ok(())
    }

    fn metadata(&self) -> &VideoMetadata {
        &self.metadata
    }

    fn pause(&mut self) -> Result<(), VideoError> {
        self.pipeline
            .set_state(gst::State::Paused)
            .map_err(|e| VideoError::Generic(format!("Pause failed: {:?}", e)))?;
        Ok(())
    }

    fn resume(&mut self) -> Result<(), VideoError> {
        self.pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| VideoError::Generic(format!("Resume failed: {:?}", e)))?;
        Ok(())
    }

    fn hw_accel_type(&self) -> HwAccelType {
        // GStreamer handles HW accel internally (auto-selects VA-API, etc.)
        HwAccelType::Vaapi
    }
}
