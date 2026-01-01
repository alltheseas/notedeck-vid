//! GStreamer-based video decoder for Linux.
//!
//! This module provides hardware-accelerated video decoding using GStreamer,
//! which handles codec edge cases (frame_num gaps, broken streams) robustly.
//!
//! GStreamer automatically selects the best decoder (VA-API, software fallback)
//! and handles all the complexity of H.264/VP8/VP9/AV1 decoding.
//!
//! Audio is played directly by GStreamer via autoaudiosink, with volume control
//! exposed through the GStreamer volume element.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;

use crate::media::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata,
};

/// Shared audio state for GStreamer audio control.
/// This is used to control volume/mute from the UI thread.
#[derive(Clone)]
pub struct GstAudioHandle {
    inner: Arc<GstAudioHandleInner>,
}

struct GstAudioHandleInner {
    /// Volume element for control (None if no audio)
    volume_element: Option<gst::Element>,
    /// Whether audio is available
    has_audio: AtomicBool,
    /// Whether audio is muted
    muted: AtomicBool,
    /// Volume level (0.0 - 1.0)
    volume: std::sync::atomic::AtomicU32, // stored as volume * 100
}

impl GstAudioHandle {
    fn new(volume_element: Option<gst::Element>) -> Self {
        let has_audio = volume_element.is_some();
        Self {
            inner: Arc::new(GstAudioHandleInner {
                volume_element,
                has_audio: AtomicBool::new(has_audio),
                muted: AtomicBool::new(false),
                volume: std::sync::atomic::AtomicU32::new(100), // 100%
            }),
        }
    }

    /// Returns whether audio is available.
    pub fn has_audio(&self) -> bool {
        self.inner.has_audio.load(Ordering::Relaxed)
    }

    /// Returns whether audio is muted.
    pub fn is_muted(&self) -> bool {
        self.inner.muted.load(Ordering::Relaxed)
    }

    /// Sets the mute state.
    pub fn set_muted(&self, muted: bool) {
        self.inner.muted.store(muted, Ordering::Relaxed);
        self.apply_volume();
    }

    /// Toggles mute state.
    pub fn toggle_mute(&self) {
        let current = self.inner.muted.load(Ordering::Relaxed);
        self.inner.muted.store(!current, Ordering::Relaxed);
        self.apply_volume();
    }

    /// Returns the current volume (0-100).
    pub fn volume(&self) -> u32 {
        self.inner.volume.load(Ordering::Relaxed)
    }

    /// Sets the volume (0-100).
    pub fn set_volume(&self, volume: u32) {
        self.inner.volume.store(volume.min(100), Ordering::Relaxed);
        self.apply_volume();
    }

    /// Applies the current volume/mute state to the GStreamer element.
    fn apply_volume(&self) {
        if let Some(ref vol_elem) = self.inner.volume_element {
            let effective_volume = if self.inner.muted.load(Ordering::Relaxed) {
                0.0
            } else {
                self.inner.volume.load(Ordering::Relaxed) as f64 / 100.0
            };
            vol_elem.set_property("volume", effective_volume);
        }
    }
}

/// Buffering thresholds for hysteresis to prevent rapid pause/resume oscillation.
/// - Low threshold: pause only when buffer drops critically low
/// - High threshold: resume only when buffer is sufficiently full
/// The gap between thresholds prevents rapid state changes on marginal connections.
const BUFFER_LOW_THRESHOLD: i32 = 10; // Pause when buffer drops below this %
const BUFFER_HIGH_THRESHOLD: i32 = 100; // Resume when buffer reaches this %

/// GStreamer-based video decoder for Linux.
///
/// Uses a GStreamer pipeline:
/// - Video: `uridecodebin ! videoconvert ! video/x-raw,format=NV12 ! appsink`
/// - Audio: `uridecodebin ! audioconvert ! audioresample ! volume ! autoaudiosink`
///
/// This handles:
/// - HTTP/HTTPS streaming
/// - All common codecs (H.264, VP8, VP9, AV1)
/// - Hardware acceleration via VA-API (automatic)
/// - Edge cases that break other decoders
/// - Audio playback with volume control
pub struct GStreamerDecoder {
    pipeline: gst::Pipeline,
    appsink: gst_app::AppSink,
    metadata: VideoMetadata,
    position: Duration,
    eof: bool,
    /// True if we just seeked and are waiting for first frame
    seeking: bool,
    /// Target position of the last seek (for stale frame detection)
    seek_target: Option<Duration>,
    /// Cached preroll sample for first decode_next() call
    preroll_sample: Option<gst::Sample>,
    /// Buffering percentage (0-100), 100 means fully buffered
    buffering_percent: i32,
    /// True once we've reached 100% buffering at least once (for rebuffer detection)
    was_fully_buffered: bool,
    /// Audio control handle
    audio_handle: GstAudioHandle,
}

impl GStreamerDecoder {
    /// Creates a new GStreamer decoder for the given URL.
    pub fn new(url: &str) -> Result<Self, VideoError> {
        // Initialize GStreamer (safe to call multiple times)
        gst::init().map_err(|e| VideoError::DecoderInit(format!("GStreamer init failed: {e}")))?;

        // Build the pipeline
        let pipeline = gst::Pipeline::new();

        // Source element - handles HTTP, HTTPS, file://
        let source = gst::ElementFactory::make("uridecodebin")
            .property("uri", url)
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create uridecodebin: {e}")))?;

        // === Video elements ===
        let videoconvert = gst::ElementFactory::make("videoconvert")
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create videoconvert: {e}")))?;

        // App sink to pull video frames - constrained buffering for better seek behavior
        let appsink = gst_app::AppSink::builder()
            .caps(
                &gst_video::VideoCapsBuilder::new()
                    .format(gst_video::VideoFormat::Nv12)
                    .build(),
            )
            .max_buffers(1)
            .drop(true)
            .build();

        // === Audio elements ===
        let audioconvert = gst::ElementFactory::make("audioconvert")
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create audioconvert: {e}")))?;

        let audioresample = gst::ElementFactory::make("audioresample")
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create audioresample: {e}")))?;

        let volume = gst::ElementFactory::make("volume")
            .property("volume", 1.0f64)
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create volume: {e}")))?;

        let audiosink = gst::ElementFactory::make("autoaudiosink")
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create autoaudiosink: {e}")))?;

        // Add all elements to pipeline
        pipeline
            .add_many([
                &source,
                &videoconvert,
                appsink.upcast_ref(),
                &audioconvert,
                &audioresample,
                &volume,
                &audiosink,
            ])
            .map_err(|e| VideoError::DecoderInit(format!("Failed to add elements: {e}")))?;

        // Link video chain: videoconvert -> appsink
        videoconvert
            .link(&appsink)
            .map_err(|e| VideoError::DecoderInit(format!("Failed to link video elements: {e}")))?;

        // Link audio chain: audioconvert -> audioresample -> volume -> audiosink
        gst::Element::link_many([&audioconvert, &audioresample, &volume, &audiosink])
            .map_err(|e| VideoError::DecoderInit(format!("Failed to link audio elements: {e}")))?;

        // Handle dynamic pad creation from uridecodebin
        let videoconvert_weak = videoconvert.downgrade();
        let audioconvert_weak = audioconvert.downgrade();
        source.connect_pad_added(move |_src, src_pad| {
            let caps = src_pad
                .current_caps()
                .unwrap_or_else(|| src_pad.query_caps(None));
            let Some(structure) = caps.structure(0) else {
                return;
            };
            let name = structure.name();

            if name.starts_with("video/") {
                if let Some(videoconvert) = videoconvert_weak.upgrade() {
                    let sink_pad = videoconvert.static_pad("sink").unwrap();
                    if !sink_pad.is_linked() {
                        if let Err(e) = src_pad.link(&sink_pad) {
                            tracing::warn!("Failed to link video pad: {:?}", e);
                        } else {
                            tracing::info!("Linked video pad: {}", name);
                        }
                    }
                }
            } else if name.starts_with("audio/") {
                if let Some(audioconvert) = audioconvert_weak.upgrade() {
                    let sink_pad = audioconvert.static_pad("sink").unwrap();
                    if !sink_pad.is_linked() {
                        if let Err(e) = src_pad.link(&sink_pad) {
                            tracing::warn!("Failed to link audio pad: {:?}", e);
                        } else {
                            tracing::info!("Linked audio pad: {}", name);
                        }
                    }
                }
            }
        });

        // Create audio handle with volume element
        let audio_handle = GstAudioHandle::new(Some(volume));

        // Set pipeline to Paused to get metadata without starting playback
        // (Playing state would autoplay the video)
        pipeline
            .set_state(gst::State::Paused)
            .map_err(|e| VideoError::DecoderInit(format!("Failed to start pipeline: {e:?}")))?;

        // Wait for pipeline to reach paused state (preroll) or error
        let bus = pipeline.bus().unwrap();
        let mut width = 0u32;
        let mut height = 0u32;
        let mut duration = None;

        // Track buffering during init (in case 100% is reached before decode loop starts)
        let mut init_buffering_percent = 0i32;

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
                    // Clean up pipeline before returning error
                    let _ = pipeline.set_state(gst::State::Null);
                    let _ = pipeline.state(gst::ClockTime::from_seconds(2));
                    return Err(VideoError::DecoderInit(format!(
                        "Pipeline error: {} ({:?})",
                        err.error(),
                        err.debug()
                    )));
                }
                gst::MessageView::StateChanged(state) => {
                    if state
                        .src()
                        .map(|s| s == pipeline.upcast_ref::<gst::Object>())
                        .unwrap_or(false)
                    {
                        tracing::debug!(
                            "Pipeline state: {:?} -> {:?}",
                            state.old(),
                            state.current()
                        );
                    }
                }
                gst::MessageView::Buffering(buffering) => {
                    // Track buffering during init - important for fast streams
                    // that reach 100% before decode loop starts
                    init_buffering_percent = buffering.percent();
                    tracing::debug!("Init buffering: {}%", init_buffering_percent);
                }
                _ => {}
            }
        }

        // Get video dimensions and frame rate from appsink caps
        let mut frame_rate = 30.0f32; // Default fallback
        if let Some(caps) = appsink.sink_pads().first().and_then(|p| p.current_caps()) {
            if let Some(s) = caps.structure(0) {
                width = s.get::<i32>("width").unwrap_or(0) as u32;
                height = s.get::<i32>("height").unwrap_or(0) as u32;
                // Extract frame rate from caps (stored as fraction)
                if let Ok(fps) = s.get::<gst::Fraction>("framerate") {
                    if fps.denom() != 0 {
                        frame_rate = fps.numer() as f32 / fps.denom() as f32;
                        tracing::debug!("Detected frame rate: {:.2} fps", frame_rate);
                    }
                }
            }
        }

        // Try to pull preroll sample - this gives us dimensions AND the first frame
        // We cache this sample to return on the first decode_next() call
        // Use generous timeout for slow network streams
        let preroll_sample = appsink.try_pull_preroll(gst::ClockTime::from_seconds(10));

        // If we couldn't get dimensions/framerate from caps, try from preroll sample
        if width == 0 || height == 0 || frame_rate == 30.0 {
            if let Some(ref sample) = preroll_sample {
                if let Some(caps) = sample.caps() {
                    if let Some(s) = caps.structure(0) {
                        if width == 0 {
                            width = s.get::<i32>("width").unwrap_or(0) as u32;
                        }
                        if height == 0 {
                            height = s.get::<i32>("height").unwrap_or(0) as u32;
                        }
                        // Try to get frame rate from preroll sample caps
                        if frame_rate == 30.0 {
                            if let Ok(fps) = s.get::<gst::Fraction>("framerate") {
                                if fps.denom() != 0 {
                                    frame_rate = fps.numer() as f32 / fps.denom() as f32;
                                    tracing::debug!(
                                        "Detected frame rate from preroll: {:.2} fps",
                                        frame_rate
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        if width == 0 || height == 0 {
            // Clean up pipeline before returning error
            let _ = pipeline.set_state(gst::State::Null);
            let _ = pipeline.state(gst::ClockTime::from_seconds(2));
            return Err(VideoError::DecoderInit(
                "Could not determine video dimensions".to_string(),
            ));
        }

        tracing::info!(
            "GStreamer decoder initialized: {}x{}, duration: {:?}, audio: {}",
            width,
            height,
            duration,
            audio_handle.has_audio()
        );

        let metadata = VideoMetadata {
            width,
            height,
            duration,
            frame_rate, // Extracted from caps, defaults to 30fps if not found
            codec: "unknown".to_string(), // GStreamer handles codec internally
            pixel_aspect_ratio: 1.0,
        };

        // For network streams, use buffering tracked during init (may have reached 100% already)
        // For local files, assume 100%
        let initial_buffering = if url.starts_with("http://") || url.starts_with("https://") {
            // Use the buffering percentage observed during init
            // This handles fast streams that buffer completely during preroll
            init_buffering_percent
        } else {
            100 // Local files are immediately available
        };

        Ok(Self {
            pipeline,
            appsink,
            metadata,
            position: Duration::ZERO,
            eof: false,
            seeking: false,
            seek_target: None,
            preroll_sample,
            buffering_percent: initial_buffering,
            was_fully_buffered: initial_buffering >= 100,
            audio_handle,
        })
    }

    /// Returns the audio handle for volume/mute control.
    pub fn audio_handle(&self) -> &GstAudioHandle {
        &self.audio_handle
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
            .map_err(|e| VideoError::DecodeFailed(format!("Invalid video caps: {e}")))?;

        let pts = buffer
            .pts()
            .map(|t| Duration::from_nanos(t.nseconds()))
            .unwrap_or(self.position);

        // Map the buffer for reading
        let map = buffer
            .map_readable()
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to map buffer: {e}")))?;

        let width = video_info.width();
        let height = video_info.height();

        // For NV12: Y plane followed by interleaved UV plane
        let y_stride = video_info.stride()[0] as usize;
        let uv_stride = video_info.stride()[1] as usize;
        let y_offset = video_info.offset()[0];
        let uv_offset = video_info.offset()[1];

        let y_size = y_stride * height as usize;
        let uv_size = uv_stride * (height as usize / 2);

        let data = map.as_slice();

        // Extract Y plane
        let y_data = if y_offset + y_size <= data.len() {
            data[y_offset..y_offset + y_size].to_vec()
        } else {
            return Err(VideoError::DecodeFailed(
                "Y plane out of bounds".to_string(),
            ));
        };

        // Extract UV plane
        let uv_data = if uv_offset + uv_size <= data.len() {
            data[uv_offset..uv_offset + uv_size].to_vec()
        } else {
            return Err(VideoError::DecodeFailed(
                "UV plane out of bounds".to_string(),
            ));
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

    /// Internal seek implementation (may be retried on transient errors).
    fn seek_internal(&mut self, position: Duration) -> Result<(), VideoError> {
        let position_ns = position.as_nanos() as u64;

        // Mark that we're seeking - decode_next will skip bus polling
        self.seeking = true;
        self.seek_target = Some(position);

        // Choose seek flags based on direction:
        // - Forward: KEY_UNIT for fast keyframe-based seeking
        // - Backward: ACCURATE for reliable frame-accurate seeking
        //   (KEY_UNIT + SNAP_BEFORE caused video freeze, see notedeck-vid-w4r)
        let flags = if position < self.position {
            gst::SeekFlags::FLUSH | gst::SeekFlags::ACCURATE
        } else {
            gst::SeekFlags::FLUSH | gst::SeekFlags::KEY_UNIT
        };

        if let Err(e) = self
            .pipeline
            .seek_simple(flags, gst::ClockTime::from_nseconds(position_ns))
        {
            // Clear seeking state on error to avoid getting stuck
            self.seeking = false;
            self.seek_target = None;
            return Err(VideoError::SeekFailed(format!("Seek failed: {e:?}")));
        }

        // Wait for seek completion using filtered pop - only consume ASYNC_DONE or ERROR
        // This prevents swallowing other messages that decode_next needs
        // Use generous timeout for slow network streams that need to rebuffer
        if let Some(bus) = self.pipeline.bus() {
            let msg = bus.timed_pop_filtered(
                gst::ClockTime::from_seconds(10),
                &[gst::MessageType::AsyncDone, gst::MessageType::Error],
            );
            match msg {
                Some(msg) => match msg.view() {
                    gst::MessageView::AsyncDone(_) => {
                        let direction = if position < self.position {
                            "backward"
                        } else {
                            "forward"
                        };
                        tracing::debug!(
                            "Seek {} completed: {:?} -> {:?}",
                            direction,
                            self.position,
                            position
                        );
                    }
                    gst::MessageView::Error(err) => {
                        self.seeking = false;
                        self.seek_target = None;
                        return Err(VideoError::SeekFailed(format!(
                            "Seek error: {} ({:?})",
                            err.error(),
                            err.debug()
                        )));
                    }
                    _ => {}
                },
                None => {
                    // Timeout waiting for seek completion
                    self.seeking = false;
                    self.seek_target = None;
                    return Err(VideoError::SeekFailed("Seek timed out".into()));
                }
            }
        }

        self.position = position;
        self.eof = false;
        // Assume rebuffering will be needed after seek (HTTP streams)
        self.buffering_percent = 0;
        // Reset so we don't pause during post-seek buffering
        self.was_fully_buffered = false;

        Ok(())
    }
}

impl Drop for GStreamerDecoder {
    fn drop(&mut self) {
        // Fire and forget - don't block the UI thread at all
        // GStreamer handles cleanup asynchronously
        let _ = self.pipeline.set_state(gst::State::Null);
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

        // Return cached preroll sample on first call (consumed during init for dimensions)
        if let Some(sample) = self.preroll_sample.take() {
            let frame = self.sample_to_frame(sample)?;
            tracing::debug!("Returning cached preroll frame at {:?}", frame.pts);
            self.position = frame.pts;
            return Ok(Some(frame));
        }

        // Poll bus for messages - always check for buffering, but skip Error/EOS while seeking
        // (seek() handles AsyncDone/Error, but we still need buffering updates for UI)
        if let Some(bus) = self.pipeline.bus() {
            while let Some(msg) = bus.pop() {
                match msg.view() {
                    gst::MessageView::Error(err) => {
                        // Only handle errors when not seeking - seek() handles its own errors
                        if !self.seeking {
                            return Err(VideoError::DecodeFailed(format!(
                                "Pipeline error: {}",
                                err.error()
                            )));
                        }
                    }
                    gst::MessageView::Eos(_) => {
                        // Only handle EOS when not seeking
                        if !self.seeking {
                            self.eof = true;
                            return Ok(None);
                        }
                    }
                    gst::MessageView::Buffering(buffering) => {
                        // Always handle buffering messages for UI feedback
                        let percent = buffering.percent();
                        if percent != self.buffering_percent {
                            tracing::debug!("Buffering: {}%", percent);
                            self.buffering_percent = percent;

                            // Hysteresis buffering: use different thresholds for pause vs resume
                            // to prevent rapid oscillation on marginal connections.
                            //
                            // - Resume only when buffer is full (HIGH_THRESHOLD = 100%)
                            // - Pause only when buffer is critically low (LOW_THRESHOLD = 10%)
                            // - The 90% gap prevents rapid pause/play cycling
                            //
                            // Only do this for rebuffering (after we've been at 100% once),
                            // not during initial buffering which happens before playback starts.
                            if percent >= BUFFER_HIGH_THRESHOLD {
                                self.was_fully_buffered = true;
                                // Resume playback - buffer is healthy
                                let _ = self.pipeline.set_state(gst::State::Playing);
                            } else if self.was_fully_buffered && percent < BUFFER_LOW_THRESHOLD {
                                // Buffer critically low during playback - pause to let it refill
                                tracing::info!(
                                    "Buffer critically low ({}%), pausing to refill",
                                    percent
                                );
                                let _ = self.pipeline.set_state(gst::State::Paused);
                            }
                            // Between thresholds: maintain current state (hysteresis)
                            // During initial buffering (was_fully_buffered = false),
                            // don't change state - let it fill naturally
                        }
                    }
                    _ => {}
                }
            }
        }

        // Use longer timeout when buffering or after seek
        let timeout_ms = if self.seeking || self.buffering_percent < 100 {
            1000
        } else {
            100
        };

        // When seeking, we may need to discard stale frames
        let max_stale_frames = if self.seeking { 5 } else { 0 };
        let mut discarded = 0;

        loop {
            match self
                .appsink
                .try_pull_sample(gst::ClockTime::from_mseconds(timeout_ms))
            {
                Some(sample) => {
                    let frame = self.sample_to_frame(sample)?;

                    // Check for stale frames after seek
                    if self.seeking {
                        if let Some(target) = self.seek_target {
                            // Determine if this is a forward or backward seek
                            let is_backward_seek = target < self.position;

                            // For backward seeks: discard frames far AFTER the target
                            // (these are old buffered frames from the later position)
                            let too_far_after = frame.pts > target + Duration::from_secs(2);

                            // For forward seeks: discard frames BEFORE the target
                            // (these are old buffered frames from the earlier position)
                            // Use small tolerance to avoid discarding the target frame
                            let too_far_before = !is_backward_seek
                                && frame.pts + Duration::from_millis(100) < target;

                            if (too_far_after || too_far_before) && discarded < max_stale_frames {
                                discarded += 1;
                                tracing::debug!(
                                    "Discarding stale frame at {:?} (seek target {:?}, {})",
                                    frame.pts,
                                    target,
                                    if too_far_before { "before" } else { "after" }
                                );
                                continue; // Try to get another frame
                            }
                        }

                        tracing::debug!(
                            "First frame after seek at {:?} (expected ~{:?})",
                            frame.pts,
                            self.position
                        );
                    }

                    self.position = frame.pts;
                    self.seeking = false;
                    self.seek_target = None;
                    return Ok(Some(frame));
                }
                None => {
                    // Debug: log appsink state when we get None
                    if self.seeking {
                        tracing::debug!(
                            "No frame after seek: eos={}, position={:?}",
                            self.appsink.is_eos(),
                            self.position
                        );
                    }
                    // Check if truly at EOS - only clear seeking if at EOS
                    if self.appsink.is_eos() {
                        self.eof = true;
                        self.seeking = false;
                        self.seek_target = None;
                    }
                    // Keep seeking flag true until we get a frame - this ensures
                    // we use longer timeouts while waiting for post-seek frames
                    return Ok(None);
                }
            }
        }
    }

    fn seek(&mut self, position: Duration) -> Result<(), VideoError> {
        // Retry seek up to 3 times for transient HTTP errors
        const MAX_RETRIES: u32 = 3;
        let mut last_error = None;

        for attempt in 0..=MAX_RETRIES {
            match self.seek_internal(position) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    if attempt < MAX_RETRIES {
                        tracing::warn!("Seek attempt {} failed, retrying: {}", attempt + 1, e);
                        // Reset pipeline state before retry - helps recover from HTTP errors
                        let _ = self.pipeline.set_state(gst::State::Paused);
                        let _ = self.pipeline.state(gst::ClockTime::from_mseconds(500));
                        let _ = self.pipeline.set_state(gst::State::Playing);
                        let _ = self.pipeline.state(gst::ClockTime::from_mseconds(500));
                        // Longer delay for HTTP reconnection
                        std::thread::sleep(std::time::Duration::from_millis(500));
                    }
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap())
    }

    fn metadata(&self) -> &VideoMetadata {
        &self.metadata
    }

    fn pause(&mut self) -> Result<(), VideoError> {
        self.pipeline
            .set_state(gst::State::Paused)
            .map_err(|e| VideoError::Generic(format!("Pause failed: {e:?}")))?;
        Ok(())
    }

    fn resume(&mut self) -> Result<(), VideoError> {
        tracing::debug!("GStreamer: resuming pipeline to Playing state");
        self.pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| VideoError::Generic(format!("Resume failed: {e:?}")))?;
        Ok(())
    }

    fn set_muted(&mut self, muted: bool) -> Result<(), VideoError> {
        self.audio_handle.set_muted(muted);
        Ok(())
    }

    fn set_volume(&mut self, volume: f32) -> Result<(), VideoError> {
        // Convert 0.0-1.0 to 0-100
        self.audio_handle.set_volume((volume * 100.0) as u32);
        Ok(())
    }

    fn is_eof(&self) -> bool {
        self.eof
    }

    fn buffering_percent(&self) -> i32 {
        self.buffering_percent
    }

    fn hw_accel_type(&self) -> HwAccelType {
        // GStreamer handles HW accel internally (auto-selects VA-API, etc.)
        HwAccelType::Vaapi
    }
}
