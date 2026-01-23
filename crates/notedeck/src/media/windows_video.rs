//! Native Windows video decoder using Media Foundation with D3D11 hardware acceleration.
//!
//! This module provides hardware-accelerated video decoding on Windows using:
//! - Media Foundation (MF) for video demuxing and decoding
//! - DXVA2/D3D11VA for GPU-accelerated decode
//! - D3D11 textures for zero-copy frame output
//!
//! # Architecture
//!
//! The decoder uses `IMFSourceReader` in synchronous mode to poll frames,
//! similar to how the macOS decoder uses `AVPlayerItemVideoOutput`. This avoids
//! the broken `IMFAsyncCallback` implementation in windows-rs.
//!
//! Frame flow:
//! ```text
//! IMFSourceReader::ReadSample()
//!     → IMFSample
//!     → IMFDXGIBuffer (if HW accel)
//!     → ID3D11Texture2D
//!     → Copy to CPU (NV12/BGRA)
//!     → VideoFrame
//! ```
//!
//! # Hardware Acceleration
//!
//! When `MF_SOURCE_READER_D3D11_DEVICE` is set, Media Foundation automatically
//! uses DXVA2/D3D11VA for hardware decoding when available. The decoder falls
//! back to software decode if hardware acceleration fails.

use crate::media::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata,
};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;
use tracing::{debug, error, info, warn};

// ============================================================================
// Media Foundation Lifecycle Guard
// ============================================================================
//
// Media Foundation requires process-wide initialization via MFStartup/MFShutdown.
// These must be balanced: MFShutdown can only be called once for each MFStartup.
// If multiple decoders exist, calling MFShutdown when one drops would break others.
//
// This guard uses OnceLock<Mutex<Weak<MfGuard>>> to ensure:
// 1. MFStartup is called when the first decoder is created
// 2. MFShutdown is called when the last decoder drops (Weak allows refcount to reach 0)
// 3. If all decoders drop and a new one is created, MFStartup is called again
//
// NOTE: This is a necessary exception to the "no globals" rule because
// Media Foundation's API design requires process-wide state management.
// ============================================================================

/// Global Media Foundation guard slot. Stores a Weak reference so the guard
/// can actually be dropped when all decoders are gone.
static MF_GUARD: OnceLock<Mutex<Weak<MfGuard>>> = OnceLock::new();

/// RAII guard for Media Foundation lifecycle.
///
/// MFStartup is called when the guard is created.
/// MFShutdown is called when the guard is dropped (when last Arc reference drops).
struct MfGuard {
    /// Debug flag for logging.
    debug: bool,
}

impl MfGuard {
    /// Creates a new MF guard, calling MFStartup.
    fn new(debug: bool) -> Result<Self, VideoError> {
        if debug {
            info!("MFStartup: Initializing Media Foundation");
        }
        unsafe {
            MFStartup(MF_VERSION, MFSTARTUP_LITE)
                .map_err(|e| VideoError::DecoderInit(format!("MFStartup failed: {}", e)))?;
        }
        Ok(Self { debug })
    }
}

impl Drop for MfGuard {
    fn drop(&mut self) {
        if self.debug {
            info!("MFShutdown: Cleaning up Media Foundation");
        }
        unsafe {
            let _ = MFShutdown();
        }
    }
}

/// Gets or creates the global MF guard.
///
/// Returns a strong Arc to the guard. When all Arcs are dropped, the guard
/// is dropped and MFShutdown is called. A subsequent call will create a new guard.
fn get_mf_guard(debug: bool) -> Result<Arc<MfGuard>, VideoError> {
    let slot = MF_GUARD.get_or_init(|| Mutex::new(Weak::new()));
    let mut weak = slot.lock();

    // Try to upgrade existing weak reference
    if let Some(existing) = weak.upgrade() {
        return Ok(existing);
    }

    // No existing guard - create a new one
    let strong = Arc::new(MfGuard::new(debug)?);
    *weak = Arc::downgrade(&strong);
    Ok(strong)
}
use windows::{
    core::{Interface, HSTRING, PCWSTR},
    Win32::{
        Graphics::Direct3D::{D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL_11_0},
        Graphics::Direct3D11::{
            D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D,
            D3D11_CPU_ACCESS_READ, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            D3D11_CREATE_DEVICE_VIDEO_SUPPORT, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC,
            D3D11_USAGE_STAGING,
        },
        Graphics::Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_NV12},
        Media::MediaFoundation::{
            IMF2DBuffer2, IMFAttributes, IMFDXGIBuffer, IMFDXGIDeviceManager, IMFMediaBuffer,
            IMFMediaType, IMFSample, IMFSourceReader, MFCreateAttributes,
            MFCreateDXGIDeviceManager, MFCreateSourceReaderFromURL, MFMediaType_Video, MFShutdown,
            MFStartup, MFVideoFormat_NV12, MFVideoFormat_RGB32, MFSTARTUP_LITE, MF_MT_FRAME_RATE,
            MF_MT_FRAME_SIZE, MF_MT_MAJOR_TYPE, MF_MT_PIXEL_ASPECT_RATIO, MF_MT_SUBTYPE,
            MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, MF_SOURCE_READER_D3D_MANAGER,
            MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, MF_SOURCE_READER_FIRST_VIDEO_STREAM,
        },
        System::Com::{CoInitializeEx, CoUninitialize, COINIT_MULTITHREADED},
    },
};

/// Media Foundation version constant.
const MF_VERSION: u32 = 0x0002_0070; // MF_VERSION from SDK

/// Output format for the video decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    /// NV12 (native hardware decoder format)
    Nv12,
    /// RGB32/BGRA (fallback format)
    Rgb32,
}

/// Windows Media Foundation video decoder.
///
/// Uses `IMFSourceReader` for synchronous frame polling with D3D11 hardware acceleration.
pub struct WindowsVideoDecoder {
    /// Media Foundation source reader for video decode.
    source_reader: IMFSourceReader,

    /// D3D11 device for hardware acceleration.
    device: ID3D11Device,

    /// D3D11 device context for GPU operations.
    context: ID3D11DeviceContext,

    /// DXGI device manager for MF↔D3D11 integration.
    dxgi_manager: IMFDXGIDeviceManager,

    /// Video metadata (dimensions, duration, codec, etc.).
    metadata: VideoMetadata,

    /// Current playback position.
    position: Duration,

    /// Whether end-of-stream has been reached.
    eof: AtomicBool,

    /// Current hardware acceleration type.
    hw_accel: HwAccelType,

    /// Staging texture for CPU readback (reused to avoid allocations).
    staging_texture: Option<ID3D11Texture2D>,

    /// Debug logging enabled.
    debug_logging: bool,

    /// Media Foundation lifecycle guard.
    /// Ensures MFStartup is called once and MFShutdown only when last decoder drops.
    _mf_guard: Arc<MfGuard>,

    /// Output format (NV12 or RGB32).
    output_format: OutputFormat,
}

impl WindowsVideoDecoder {
    /// Creates a new Windows video decoder with debug logging control.
    ///
    /// # Arguments
    /// * `url` - URL or file path to the video
    /// * `debug_logging` - Enable verbose debug logging
    ///
    /// # Errors
    /// Returns `VideoError` if initialization fails.
    pub fn new(url: &str, debug_logging: bool) -> Result<Self, VideoError> {
        if debug_logging {
            info!("WindowsVideoDecoder::new() - Initializing for URL: {}", url);
        }

        // Initialize COM for this thread
        unsafe {
            CoInitializeEx(None, COINIT_MULTITHREADED).map_err(|e| {
                VideoError::DecoderInit(format!("COM initialization failed: {}", e))
            })?;
        }

        // Initialize Media Foundation via global guard
        // This ensures MFStartup is called once and MFShutdown only when last decoder drops
        let mf_guard = get_mf_guard(debug_logging)?;

        // Create D3D11 device with video support
        let (device, context) = Self::create_d3d11_device(debug_logging)?;

        // Create DXGI device manager for hardware acceleration
        let dxgi_manager = Self::create_dxgi_manager(&device, debug_logging)?;

        // Create source reader with hardware acceleration
        let (source_reader, output_format) =
            Self::create_source_reader(url, &dxgi_manager, debug_logging)?;

        // Get video metadata
        let metadata = Self::extract_metadata(&source_reader, debug_logging)?;

        let hw_accel = HwAccelType::D3d11va;

        if debug_logging {
            info!(
                "WindowsVideoDecoder initialized: {}x{}, {:?}, hw_accel={:?}, output_format={:?}",
                metadata.width, metadata.height, metadata.codec, hw_accel, output_format
            );
        }

        Ok(Self {
            source_reader,
            device,
            context,
            dxgi_manager,
            metadata,
            position: Duration::ZERO,
            eof: AtomicBool::new(false),
            hw_accel,
            staging_texture: None,
            debug_logging,
            _mf_guard: mf_guard,
            output_format,
        })
    }

    /// Creates a D3D11 device with video support.
    fn create_d3d11_device(
        debug_logging: bool,
    ) -> Result<(ID3D11Device, ID3D11DeviceContext), VideoError> {
        if debug_logging {
            debug!("Creating D3D11 device with video support");
        }

        let flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT;
        let feature_levels = [D3D_FEATURE_LEVEL_11_0];

        let mut device: Option<ID3D11Device> = None;
        let mut context: Option<ID3D11DeviceContext> = None;

        unsafe {
            D3D11CreateDevice(
                None, // Default adapter
                D3D_DRIVER_TYPE_HARDWARE,
                None, // No software rasterizer
                flags,
                Some(&feature_levels),
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
            .map_err(|e| VideoError::DecoderInit(format!("D3D11CreateDevice failed: {}", e)))?;
        }

        let device = device.ok_or_else(|| {
            VideoError::DecoderInit("D3D11CreateDevice returned null device".to_string())
        })?;
        let context = context.ok_or_else(|| {
            VideoError::DecoderInit("D3D11CreateDevice returned null context".to_string())
        })?;

        if debug_logging {
            debug!("D3D11 device created successfully");
        }

        Ok((device, context))
    }

    /// Creates a DXGI device manager for Media Foundation hardware acceleration.
    fn create_dxgi_manager(
        device: &ID3D11Device,
        debug_logging: bool,
    ) -> Result<IMFDXGIDeviceManager, VideoError> {
        if debug_logging {
            debug!("Creating DXGI device manager");
        }

        let mut reset_token: u32 = 0;
        let manager: IMFDXGIDeviceManager = unsafe {
            MFCreateDXGIDeviceManager(&mut reset_token).map_err(|e| {
                VideoError::DecoderInit(format!("MFCreateDXGIDeviceManager failed: {}", e))
            })?
        };

        unsafe {
            manager
                .ResetDevice(device, reset_token)
                .map_err(|e| VideoError::DecoderInit(format!("ResetDevice failed: {}", e)))?;
        }

        if debug_logging {
            debug!("DXGI device manager created with token {}", reset_token);
        }

        Ok(manager)
    }

    /// Creates a source reader with hardware acceleration enabled.
    ///
    /// Returns the source reader and the output format that was configured.
    fn create_source_reader(
        url: &str,
        dxgi_manager: &IMFDXGIDeviceManager,
        debug_logging: bool,
    ) -> Result<(IMFSourceReader, OutputFormat), VideoError> {
        if debug_logging {
            debug!("Creating source reader for: {}", url);
        }

        // Create attributes for source reader
        let attributes: IMFAttributes = unsafe {
            MFCreateAttributes(4)
                .map_err(|e| VideoError::DecoderInit(format!("MFCreateAttributes failed: {}", e)))?
        };

        // Enable hardware transforms
        unsafe {
            attributes
                .SetUINT32(&MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, 1)
                .map_err(|e| {
                    VideoError::DecoderInit(format!("SetUINT32 hardware transforms failed: {}", e))
                })?;
        }

        // Set DXGI device manager for D3D11 integration
        unsafe {
            attributes
                .SetUnknown(&MF_SOURCE_READER_D3D_MANAGER, dxgi_manager)
                .map_err(|e| {
                    VideoError::DecoderInit(format!("SetUnknown D3D manager failed: {}", e))
                })?;
        }

        // Enable video processing for format conversion
        unsafe {
            attributes
                .SetUINT32(&MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, 1)
                .map_err(|e| {
                    VideoError::DecoderInit(format!("SetUINT32 video processing failed: {}", e))
                })?;
        }

        // Create the source reader
        let url_hstring = HSTRING::from(url);
        let reader: IMFSourceReader = unsafe {
            MFCreateSourceReaderFromURL(&url_hstring, &attributes).map_err(|e| {
                VideoError::OpenFailed(format!("MFCreateSourceReaderFromURL failed: {}", e))
            })?
        };

        // Configure output format (NV12 preferred, RGB32 fallback)
        let output_format = Self::configure_output_format(&reader, debug_logging)?;

        if debug_logging {
            debug!(
                "Source reader created successfully with format {:?}",
                output_format
            );
        }

        Ok((reader, output_format))
    }

    /// Configures the source reader output format.
    ///
    /// Tries NV12 first (native hardware decoder format), falls back to RGB32
    /// if the GPU doesn't support NV12 output.
    ///
    /// Returns the format that was successfully configured.
    fn configure_output_format(
        reader: &IMFSourceReader,
        debug_logging: bool,
    ) -> Result<OutputFormat, VideoError> {
        // Get the native media type to copy frame size
        let native_type: IMFMediaType = unsafe {
            reader
                .GetNativeMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32, 0)
                .map_err(|e| VideoError::DecoderInit(format!("GetNativeMediaType failed: {}", e)))?
        };

        let mut frame_size: u64 = 0;
        unsafe {
            native_type
                .GetUINT64(&MF_MT_FRAME_SIZE, &mut frame_size)
                .ok();
        }

        // Try NV12 first (native HW decoder format)
        if Self::try_set_output_format(reader, &MFVideoFormat_NV12, frame_size, debug_logging) {
            if debug_logging {
                debug!("Output format configured to NV12");
            }
            return Ok(OutputFormat::Nv12);
        }

        // Fall back to RGB32 if NV12 not supported
        if debug_logging {
            warn!("NV12 output not supported, falling back to RGB32");
        }

        if Self::try_set_output_format(reader, &MFVideoFormat_RGB32, frame_size, debug_logging) {
            if debug_logging {
                debug!("Output format configured to RGB32");
            }
            return Ok(OutputFormat::Rgb32);
        }

        Err(VideoError::DecoderInit(
            "Failed to configure output format (neither NV12 nor RGB32 supported)".to_string(),
        ))
    }

    /// Attempts to set the output format to a specific subtype.
    ///
    /// Returns true on success, false if the format is not supported.
    fn try_set_output_format(
        reader: &IMFSourceReader,
        subtype: &windows::core::GUID,
        frame_size: u64,
        debug_logging: bool,
    ) -> bool {
        let output_type: IMFMediaType = unsafe {
            match windows::Win32::Media::MediaFoundation::MFCreateMediaType() {
                Ok(t) => t,
                Err(e) => {
                    if debug_logging {
                        debug!("MFCreateMediaType failed: {}", e);
                    }
                    return false;
                }
            }
        };

        unsafe {
            // Set major type to video
            if output_type
                .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)
                .is_err()
            {
                return false;
            }

            // Set requested subtype
            if output_type.SetGUID(&MF_MT_SUBTYPE, subtype).is_err() {
                return false;
            }

            // Set frame size if available
            if frame_size != 0
                && output_type
                    .SetUINT64(&MF_MT_FRAME_SIZE, frame_size)
                    .is_err()
            {
                return false;
            }

            // Try to set the output type
            reader
                .SetCurrentMediaType(
                    MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32,
                    None,
                    &output_type,
                )
                .is_ok()
        }
    }

    /// Extracts video metadata from the source reader.
    fn extract_metadata(
        reader: &IMFSourceReader,
        debug_logging: bool,
    ) -> Result<VideoMetadata, VideoError> {
        if debug_logging {
            debug!("Extracting video metadata");
        }

        let media_type: IMFMediaType = unsafe {
            reader
                .GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32)
                .map_err(|e| {
                    VideoError::DecoderInit(format!("GetCurrentMediaType failed: {}", e))
                })?
        };

        // Extract frame size
        let mut frame_size: u64 = 0;
        unsafe {
            media_type
                .GetUINT64(&MF_MT_FRAME_SIZE, &mut frame_size)
                .ok();
        }
        let width = (frame_size >> 32) as u32;
        let height = (frame_size & 0xFFFFFFFF) as u32;

        // Extract frame rate
        let mut frame_rate: u64 = 0;
        unsafe {
            media_type
                .GetUINT64(&MF_MT_FRAME_RATE, &mut frame_rate)
                .ok();
        }
        let fps_num = (frame_rate >> 32) as f32;
        let fps_den = (frame_rate & 0xFFFFFFFF) as f32;
        let frame_rate = if fps_den > 0.0 {
            fps_num / fps_den
        } else {
            30.0
        };

        // Extract pixel aspect ratio
        let mut par: u64 = 0;
        unsafe {
            media_type
                .GetUINT64(&MF_MT_PIXEL_ASPECT_RATIO, &mut par)
                .ok();
        }
        let par_num = (par >> 32) as f32;
        let par_den = (par & 0xFFFFFFFF) as f32;
        let pixel_aspect_ratio = if par_den > 0.0 {
            par_num / par_den
        } else {
            1.0
        };

        // Get duration from presentation descriptor
        let duration = Self::get_duration(reader);

        let metadata = VideoMetadata {
            width,
            height,
            duration,
            frame_rate,
            codec: "h264".to_string(), // TODO: Extract actual codec
            pixel_aspect_ratio,
        };

        if debug_logging {
            info!(
                "Video metadata: {}x{} @ {:.2} fps, duration: {:?}",
                metadata.width, metadata.height, metadata.frame_rate, metadata.duration
            );
        }

        Ok(metadata)
    }

    /// Gets the video duration from the source reader.
    fn get_duration(reader: &IMFSourceReader) -> Option<Duration> {
        // MF_PD_DURATION attribute GUID
        let mf_pd_duration = windows::core::GUID::from_u128(0xc8c9b0c8_5c0a_4a8d_9801_c0c5a15e6a1f);

        let mut duration_100ns: u64 = 0;
        unsafe {
            if let Ok(source) = reader.GetServiceForStream(
                MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32,
                &windows::core::GUID::zeroed(),
                &IMFAttributes::IID,
            ) {
                let attrs: IMFAttributes = source.cast().ok()?;
                if attrs
                    .GetUINT64(&mf_pd_duration, &mut duration_100ns)
                    .is_ok()
                {
                    return Some(Duration::from_nanos(duration_100ns * 100));
                }
            }
        }
        None
    }

    /// Reads and decodes the next video frame.
    #[profiling::function]
    fn read_sample(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        let mut flags: u32 = 0;
        let mut timestamp: i64 = 0;
        let mut sample: Option<IMFSample> = None;

        unsafe {
            self.source_reader
                .ReadSample(
                    MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32,
                    0,
                    None,
                    Some(&mut flags),
                    Some(&mut timestamp),
                    Some(&mut sample),
                )
                .map_err(|e| VideoError::DecodeFailed(format!("ReadSample failed: {}", e)))?;
        }

        // Check for end of stream
        const MF_SOURCE_READERF_ENDOFSTREAM: u32 = 0x1;
        if flags & MF_SOURCE_READERF_ENDOFSTREAM != 0 {
            self.eof.store(true, Ordering::SeqCst);
            if self.debug_logging {
                debug!("End of stream reached");
            }
            return Ok(None);
        }

        // Check for stream tick (no data yet)
        const MF_SOURCE_READERF_STREAMTICK: u32 = 0x100;
        if flags & MF_SOURCE_READERF_STREAMTICK != 0 {
            if self.debug_logging {
                debug!("Stream tick, no frame yet");
            }
            return Ok(None);
        }

        let sample = match sample {
            Some(s) => s,
            None => return Ok(None),
        };

        // Convert timestamp to Duration (100ns units)
        let pts = Duration::from_nanos(timestamp as u64 * 100);
        self.position = pts;

        // Extract frame data from sample
        let frame = self.extract_frame(&sample)?;

        if self.debug_logging {
            debug!("Decoded frame at PTS {:?}", pts);
        }

        Ok(Some(VideoFrame::new(pts, frame)))
    }

    /// Extracts frame data from an IMFSample.
    ///
    /// Checks for DXGI buffers BEFORE calling ConvertToContiguousBuffer,
    /// since that operation may copy GPU data to system memory.
    #[profiling::function]
    fn extract_frame(&mut self, sample: &IMFSample) -> Result<DecodedFrame, VideoError> {
        // First, check original sample buffers for DXGI (hardware decode path)
        // We must do this BEFORE ConvertToContiguousBuffer which may copy to CPU memory.
        let buffer_count = unsafe {
            sample
                .GetBufferCount()
                .map_err(|e| VideoError::DecodeFailed(format!("GetBufferCount failed: {}", e)))?
        };

        for i in 0..buffer_count {
            let buffer: IMFMediaBuffer = unsafe {
                match sample.GetBufferByIndex(i) {
                    Ok(b) => b,
                    Err(_) => continue,
                }
            };

            // Try to cast to DXGI buffer for zero-copy hardware path
            if let Ok(dxgi_buffer) = buffer.cast::<IMFDXGIBuffer>() {
                if self.debug_logging {
                    debug!("Found DXGI buffer at index {}, using HW path", i);
                }
                return self.extract_frame_from_dxgi(&dxgi_buffer);
            }
        }

        // No DXGI buffer found - use CPU path
        // ConvertToContiguousBuffer is safe here since we're already on CPU path
        let buffer: IMFMediaBuffer = unsafe {
            sample.ConvertToContiguousBuffer().map_err(|e| {
                VideoError::DecodeFailed(format!("ConvertToContiguousBuffer failed: {}", e))
            })?
        };

        self.extract_frame_from_cpu(&buffer)
    }

    /// Extracts frame from DXGI buffer (hardware decode path).
    #[profiling::function]
    fn extract_frame_from_dxgi(
        &mut self,
        dxgi_buffer: &IMFDXGIBuffer,
    ) -> Result<DecodedFrame, VideoError> {
        if self.debug_logging {
            debug!("Extracting frame from DXGI buffer (HW path)");
        }

        // Get the D3D11 texture and subresource index from DXGI buffer
        let (texture, subresource_index): (ID3D11Texture2D, u32) = unsafe {
            let mut resource: Option<ID3D11Texture2D> = None;
            let mut subresource: u32 = 0;
            dxgi_buffer
                .GetResource(&ID3D11Texture2D::IID, &mut resource as *mut _ as *mut _)
                .map_err(|e| VideoError::DecodeFailed(format!("GetResource failed: {}", e)))?;
            dxgi_buffer
                .GetSubresourceIndex(&mut subresource)
                .map_err(|e| {
                    VideoError::DecodeFailed(format!("GetSubresourceIndex failed: {}", e))
                })?;

            let tex = resource.ok_or_else(|| {
                VideoError::DecodeFailed("DXGI buffer resource is null".to_string())
            })?;
            (tex, subresource)
        };

        // Get texture description
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe {
            texture.GetDesc(&mut desc);
        }

        if self.debug_logging {
            debug!(
                "DXGI texture: {}x{}, format={:?}, subresource={}",
                desc.Width, desc.Height, desc.Format, subresource_index
            );
        }

        // Create or reuse staging texture for CPU readback
        let staging = self.get_or_create_staging_texture(desc.Width, desc.Height, desc.Format)?;

        // Copy from the specific subresource of the GPU texture to subresource 0 of staging
        // This is important because DXVA decoders may use texture arrays where each
        // decoded frame is in a different array slice (subresource).
        unsafe {
            self.context.CopySubresourceRegion(
                &staging,
                0, // Destination subresource (staging texture has only one)
                0,
                0,
                0, // Destination x, y, z
                &texture,
                subresource_index, // Source subresource from DXGI buffer
                None,              // Copy entire subresource
            );
        }

        // Map staging texture and extract pixel data
        self.map_staging_texture(&staging, desc.Width, desc.Height, desc.Format)
    }

    /// Gets or creates a staging texture for CPU readback.
    fn get_or_create_staging_texture(
        &mut self,
        width: u32,
        height: u32,
        format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    ) -> Result<ID3D11Texture2D, VideoError> {
        // Check if existing staging texture is compatible
        if let Some(ref staging) = self.staging_texture {
            let mut desc = D3D11_TEXTURE2D_DESC::default();
            unsafe {
                staging.GetDesc(&mut desc);
            }
            if desc.Width == width && desc.Height == height && desc.Format == format {
                return Ok(staging.clone());
            }
        }

        // Create new staging texture
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: format,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_STAGING,
            BindFlags: windows::Win32::Graphics::Direct3D11::D3D11_BIND_FLAG(0),
            CPUAccessFlags: D3D11_CPU_ACCESS_READ,
            MiscFlags: windows::Win32::Graphics::Direct3D11::D3D11_RESOURCE_MISC_FLAG(0),
        };

        let staging: ID3D11Texture2D = unsafe {
            let mut texture: Option<ID3D11Texture2D> = None;
            self.device
                .CreateTexture2D(&desc, None, Some(&mut texture))
                .map_err(|e| VideoError::DecodeFailed(format!("CreateTexture2D failed: {}", e)))?;
            texture.ok_or_else(|| {
                VideoError::DecodeFailed("CreateTexture2D returned null".to_string())
            })?
        };

        self.staging_texture = Some(staging.clone());
        Ok(staging)
    }

    /// Maps a staging texture and extracts pixel data.
    fn map_staging_texture(
        &self,
        staging: &ID3D11Texture2D,
        width: u32,
        height: u32,
        format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    ) -> Result<DecodedFrame, VideoError> {
        use windows::Win32::Graphics::Direct3D11::{D3D11_MAPPED_SUBRESOURCE, D3D11_MAP_READ};

        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe {
            self.context
                .Map(staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .map_err(|e| VideoError::DecodeFailed(format!("Map failed: {}", e)))?;
        }

        let result = self.copy_mapped_data(&mapped, width, height, format);

        unsafe {
            self.context.Unmap(staging, 0);
        }

        result
    }

    /// Copies mapped texture data to a CpuFrame.
    fn copy_mapped_data(
        &self,
        mapped: &windows::Win32::Graphics::Direct3D11::D3D11_MAPPED_SUBRESOURCE,
        width: u32,
        height: u32,
        format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    ) -> Result<DecodedFrame, VideoError> {
        let stride = mapped.RowPitch as usize;
        let data_ptr = mapped.pData as *const u8;

        match format {
            f if f == DXGI_FORMAT_NV12 => {
                // NV12: Y plane followed by interleaved UV plane
                let y_size = stride * height as usize;
                let uv_height = (height as usize + 1) / 2;
                let uv_size = stride * uv_height;

                let y_data = unsafe { std::slice::from_raw_parts(data_ptr, y_size).to_vec() };
                let uv_data =
                    unsafe { std::slice::from_raw_parts(data_ptr.add(y_size), uv_size).to_vec() };

                let frame = CpuFrame::new(
                    PixelFormat::Nv12,
                    width,
                    height,
                    vec![
                        Plane {
                            data: y_data,
                            stride,
                        },
                        Plane {
                            data: uv_data,
                            stride,
                        },
                    ],
                );

                Ok(DecodedFrame::Cpu(frame))
            }
            f if f == DXGI_FORMAT_B8G8R8A8_UNORM => {
                // BGRA: single plane
                let size = stride * height as usize;
                let data = unsafe { std::slice::from_raw_parts(data_ptr, size).to_vec() };

                let frame = CpuFrame::new(
                    PixelFormat::Bgra,
                    width,
                    height,
                    vec![Plane { data, stride }],
                );

                Ok(DecodedFrame::Cpu(frame))
            }
            _ => Err(VideoError::UnsupportedFormat(format!(
                "Unsupported DXGI format: {:?}",
                format
            ))),
        }
    }

    /// Extracts frame from CPU buffer (software decode fallback).
    ///
    /// Handles stride alignment properly using IMF2DBuffer2 when available,
    /// falling back to stride calculation from buffer size.
    fn extract_frame_from_cpu(&self, buffer: &IMFMediaBuffer) -> Result<DecodedFrame, VideoError> {
        if self.debug_logging {
            debug!(
                "Extracting frame from CPU buffer (SW path), format={:?}",
                self.output_format
            );
        }

        let width = self.metadata.width;
        let height = self.metadata.height;

        // Try IMF2DBuffer2 first for proper stride handling
        if let Ok(buffer_2d) = buffer.cast::<IMF2DBuffer2>() {
            return self.extract_from_2d_buffer(&buffer_2d, width, height);
        }

        // Fallback: Lock buffer and calculate stride from buffer size
        let mut data_ptr: *mut u8 = std::ptr::null_mut();
        let mut max_length: u32 = 0;
        let mut current_length: u32 = 0;

        unsafe {
            buffer
                .Lock(
                    &mut data_ptr,
                    Some(&mut max_length),
                    Some(&mut current_length),
                )
                .map_err(|e| VideoError::DecodeFailed(format!("Lock failed: {}", e)))?;
        }

        let height_usize = height as usize;
        let frame = match self.output_format {
            OutputFormat::Nv12 => {
                // For NV12: total_size = stride * height * 1.5
                let uv_height = (height_usize + 1) / 2;
                let total_height = height_usize + uv_height;
                let stride = (current_length as usize / total_height).max(width as usize);

                if self.debug_logging {
                    debug!(
                        "NV12 CPU buffer: {}x{}, stride={}, buffer_len={}",
                        width, height, stride, current_length
                    );
                }

                let y_size = stride * height_usize;
                let uv_size = stride * uv_height;

                let y_data = unsafe { std::slice::from_raw_parts(data_ptr, y_size).to_vec() };
                let uv_data =
                    unsafe { std::slice::from_raw_parts(data_ptr.add(y_size), uv_size).to_vec() };

                CpuFrame::new(
                    PixelFormat::Nv12,
                    width,
                    height,
                    vec![
                        Plane {
                            data: y_data,
                            stride,
                        },
                        Plane {
                            data: uv_data,
                            stride,
                        },
                    ],
                )
            }
            OutputFormat::Rgb32 => {
                // For RGB32: total_size = stride * height, where stride >= width * 4
                let bytes_per_pixel = 4usize;
                let stride =
                    (current_length as usize / height_usize).max(width as usize * bytes_per_pixel);

                if self.debug_logging {
                    debug!(
                        "RGB32 CPU buffer: {}x{}, stride={}, buffer_len={}",
                        width, height, stride, current_length
                    );
                }

                let size = stride * height_usize;
                let data = unsafe { std::slice::from_raw_parts(data_ptr, size).to_vec() };

                CpuFrame::new(
                    PixelFormat::Bgra,
                    width,
                    height,
                    vec![Plane { data, stride }],
                )
            }
        };

        unsafe {
            buffer.Unlock().ok();
        }

        Ok(DecodedFrame::Cpu(frame))
    }

    /// Extracts frame from IMF2DBuffer2 with proper stride handling.
    fn extract_from_2d_buffer(
        &self,
        buffer_2d: &IMF2DBuffer2,
        width: u32,
        height: u32,
    ) -> Result<DecodedFrame, VideoError> {
        if self.debug_logging {
            debug!(
                "Using IMF2DBuffer2 for proper stride handling, format={:?}",
                self.output_format
            );
        }

        let mut scanline0: *mut u8 = std::ptr::null_mut();
        let mut pitch: i32 = 0;
        let mut buffer_start: *mut u8 = std::ptr::null_mut();
        let mut buffer_length: u32 = 0;

        unsafe {
            buffer_2d
                .Lock2DSize(
                    windows::Win32::Media::MediaFoundation::MF2DBuffer_LockFlags_Read,
                    &mut scanline0,
                    &mut pitch,
                    &mut buffer_start,
                    &mut buffer_length,
                )
                .map_err(|e| VideoError::DecodeFailed(format!("Lock2DSize failed: {}", e)))?;
        }

        let stride = pitch.unsigned_abs() as usize;
        let height_usize = height as usize;

        if self.debug_logging {
            debug!(
                "IMF2DBuffer2: {}x{}, pitch={}, buffer_len={}, format={:?}",
                width, height, pitch, buffer_length, self.output_format
            );
        }

        let frame = match self.output_format {
            OutputFormat::Nv12 => {
                let uv_height = (height_usize + 1) / 2;
                let y_size = stride * height_usize;
                let uv_size = stride * uv_height;

                let y_data = unsafe { std::slice::from_raw_parts(scanline0, y_size).to_vec() };
                let uv_data =
                    unsafe { std::slice::from_raw_parts(scanline0.add(y_size), uv_size).to_vec() };

                CpuFrame::new(
                    PixelFormat::Nv12,
                    width,
                    height,
                    vec![
                        Plane {
                            data: y_data,
                            stride,
                        },
                        Plane {
                            data: uv_data,
                            stride,
                        },
                    ],
                )
            }
            OutputFormat::Rgb32 => {
                let size = stride * height_usize;
                let data = unsafe { std::slice::from_raw_parts(scanline0, size).to_vec() };

                CpuFrame::new(
                    PixelFormat::Bgra,
                    width,
                    height,
                    vec![Plane { data, stride }],
                )
            }
        };

        unsafe {
            buffer_2d.Unlock2D().ok();
        }

        Ok(DecodedFrame::Cpu(frame))
    }
}

impl VideoDecoderBackend for WindowsVideoDecoder {
    #[profiling::function]
    fn open(url: &str) -> Result<Self, VideoError>
    where
        Self: Sized,
    {
        // Enable debug logging via environment variable for easy tester control
        // Set NOTEDECK_VIDEO_DEBUG=1 to enable verbose logging
        let debug_logging = std::env::var("NOTEDECK_VIDEO_DEBUG")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(true); // Default ON for initial testing phase

        Self::new(url, debug_logging)
    }

    #[profiling::function]
    fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        self.read_sample()
    }

    #[profiling::function]
    fn seek(&mut self, position: Duration) -> Result<(), VideoError> {
        if self.debug_logging {
            debug!("Seeking to {:?}", position);
        }

        // Convert Duration to 100ns units for Media Foundation
        let position_100ns = position.as_nanos() as i64 / 100;
        let prop_variant = windows::Win32::System::Com::StructuredStorage::PROPVARIANT::default();

        // Set position in PROPVARIANT
        unsafe {
            let inner = &mut *(std::ptr::addr_of!(prop_variant)
                as *mut windows::Win32::System::Com::StructuredStorage::PROPVARIANT);
            inner.Anonymous.Anonymous.vt = windows::Win32::System::Variant::VT_I8;
            inner.Anonymous.Anonymous.Anonymous.hVal = std::mem::transmute(position_100ns);
        }

        unsafe {
            self.source_reader
                .SetCurrentPosition(&windows::core::GUID::zeroed(), &prop_variant)
                .map_err(|e| VideoError::SeekFailed(format!("SetCurrentPosition failed: {}", e)))?;
        }

        self.position = position;
        self.eof.store(false, Ordering::SeqCst);

        if self.debug_logging {
            debug!("Seek completed to {:?}", position);
        }

        Ok(())
    }

    fn metadata(&self) -> &VideoMetadata {
        &self.metadata
    }

    fn is_eof(&self) -> bool {
        self.eof.load(Ordering::SeqCst)
    }

    fn hw_accel_type(&self) -> HwAccelType {
        self.hw_accel
    }
}

impl Drop for WindowsVideoDecoder {
    fn drop(&mut self) {
        if self.debug_logging {
            debug!("WindowsVideoDecoder dropping, cleaning up");
        }

        // Release staging texture
        self.staging_texture = None;

        // Note: MFShutdown is handled by the _mf_guard Arc<MfGuard>.
        // It will only call MFShutdown when the last decoder is dropped.

        // Uninitialize COM for this thread
        unsafe {
            CoUninitialize();
        }

        if self.debug_logging {
            info!("WindowsVideoDecoder cleanup complete");
        }
    }
}

// Note: WindowsVideoDecoder is NOT Send because COM objects are thread-affine.
// The decoder must be created and used on the same thread (the decode thread).
// Use DecodeThread::new_from_factory to ensure proper thread confinement.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hw_accel_type() {
        assert_eq!(HwAccelType::platform_default(), HwAccelType::D3d11va);
    }
}
