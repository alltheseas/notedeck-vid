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
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tracing::{debug, error, info, warn};
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
            IMFAttributes, IMFDXGIBuffer, IMFDXGIDeviceManager, IMFMediaBuffer, IMFMediaType,
            IMFSample, IMFSourceReader, MFCreateAttributes, MFCreateDXGIDeviceManager,
            MFCreateSourceReaderFromURL, MFMediaType_Video, MFShutdown, MFStartup,
            MFVideoFormat_NV12, MFVideoFormat_RGB32, MFSTARTUP_LITE, MF_MT_FRAME_RATE,
            MF_MT_FRAME_SIZE, MF_MT_MAJOR_TYPE, MF_MT_PIXEL_ASPECT_RATIO, MF_MT_SUBTYPE,
            MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, MF_SOURCE_READER_D3D_MANAGER,
            MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, MF_SOURCE_READER_FIRST_VIDEO_STREAM,
        },
        System::Com::{CoInitializeEx, CoUninitialize, COINIT_MULTITHREADED},
    },
};

/// Media Foundation version constant.
const MF_VERSION: u32 = 0x0002_0070; // MF_VERSION from SDK

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

        // Initialize Media Foundation
        unsafe {
            MFStartup(MF_VERSION, MFSTARTUP_LITE)
                .map_err(|e| VideoError::DecoderInit(format!("MFStartup failed: {}", e)))?;
        }

        // Create D3D11 device with video support
        let (device, context) = Self::create_d3d11_device(debug_logging)?;

        // Create DXGI device manager for hardware acceleration
        let dxgi_manager = Self::create_dxgi_manager(&device, debug_logging)?;

        // Create source reader with hardware acceleration
        let source_reader = Self::create_source_reader(url, &dxgi_manager, debug_logging)?;

        // Get video metadata
        let metadata = Self::extract_metadata(&source_reader, debug_logging)?;

        let hw_accel = HwAccelType::D3d11va;

        if debug_logging {
            info!(
                "WindowsVideoDecoder initialized: {}x{}, {:?}, hw_accel={:?}",
                metadata.width, metadata.height, metadata.codec, hw_accel
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
    fn create_source_reader(
        url: &str,
        dxgi_manager: &IMFDXGIDeviceManager,
        debug_logging: bool,
    ) -> Result<IMFSourceReader, VideoError> {
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

        // Configure output to NV12 (native hardware decoder format)
        Self::configure_output_format(&reader, debug_logging)?;

        if debug_logging {
            debug!("Source reader created successfully");
        }

        Ok(reader)
    }

    /// Configures the source reader to output NV12 format.
    fn configure_output_format(
        reader: &IMFSourceReader,
        debug_logging: bool,
    ) -> Result<(), VideoError> {
        if debug_logging {
            debug!("Configuring output format to NV12");
        }

        // Get the native media type to understand input format
        let native_type: IMFMediaType = unsafe {
            reader
                .GetNativeMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32, 0)
                .map_err(|e| VideoError::DecoderInit(format!("GetNativeMediaType failed: {}", e)))?
        };

        // Create output media type requesting NV12
        let output_type: IMFMediaType = unsafe {
            windows::Win32::Media::MediaFoundation::MFCreateMediaType()
                .map_err(|e| VideoError::DecoderInit(format!("MFCreateMediaType failed: {}", e)))?
        };

        unsafe {
            // Set major type to video
            output_type
                .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)
                .map_err(|e| {
                    VideoError::DecoderInit(format!("SetGUID major type failed: {}", e))
                })?;

            // Request NV12 output (native HW decoder format)
            output_type
                .SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_NV12)
                .map_err(|e| VideoError::DecoderInit(format!("SetGUID subtype failed: {}", e)))?;

            // Copy frame size from native type
            let mut frame_size: u64 = 0;
            if native_type
                .GetUINT64(&MF_MT_FRAME_SIZE, &mut frame_size)
                .is_ok()
            {
                output_type
                    .SetUINT64(&MF_MT_FRAME_SIZE, frame_size)
                    .map_err(|e| {
                        VideoError::DecoderInit(format!("SetUINT64 frame size failed: {}", e))
                    })?;
            }

            // Set the output type
            reader
                .SetCurrentMediaType(
                    MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32,
                    None,
                    &output_type,
                )
                .map_err(|e| {
                    VideoError::DecoderInit(format!("SetCurrentMediaType failed: {}", e))
                })?;
        }

        if debug_logging {
            debug!("Output format configured to NV12");
        }

        Ok(())
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
    #[profiling::function]
    fn extract_frame(&mut self, sample: &IMFSample) -> Result<DecodedFrame, VideoError> {
        // Get the media buffer from the sample
        let buffer: IMFMediaBuffer = unsafe {
            sample.ConvertToContiguousBuffer().map_err(|e| {
                VideoError::DecodeFailed(format!("ConvertToContiguousBuffer failed: {}", e))
            })?
        };

        // Try to get DXGI buffer for zero-copy (hardware decode path)
        if let Ok(dxgi_buffer) = buffer.cast::<IMFDXGIBuffer>() {
            return self.extract_frame_from_dxgi(&dxgi_buffer);
        }

        // Fall back to CPU buffer extraction
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

        // Get the D3D11 texture from DXGI buffer
        let texture: ID3D11Texture2D = unsafe {
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

            resource.ok_or_else(|| {
                VideoError::DecodeFailed("DXGI buffer resource is null".to_string())
            })?
        };

        // Get texture description
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe {
            texture.GetDesc(&mut desc);
        }

        // Create or reuse staging texture for CPU readback
        let staging = self.get_or_create_staging_texture(desc.Width, desc.Height, desc.Format)?;

        // Copy GPU texture to staging texture
        unsafe {
            self.context.CopyResource(&staging, &texture);
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
    fn extract_frame_from_cpu(&self, buffer: &IMFMediaBuffer) -> Result<DecodedFrame, VideoError> {
        if self.debug_logging {
            debug!("Extracting frame from CPU buffer (SW path)");
        }

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

        // Assume NV12 format for now
        let width = self.metadata.width;
        let height = self.metadata.height;
        let stride = width as usize;
        let y_size = stride * height as usize;
        let uv_height = (height as usize + 1) / 2;
        let uv_size = stride * uv_height;

        let y_data = unsafe { std::slice::from_raw_parts(data_ptr, y_size).to_vec() };
        let uv_data = unsafe { std::slice::from_raw_parts(data_ptr.add(y_size), uv_size).to_vec() };

        unsafe {
            buffer.Unlock().ok();
        }

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

        // Shutdown Media Foundation
        unsafe {
            let _ = MFShutdown();
            CoUninitialize();
        }

        if self.debug_logging {
            info!("WindowsVideoDecoder cleanup complete");
        }
    }
}

// Safety: WindowsVideoDecoder can be sent between threads
// The IMF* interfaces are thread-safe when used correctly
unsafe impl Send for WindowsVideoDecoder {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hw_accel_type() {
        assert_eq!(HwAccelType::platform_default(), HwAccelType::D3d11va);
    }
}
