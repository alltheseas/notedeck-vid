//! Video texture management for wgpu rendering.
//!
//! This module handles uploading decoded video frames to GPU textures
//! and provides the rendering pipeline for YUV to RGB conversion.

use std::borrow::Cow;
use std::num::NonZeroU64;

use eframe::egui_wgpu::{self, wgpu};
use egui::Rect;

use super::video::{CpuFrame, DecodedFrame, PixelFormat, VideoFrame};

/// wgpu requires bytes_per_row to be aligned to this value.
const WGPU_COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

/// Aligns a value up to the nearest multiple of alignment.
fn align_up(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}

/// Pads row data to meet wgpu's bytes_per_row alignment requirement.
/// Returns (aligned_bytes_per_row, padded_data) if padding is needed,
/// or None if the original stride is already aligned.
fn pad_plane_data(data: &[u8], stride: usize, height: u32) -> Option<(u32, Vec<u8>)> {
    let stride_u32 = stride as u32;
    let aligned_stride = align_up(stride_u32, WGPU_COPY_BYTES_PER_ROW_ALIGNMENT);

    if aligned_stride == stride_u32 {
        return None; // Already aligned
    }

    // Need to pad each row
    let mut padded = Vec::with_capacity((aligned_stride * height) as usize);
    for row in 0..height as usize {
        let row_start = row * stride;
        let row_end = row_start + stride;
        if row_end <= data.len() {
            padded.extend_from_slice(&data[row_start..row_end]);
            // Add padding bytes
            padded.resize(padded.len() + (aligned_stride - stride_u32) as usize, 0);
        }
    }

    Some((aligned_stride, padded))
}

/// Resources for rendering video frames via wgpu.
///
/// This struct is stored in egui's callback resources and contains
/// all the GPU resources needed to render video frames.
pub struct VideoRenderResources {
    /// The render pipeline for YUV to RGB conversion
    pipeline_yuv420p: wgpu::RenderPipeline,
    /// The render pipeline for NV12 format
    pipeline_nv12: wgpu::RenderPipeline,
    /// The render pipeline for RGB passthrough
    pipeline_rgb: wgpu::RenderPipeline,
    /// Bind group layout for video textures
    bind_group_layout: wgpu::BindGroupLayout,
    /// Uniform buffer for transform data
    uniform_buffer: wgpu::Buffer,
    /// Texture sampler
    sampler: wgpu::Sampler,
}

impl VideoRenderResources {
    /// Creates video render resources.
    ///
    /// This should be called once during application initialization.
    pub fn new(wgpu_render_state: &egui_wgpu::RenderState) -> Self {
        let device = &wgpu_render_state.device;

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("video_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("video.wgsl"))),
        });

        // Create sampler for texture sampling
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("video_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create uniform buffer for video transform
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("video_uniform_buffer"),
            size: 16, // 4 floats for transform
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("video_bind_group_layout"),
            entries: &[
                // Uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(16),
                    },
                    count: None,
                },
                // Y texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // U texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // V texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("video_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipelines for different formats
        let create_pipeline = |entry_point: &str, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry_point),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu_render_state.target_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };

        let pipeline_yuv420p = create_pipeline("fs_main", "video_pipeline_yuv420p");
        let pipeline_nv12 = create_pipeline("fs_main_nv12", "video_pipeline_nv12");
        let pipeline_rgb = create_pipeline("fs_main_rgb", "video_pipeline_rgb");

        Self {
            pipeline_yuv420p,
            pipeline_nv12,
            pipeline_rgb,
            bind_group_layout,
            uniform_buffer,
            sampler,
        }
    }

    /// Registers the video render resources with egui's callback system.
    pub fn register(wgpu_render_state: &egui_wgpu::RenderState) {
        let resources = Self::new(wgpu_render_state);
        wgpu_render_state
            .renderer
            .write()
            .callback_resources
            .insert(resources);
    }
}

/// A video texture that can be uploaded to the GPU and rendered.
///
/// This handles the GPU-side representation of a video frame, including
/// texture creation, upload, and bind group management.
pub struct VideoTexture {
    /// Y plane texture (or RGB texture for RGB formats)
    y_texture: wgpu::Texture,
    y_view: wgpu::TextureView,
    /// U plane texture (or UV texture for NV12)
    u_texture: wgpu::Texture,
    u_view: wgpu::TextureView,
    /// V plane texture (unused for NV12/RGB)
    v_texture: wgpu::Texture,
    v_view: wgpu::TextureView,
    /// Current bind group
    bind_group: wgpu::BindGroup,
    /// Video dimensions
    width: u32,
    height: u32,
    /// Pixel format
    format: PixelFormat,
}

impl VideoTexture {
    /// Creates a new video texture with the specified dimensions and format.
    pub fn new(
        device: &wgpu::Device,
        resources: &VideoRenderResources,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Self {
        // Calculate texture sizes based on format
        let (y_size, u_size, v_size) = match format {
            PixelFormat::Yuv420p => (
                (width, height),
                (width / 2, height / 2),
                (width / 2, height / 2),
            ),
            PixelFormat::Nv12 => ((width, height), (width / 2, height / 2), (1, 1)),
            PixelFormat::Rgb24 | PixelFormat::Rgba | PixelFormat::Bgra => {
                ((width, height), (1, 1), (1, 1))
            }
        };

        // Determine texture format
        let y_format = match format {
            PixelFormat::Yuv420p | PixelFormat::Nv12 => wgpu::TextureFormat::R8Unorm,
            PixelFormat::Rgb24 => wgpu::TextureFormat::Rgba8Unorm, // Will need conversion
            PixelFormat::Rgba => wgpu::TextureFormat::Rgba8Unorm,
            PixelFormat::Bgra => wgpu::TextureFormat::Bgra8Unorm,
        };

        let u_format = match format {
            PixelFormat::Yuv420p => wgpu::TextureFormat::R8Unorm,
            PixelFormat::Nv12 => wgpu::TextureFormat::Rg8Unorm, // Interleaved UV
            _ => wgpu::TextureFormat::R8Unorm,
        };

        let v_format = wgpu::TextureFormat::R8Unorm;

        // Create textures
        let create_texture = |size: (u32, u32), format: wgpu::TextureFormat, label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size.0.max(1),
                    height: size.1.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };

        let y_texture = create_texture(y_size, y_format, "video_y_texture");
        let u_texture = create_texture(u_size, u_format, "video_u_texture");
        let v_texture = create_texture(v_size, v_format, "video_v_texture");

        let y_view = y_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let u_view = u_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let v_view = v_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("video_bind_group"),
            layout: &resources.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&u_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&v_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&resources.sampler),
                },
            ],
        });

        Self {
            y_texture,
            y_view,
            u_texture,
            u_view,
            v_texture,
            v_view,
            bind_group,
            width,
            height,
            format,
        }
    }

    /// Uploads a video frame to the GPU textures.
    pub fn upload(&self, queue: &wgpu::Queue, frame: &CpuFrame) {
        match frame.format {
            PixelFormat::Yuv420p => {
                self.upload_yuv420p(queue, frame);
            }
            PixelFormat::Nv12 => {
                self.upload_nv12(queue, frame);
            }
            PixelFormat::Rgba | PixelFormat::Bgra => {
                self.upload_rgba(queue, frame);
            }
            PixelFormat::Rgb24 => {
                self.upload_rgb24(queue, frame);
            }
        }
    }

    fn upload_yuv420p(&self, queue: &wgpu::Queue, frame: &CpuFrame) {
        // Upload Y plane
        if let Some(y_plane) = frame.plane(0) {
            let (bytes_per_row, data) =
                if let Some((aligned, padded)) = pad_plane_data(&y_plane.data, y_plane.stride, frame.height) {
                    (aligned, padded)
                } else {
                    (y_plane.stride as u32, y_plane.data.clone())
                };

            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.y_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(frame.height),
                },
                wgpu::Extent3d {
                    width: frame.width,
                    height: frame.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        let uv_height = frame.height / 2;

        // Upload U plane
        if let Some(u_plane) = frame.plane(1) {
            let (bytes_per_row, data) =
                if let Some((aligned, padded)) = pad_plane_data(&u_plane.data, u_plane.stride, uv_height) {
                    (aligned, padded)
                } else {
                    (u_plane.stride as u32, u_plane.data.clone())
                };

            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.u_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(uv_height),
                },
                wgpu::Extent3d {
                    width: frame.width / 2,
                    height: uv_height,
                    depth_or_array_layers: 1,
                },
            );
        }

        // Upload V plane
        if let Some(v_plane) = frame.plane(2) {
            let (bytes_per_row, data) =
                if let Some((aligned, padded)) = pad_plane_data(&v_plane.data, v_plane.stride, uv_height) {
                    (aligned, padded)
                } else {
                    (v_plane.stride as u32, v_plane.data.clone())
                };

            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.v_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(uv_height),
                },
                wgpu::Extent3d {
                    width: frame.width / 2,
                    height: uv_height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    fn upload_nv12(&self, queue: &wgpu::Queue, frame: &CpuFrame) {
        // Upload Y plane
        if let Some(y_plane) = frame.plane(0) {
            let (bytes_per_row, data) =
                if let Some((aligned, padded)) = pad_plane_data(&y_plane.data, y_plane.stride, frame.height) {
                    (aligned, padded)
                } else {
                    (y_plane.stride as u32, y_plane.data.clone())
                };

            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.y_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(frame.height),
                },
                wgpu::Extent3d {
                    width: frame.width,
                    height: frame.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        let uv_height = frame.height / 2;

        // Upload interleaved UV plane
        if let Some(uv_plane) = frame.plane(1) {
            let (bytes_per_row, data) =
                if let Some((aligned, padded)) = pad_plane_data(&uv_plane.data, uv_plane.stride, uv_height) {
                    (aligned, padded)
                } else {
                    (uv_plane.stride as u32, uv_plane.data.clone())
                };

            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.u_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(uv_height),
                },
                wgpu::Extent3d {
                    width: frame.width / 2,
                    height: uv_height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    fn upload_rgba(&self, queue: &wgpu::Queue, frame: &CpuFrame) {
        if let Some(plane) = frame.plane(0) {
            let (bytes_per_row, data) =
                if let Some((aligned, padded)) = pad_plane_data(&plane.data, plane.stride, frame.height) {
                    (aligned, padded)
                } else {
                    (plane.stride as u32, plane.data.clone())
                };

            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.y_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(frame.height),
                },
                wgpu::Extent3d {
                    width: frame.width,
                    height: frame.height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    fn upload_rgb24(&self, queue: &wgpu::Queue, frame: &CpuFrame) {
        // RGB24 needs to be converted to RGBA with proper alignment
        if let Some(plane) = frame.plane(0) {
            let rgba_stride = frame.width * 4;
            let aligned_stride = align_up(rgba_stride, WGPU_COPY_BYTES_PER_ROW_ALIGNMENT);
            let padding = (aligned_stride - rgba_stride) as usize;

            let mut rgba_data = Vec::with_capacity((aligned_stride * frame.height) as usize);
            for y in 0..frame.height as usize {
                for x in 0..frame.width as usize {
                    let offset = y * plane.stride + x * 3;
                    if offset + 2 < plane.data.len() {
                        rgba_data.push(plane.data[offset]); // R
                        rgba_data.push(plane.data[offset + 1]); // G
                        rgba_data.push(plane.data[offset + 2]); // B
                        rgba_data.push(255); // A
                    }
                }
                // Add padding bytes for alignment
                rgba_data.resize(rgba_data.len() + padding, 0);
            }

            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.y_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &rgba_data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_stride),
                    rows_per_image: Some(frame.height),
                },
                wgpu::Extent3d {
                    width: frame.width,
                    height: frame.height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    /// Returns the video dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Returns the pixel format.
    pub fn format(&self) -> PixelFormat {
        self.format
    }
}

/// Data passed to the video render callback.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VideoRenderData {
    /// Transform: [scale_x, scale_y, offset_x, offset_y]
    pub transform: [f32; 4],
}

/// Callback for rendering video frames via egui's paint callback system.
use super::video_player::PendingFrame;

pub struct VideoRenderCallback {
    /// The video texture to render
    pub texture: std::sync::Arc<std::sync::Mutex<Option<VideoTexture>>>,
    /// Pending frame data for texture creation/upload
    pub pending_frame: std::sync::Arc<std::sync::Mutex<PendingFrame>>,
    /// The pixel format of the current frame
    pub format: PixelFormat,
    /// The destination rectangle in clip space
    pub rect: Rect,
}

impl egui_wgpu::CallbackTrait for VideoRenderCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let video_resources: &VideoRenderResources = resources.get().unwrap();

        // Handle pending frame: create texture if needed and upload data
        {
            let mut pending = self.pending_frame.lock().unwrap();
            if let Some(cpu_frame) = pending.frame.take() {
                let needs_recreate = pending.needs_recreate;
                pending.needs_recreate = false;
                drop(pending); // Release lock before acquiring texture lock

                let mut texture_guard = self.texture.lock().unwrap();

                // Create texture if needed
                if needs_recreate || texture_guard.is_none() {
                    let new_texture = VideoTexture::new(
                        device,
                        video_resources,
                        cpu_frame.width,
                        cpu_frame.height,
                        cpu_frame.format,
                    );
                    *texture_guard = Some(new_texture);
                }

                // Upload frame data
                if let Some(ref texture) = *texture_guard {
                    texture.upload(queue, &cpu_frame);
                }
            }
        }

        // Calculate transform from rect to clip space
        let width = screen_descriptor.size_in_pixels[0] as f32;
        let height = screen_descriptor.size_in_pixels[1] as f32;

        let scale_x = self.rect.width() / width;
        let scale_y = self.rect.height() / height;
        let offset_x = (self.rect.center().x / width) * 2.0 - 1.0;
        let offset_y = -((self.rect.center().y / height) * 2.0 - 1.0); // Flip Y

        let transform = [scale_x, scale_y, offset_x, offset_y];

        queue.write_buffer(
            &video_resources.uniform_buffer,
            0,
            bytemuck::bytes_of(&transform),
        );

        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let video_resources: &VideoRenderResources = resources.get().unwrap();

        // Select pipeline based on format
        let pipeline = match self.format {
            PixelFormat::Yuv420p => &video_resources.pipeline_yuv420p,
            PixelFormat::Nv12 => &video_resources.pipeline_nv12,
            PixelFormat::Rgb24 | PixelFormat::Rgba | PixelFormat::Bgra => {
                &video_resources.pipeline_rgb
            }
        };

        // Get the video texture
        let texture_guard = self.texture.lock().unwrap();
        if let Some(ref texture) = *texture_guard {
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, &texture.bind_group, &[]);
            render_pass.draw(0..6, 0..1); // Draw fullscreen quad
        }
    }
}
