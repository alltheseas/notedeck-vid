pub mod action;
#[cfg(target_os = "android")]
pub mod android_video;
pub mod audio;
#[cfg(feature = "ffmpeg")]
pub mod audio_decoder;
pub mod blur;
pub mod frame_queue;
pub mod gif;
pub mod images;
pub mod imeta;
pub mod latest;
pub mod network;
pub mod renderable;
pub mod static_imgs;
pub mod video;
pub mod video_controls;
#[cfg(feature = "ffmpeg")]
pub mod video_decoder;
pub mod video_player;
pub mod video_texture;

pub use action::{MediaAction, MediaInfo, ViewMediaInfo};
pub use audio::{AudioConfig, AudioHandle, AudioPlayer, AudioSamples, AudioState, AudioSync};
pub use blur::{
    update_imeta_blurhashes, BlurCache, ImageMetadata, ObfuscationType, PixelDimensions,
    PointDimensions,
};
use egui::{ColorImage, TextureHandle};
pub use images::ImageType;
pub use latest::{
    MediaRenderState, NoLoadingLatestTex, TrustedMediaLatestTex, UntrustedMediaLatestTex,
};
pub use renderable::RenderableMedia;
pub use video::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata, VideoPlayerHandle, VideoState,
};
pub use video_controls::{VideoControls, VideoControlsConfig, VideoControlsResponse};
#[cfg(feature = "ffmpeg")]
pub use video_decoder::{FfmpegDecoder, FfmpegDecoderBuilder, HwAccelConfig};
pub use video_player::{VideoPlayer, VideoPlayerExt, VideoPlayerResponse};

#[cfg(target_os = "android")]
pub use android_video::AndroidVideoDecoder;

#[derive(Copy, Clone, Debug)]
pub enum AnimationMode {
    /// Only render when scrolling, network activity, etc
    Reactive,

    /// Continuous with an optional target fps
    Continuous { fps: Option<f32> },

    /// Disable animation
    NoAnimation,
}

impl AnimationMode {
    pub fn can_animate(&self) -> bool {
        !matches!(self, Self::NoAnimation)
    }
}

// max size wgpu can handle without panicing
pub const MAX_SIZE_WGPU: usize = 8192;

pub fn load_texture_checked(
    ctx: &egui::Context,
    name: impl Into<String>,
    image: ColorImage,
    options: egui::TextureOptions,
) -> TextureHandle {
    let size = image.size;

    if size[0] > MAX_SIZE_WGPU || size[1] > MAX_SIZE_WGPU {
        panic!("The image MUST be less than or equal to {MAX_SIZE_WGPU} pixels in each direction");
    }

    #[allow(clippy::disallowed_methods, reason = "centralized safe wrapper")]
    ctx.load_texture(name, image, options)
}
