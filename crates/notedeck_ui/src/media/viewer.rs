use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use bitflags::bitflags;
use egui::{emath::TSTransform, pos2, Color32, Rangef, Rect};
use notedeck::media::{AnimationMode, MediaInfo, ViewMediaInfo, VideoPlayer};
use notedeck::{ImageType, Images, MediaCacheType, MediaJobSender};

use crate::note::media::InlineVideoPlayers;

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct MediaViewerFlags: u64 {
        /// Open the media viewer fullscreen
        const Fullscreen = 1 << 0;

        /// Enable a transition animation
        const Transition = 1 << 1;

        /// Are we open or closed?
        const Open = 1 << 2;
    }
}

/// State used in the MediaViewer ui widget.
pub struct MediaViewerState {
    /// When
    pub media_info: ViewMediaInfo,
    pub scene_rect: Option<Rect>,
    pub flags: MediaViewerFlags,
    pub anim_id: egui::Id,
    /// Video players by URL - lazily initialized when needed
    pub video_players: HashMap<String, VideoPlayer>,
}

impl Default for MediaViewerState {
    fn default() -> Self {
        Self {
            anim_id: egui::Id::new("notedeck-fullscreen-media-viewer"),
            media_info: Default::default(),
            scene_rect: None,
            flags: MediaViewerFlags::Transition | MediaViewerFlags::Fullscreen,
            video_players: HashMap::new(),
        }
    }
}

impl MediaViewerState {
    pub fn new(anim_id: egui::Id) -> Self {
        Self {
            anim_id,
            ..Default::default()
        }
    }

    /// How much is our media viewer open
    pub fn open_amount(&self, ui: &mut egui::Ui) -> f32 {
        ui.ctx().animate_bool_with_time_and_easing(
            self.anim_id,
            self.flags.contains(MediaViewerFlags::Open),
            0.3,
            egui::emath::easing::cubic_out,
        )
    }

    /// Should we show the control even if we're closed?
    /// Needed for transition animation
    pub fn should_show(&self, ui: &mut egui::Ui) -> bool {
        if self.flags.contains(MediaViewerFlags::Open) {
            return true;
        }

        // we are closing
        self.open_amount(ui) > 0.0
    }
}

/// A panning, scrolling, optionally fullscreen, and tiling media viewer
pub struct MediaViewer<'a> {
    state: &'a mut MediaViewerState,
}

impl<'a> MediaViewer<'a> {
    pub fn new(state: &'a mut MediaViewerState) -> Self {
        Self { state }
    }

    /// Is this
    pub fn fullscreen(self, enable: bool) -> Self {
        self.state.flags.set(MediaViewerFlags::Fullscreen, enable);
        self
    }

    /// Enable open transition animation
    pub fn transition(self, enable: bool) -> Self {
        self.state.flags.set(MediaViewerFlags::Transition, enable);
        self
    }

    pub fn ui(
        &mut self,
        images: &mut Images,
        jobs: &MediaJobSender,
        ui: &mut egui::Ui,
    ) -> egui::Response {
        if self.state.flags.contains(MediaViewerFlags::Fullscreen) {
            egui::Window::new("Media Viewer")
                .title_bar(false)
                .fixed_size(ui.ctx().screen_rect().size())
                .fixed_pos(ui.ctx().screen_rect().min)
                .frame(egui::Frame::NONE)
                .show(ui.ctx(), |ui| self.ui_content(images, jobs, ui))
                .unwrap() // SAFETY: we are always open
                .inner
                .unwrap()
        } else {
            self.ui_content(images, jobs, ui)
        }
    }

    fn ui_content(
        &mut self,
        images: &mut Images,
        jobs: &MediaJobSender,
        ui: &mut egui::Ui,
    ) -> egui::Response {
        let avail_rect = ui.available_rect_before_wrap();

        let is_open = self.state.flags.contains(MediaViewerFlags::Open);
        let can_transition = self.state.flags.contains(MediaViewerFlags::Transition);
        let open_amount = self.state.open_amount(ui);

        // Draw background
        ui.painter().rect_filled(
            avail_rect,
            0.0,
            egui::Color32::from_black_alpha((200.0 * open_amount) as u8),
        );

        // Check if we're showing a video - render without Scene to preserve controls
        let clicked_media = self.state.media_info.clicked_media();
        if clicked_media.media_type == MediaCacheType::Video {
            // Render video directly without Scene (zoom/pan not needed for video)
            let video_players = &mut self.state.video_players;
            let exit_fullscreen = Self::render_video_tile(&clicked_media.url, video_players, ui, open_amount);

            // Exit fullscreen if button was clicked
            if exit_fullscreen {
                self.state.flags.remove(MediaViewerFlags::Open);
            }

            let (_, response) = ui.allocate_exact_size(avail_rect.size(), egui::Sense::click());
            return response;
        }

        // For images, use Scene for zoom/pan
        let scene_rect = if let Some(scene_rect) = self.state.scene_rect {
            scene_rect
        } else {
            self.state.scene_rect = Some(avail_rect);
            avail_rect
        };

        let zoom_range: egui::Rangef = (0.0..=10.0).into();

        let transitioning = if !can_transition {
            false
        } else if is_open {
            open_amount < 1.0
        } else {
            open_amount > 0.0
        };

        let mut trans_rect = if transitioning {
            let src_pos = &clicked_media.original_position;
            let in_scene_pos = Self::first_image_rect(ui, &clicked_media, images, jobs);
            transition_scene_rect(
                &avail_rect,
                &zoom_range,
                &in_scene_pos,
                src_pos,
                open_amount,
            )
        } else {
            scene_rect
        };

        let scene = egui::Scene::new().zoom_range(zoom_range);

        // Clone media info to avoid borrow conflicts with video_players
        let medias = self.state.media_info.medias.clone();
        let video_players = &mut self.state.video_players;

        let resp = scene.show(ui, &mut trans_rect, |ui| {
            Self::render_media_tiles(&medias, video_players, images, jobs, ui, open_amount);
        });

        self.state.scene_rect = Some(trans_rect);

        resp.response
    }

    /// The rect of the first image to be placed.
    /// This is mainly used for the transition animation
    ///
    /// TODO(jb55): replace this with a "placed" variant once
    /// we have image layouts
    fn first_image_rect(
        ui: &mut egui::Ui,
        media: &MediaInfo,
        images: &mut Images,
        jobs: &MediaJobSender,
    ) -> Rect {
        // For videos, use available space with 16:9 aspect ratio
        if media.media_type == MediaCacheType::Video {
            let avail = ui.available_rect_before_wrap();
            let aspect_ratio = 16.0 / 9.0;
            let width = avail.width();
            let height = width / aspect_ratio;
            return Rect::from_min_size(avail.min, egui::vec2(width, height));
        }

        // fetch image texture
        let Some(texture) = images.latest_texture(
            jobs,
            ui,
            &media.url,
            ImageType::Content(None),
            AnimationMode::NoAnimation,
        ) else {
            tracing::error!("could not get latest texture in first_image_rect");
            return Rect::ZERO;
        };

        // the area the next image will be put in.
        let mut img_rect = ui.available_rect_before_wrap();

        let size = texture.size_vec2();
        img_rect.set_height(size.y);
        img_rect.set_width(size.x);
        img_rect
    }

    ///
    /// Tile a scene with media (images and videos).
    ///
    /// TODO(jb55): Let's improve image tiling over time, spiraling outward. We
    /// should have a way to click "next" and have the scene smoothly transition and
    /// focus on the next image
    fn render_media_tiles(
        infos: &[MediaInfo],
        video_players: &mut HashMap<String, VideoPlayer>,
        images: &mut Images,
        jobs: &MediaJobSender,
        ui: &mut egui::Ui,
        open_amount: f32,
    ) {
        for info in infos {
            let url = &info.url;

            // Handle videos separately
            if info.media_type == MediaCacheType::Video {
                let _ = Self::render_video_tile(url, video_players, ui, open_amount);
                continue;
            }

            // fetch image texture for images/gifs

            // we want to continually redraw things in the gallery
            let Some(texture) = images.latest_texture(
                jobs,
                ui,
                url,
                ImageType::Content(None),
                AnimationMode::Continuous { fps: None }, // media viewer has continuous rendering
            ) else {
                continue;
            };

            // the area the next image will be put in.
            let mut img_rect = ui.available_rect_before_wrap();
            /*
            if !ui.is_rect_visible(img_rect) {
                // just stop rendering images if we're going out of the scene
                // basic culling when we have lots of images
                break;
            }
            */

            {
                let size = texture.size_vec2();
                img_rect.set_height(size.y);
                img_rect.set_width(size.x);
                let uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));

                // image actions
                //let response = ui.interact(render_rect, carousel_id.with("img"), Sense::click());

                /*
                if response.clicked() {
                } else if background_response.clicked() {
                }
                */

                // Paint image
                ui.painter().image(
                    texture.id(),
                    img_rect,
                    uv,
                    Color32::from_white_alpha((open_amount * 255.0) as u8),
                );

                ui.advance_cursor_after_rect(img_rect);
            }
        }
    }

    /// Render a video tile in the media viewer.
    /// Uses shared InlineVideoPlayers storage so fullscreen continues from inline position.
    /// Returns true if fullscreen button was clicked (to exit fullscreen).
    fn render_video_tile(
        url: &str,
        _video_players: &mut HashMap<String, VideoPlayer>,
        ui: &mut egui::Ui,
        _open_amount: f32,
    ) -> bool {
        // Get shared video players from egui memory (same storage as inline videos)
        let players_id = egui::Id::new("inline_video_players");
        let players = ui.ctx().memory_mut(|mem| {
            mem.data.get_temp_mut_or_insert_with::<Arc<Mutex<InlineVideoPlayers>>>(
                players_id,
                || Arc::new(Mutex::new(InlineVideoPlayers::default()))
            ).clone()
        });

        let mut players_guard = players.lock().unwrap();
        let player = players_guard.players.entry(url.to_string()).or_insert_with(|| {
            VideoPlayer::new(url)
                .with_autoplay(true)
                .with_loop(true)
                .with_controls(true)
        });

        // Calculate video size - use full available space
        let avail = ui.available_rect_before_wrap();
        let max_width = avail.width();
        let max_height = avail.height();

        // Use video metadata for aspect ratio if available, otherwise 16:9
        let aspect_ratio = player
            .metadata()
            .map(|m| m.width as f32 / m.height as f32)
            .unwrap_or(16.0 / 9.0);

        // Fit video to available space while maintaining aspect ratio
        let (width, height) = if max_width / aspect_ratio <= max_height {
            (max_width, max_width / aspect_ratio)
        } else {
            (max_height * aspect_ratio, max_height)
        };

        let size = egui::vec2(width, height);

        // Center the video in the available space using a centered layout
        let mut exit_fullscreen = false;
        ui.vertical_centered(|ui| {
            ui.add_space((avail.height() - height) / 2.0);
            let response = player.show(ui, size);
            // Fullscreen button in fullscreen mode = exit fullscreen
            if response.toggle_fullscreen {
                exit_fullscreen = true;
            }
        });
        exit_fullscreen
    }
}

/// Helper: lerp a TSTransform (uniform scale + translation)
fn lerp_ts(a: TSTransform, b: TSTransform, t: f32) -> TSTransform {
    let s = egui::lerp(a.scaling..=b.scaling, t);
    let p = a.translation + (b.translation - a.translation) * t;
    TSTransform {
        scaling: s,
        translation: p,
    }
}

/// Calculate the open/close amount and transition rect
pub fn transition_scene_rect(
    outer_rect: &Rect,
    zoom_range: &Rangef,
    image_rect_in_scene: &Rect, // e.g. Rect::from_min_size(Pos2::ZERO, image_size)
    timeline_global_rect: &Rect, // saved from timeline Response.rect
    open_amt: f32,              // stable ID per media item
) -> Rect {
    // Compute the two endpoints:
    let from = fit_to_rect_in_scene(timeline_global_rect, image_rect_in_scene, zoom_range);
    let to = fit_to_rect_in_scene(outer_rect, image_rect_in_scene, zoom_range);

    // Interpolate transform and convert to scene_rect expected by Scene::show:
    let lerped = lerp_ts(from, to, open_amt);

    lerped.inverse() * (*outer_rect)
}

/// Creates a transformation that fits a given scene rectangle into the available screen size.
///
/// The resulting visual scene bounds can be larger, due to letterboxing.
///
/// Returns the transformation from `scene` to `global` coordinates.
fn fit_to_rect_in_scene(
    rect_in_global: &Rect,
    rect_in_scene: &Rect,
    zoom_range: &Rangef,
) -> TSTransform {
    // Compute the scale factor to fit the bounding rectangle into the available screen size:
    let scale = rect_in_global.size() / rect_in_scene.size();

    // Use the smaller of the two scales to ensure the whole rectangle fits on the screen:
    let scale = scale.min_elem();

    // Clamp scale to what is allowed
    let scale = zoom_range.clamp(scale);

    // Compute the translation to center the bounding rect in the screen:
    let center_in_global = rect_in_global.center().to_vec2();
    let center_scene = rect_in_scene.center().to_vec2();

    // Set the transformation to scale and then translate to center.
    TSTransform::from_translation(center_in_global - scale * center_scene)
        * TSTransform::from_scaling(scale)
}
