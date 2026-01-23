//! egui-vid Demo Application
//!
//! A simple demo showcasing the egui-vid video player capabilities.

use eframe::{egui, egui_wgpu};
use egui_vid::{VideoPlayer, VideoPlayerExt};

/// Sample videos for testing
const SAMPLE_VIDEOS: &[(&str, &str)] = &[
    (
        "Big Buck Bunny (MP4)",
        "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/1080/Big_Buck_Bunny_1080_10s_1MB.mp4",
    ),
    (
        "Sintel Trailer (MP4)",
        "https://test-videos.co.uk/vids/sintel/mp4/h264/1080/Sintel_1080_10s_1MB.mp4",
    ),
    (
        "Jellyfish (MKV)",
        "https://test-videos.co.uk/vids/jellyfish/mkv/h264/1080/Jellyfish_1080_10s_1MB.mkv",
    ),
];

fn main() -> eframe::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("egui_vid=debug".parse().unwrap())
                .add_directive("egui_vid_demo=debug".parse().unwrap()),
        )
        .init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1024.0, 768.0])
            .with_title("egui-vid Demo"),
        ..Default::default()
    };

    eframe::run_native(
        "egui-vid Demo",
        options,
        Box::new(|cc| Ok(Box::new(DemoApp::new(cc)))),
    )
}

struct DemoApp {
    /// Current video player
    player: Option<VideoPlayer>,
    /// URL input field
    url_input: String,
    /// Selected sample video index
    selected_sample: usize,
    /// Show metadata panel
    show_metadata: bool,
}

impl DemoApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Start with the first sample video
        let player = cc.wgpu_render_state.as_ref().map(|render_state| {
            VideoPlayer::with_wgpu(SAMPLE_VIDEOS[0].1, render_state)
                .with_autoplay(false)
                .with_loop(true)
                .with_controls(true)
        });

        Self {
            player,
            url_input: SAMPLE_VIDEOS[0].1.to_string(),
            selected_sample: 0,
            show_metadata: true,
        }
    }

    fn load_video(&mut self, url: &str, render_state: &egui_wgpu::RenderState) {
        self.player = Some(
            VideoPlayer::with_wgpu(url, render_state)
                .with_autoplay(true)
                .with_loop(true)
                .with_controls(true),
        );
    }
}

impl eframe::App for DemoApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Top panel with controls
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("egui-vid Demo");
                ui.separator();

                // Sample video selector
                ui.label("Sample:");
                egui::ComboBox::from_id_salt("sample_selector")
                    .selected_text(SAMPLE_VIDEOS[self.selected_sample].0)
                    .show_ui(ui, |ui| {
                        for (i, (name, url)) in SAMPLE_VIDEOS.iter().enumerate() {
                            if ui
                                .selectable_value(&mut self.selected_sample, i, *name)
                                .changed()
                            {
                                self.url_input = url.to_string();
                                if let Some(render_state) = frame.wgpu_render_state() {
                                    self.load_video(url, render_state);
                                }
                            }
                        }
                    });

                ui.separator();
                ui.checkbox(&mut self.show_metadata, "Show Metadata");
            });

            ui.horizontal(|ui| {
                ui.label("URL:");
                let response = ui.text_edit_singleline(&mut self.url_input);
                if ui.button("Load").clicked()
                    || (response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)))
                {
                    if let Some(render_state) = frame.wgpu_render_state() {
                        self.load_video(&self.url_input.clone(), render_state);
                    }
                }
            });
        });

        // Right panel with metadata (optional)
        if self.show_metadata {
            egui::SidePanel::right("metadata_panel")
                .min_width(200.0)
                .show(ctx, |ui| {
                    ui.heading("Video Info");
                    ui.separator();

                    if let Some(ref player) = self.player {
                        if let Some(metadata) = player.metadata() {
                            egui::Grid::new("metadata_grid")
                                .num_columns(2)
                                .spacing([10.0, 4.0])
                                .show(ui, |ui| {
                                    ui.label("Resolution:");
                                    ui.label(format!("{}x{}", metadata.width, metadata.height));
                                    ui.end_row();

                                    ui.label("Codec:");
                                    ui.label(&metadata.codec);
                                    ui.end_row();

                                    ui.label("Frame Rate:");
                                    ui.label(format!("{:.2} fps", metadata.frame_rate));
                                    ui.end_row();

                                    if let Some(duration) = metadata.duration {
                                        ui.label("Duration:");
                                        ui.label(format!("{:.1}s", duration.as_secs_f64()));
                                        ui.end_row();
                                    }
                                });
                        } else {
                            ui.label("Loading metadata...");
                        }

                        ui.separator();
                        ui.heading("Playback");

                        egui::Grid::new("playback_grid")
                            .num_columns(2)
                            .spacing([10.0, 4.0])
                            .show(ui, |ui| {
                                ui.label("State:");
                                ui.label(format!("{:?}", player.state()));
                                ui.end_row();

                                ui.label("Position:");
                                ui.label(format!("{:.1}s", player.position().as_secs_f64()));
                                ui.end_row();

                                ui.label("Playing:");
                                ui.label(if player.is_playing() { "Yes" } else { "No" });
                                ui.end_row();

                                ui.label("Buffering:");
                                ui.label(format!("{}%", player.buffering_percent()));
                                ui.end_row();
                            });
                    } else {
                        ui.label("No video loaded");
                    }
                });
        }

        // Central panel with video
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(ref mut player) = self.player {
                let available_size = ui.available_size();

                // Maintain aspect ratio
                let video_size = if let Some(metadata) = player.metadata() {
                    let aspect = metadata.width as f32 / metadata.height as f32;
                    let max_width = available_size.x;
                    let max_height = available_size.y;

                    if max_width / aspect <= max_height {
                        egui::vec2(max_width, max_width / aspect)
                    } else {
                        egui::vec2(max_height * aspect, max_height)
                    }
                } else {
                    // Default 16:9 while loading
                    let aspect = 16.0 / 9.0;
                    egui::vec2(available_size.x, available_size.x / aspect).min(available_size)
                };

                // Center the video
                ui.centered_and_justified(|ui| {
                    ui.video_player(player, video_size);
                });
            } else {
                ui.centered_and_justified(|ui| {
                    ui.heading("No video loaded");
                    ui.label("Select a sample video or enter a URL above");
                });
            }
        });
    }
}
