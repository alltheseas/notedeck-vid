# egui-vid

[![CI](https://github.com/egui-vid/egui-vid/actions/workflows/ci.yml/badge.svg)](https://github.com/egui-vid/egui-vid/actions/workflows/ci.yml)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

Hardware-accelerated video playback for [egui](https://github.com/emilk/egui) applications.

## Features

- **Hardware Acceleration**: Platform-native GPU decoding
  - macOS: VideoToolbox
  - Windows: D3D11VA / DXVA2
  - Linux: VAAPI
  - Android: MediaCodec via ExoPlayer
- **Zero-Copy Rendering**: Direct GPU texture upload where supported
- **Streaming Support**: HLS and progressive download
- **Audio Sync**: Integrated audio playback with video synchronization
- **No-Panic Design**: Uses `parking_lot` for panic-free mutex operations

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
egui-vid = { git = "https://github.com/egui-vid/egui-vid" }
```

### Basic Usage

```rust
use egui_vid::{VideoPlayer, VideoPlayerExt};

// In your egui app:
fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
    // Create player (once)
    if self.player.is_none() {
        self.player = Some(VideoPlayer::new(
            "https://example.com/video.mp4",
            frame.wgpu_render_state().unwrap(),
        ));
    }

    egui::CentralPanel::default().show(ctx, |ui| {
        if let Some(player) = &mut self.player {
            // Render video with controls
            player.ui(ui, [640.0, 360.0].into());
        }
    });
}
```

### Custom Controls

```rust
// Render video without built-in controls
let response = player.ui_no_controls(ui, [640.0, 360.0].into());

// Build your own controls
if ui.button("Play/Pause").clicked() {
    player.toggle_playback();
}

if let Some(duration) = player.duration() {
    let position = player.position().unwrap_or_default();
    ui.label(format!("{:.1}s / {:.1}s", position.as_secs_f32(), duration.as_secs_f32()));
}
```

## Architecture

```
egui-vid
├── video.rs           # Core types: VideoState, VideoFrame, VideoMetadata
├── video_player.rs    # Main VideoPlayer widget for egui
├── video_texture.rs   # GPU texture management and YUV→RGB shaders
├── frame_queue.rs     # Thread-safe frame buffer with decode thread
├── video_decoder.rs   # FFmpeg-based decoder (desktop)
├── android_video.rs   # ExoPlayer JNI bridge (Android)
├── macos_video.rs     # VideoToolbox native decoder (macOS)
├── windows_video.rs   # Media Foundation decoder (Windows)
└── audio.rs           # Audio playback integration
```

## Platform Support

| Platform | Decoder | Hardware Accel | Status |
|----------|---------|----------------|--------|
| macOS | FFmpeg + VideoToolbox | Yes | Stable |
| Windows | FFmpeg + D3D11VA | Yes | Stable |
| Linux | FFmpeg + VAAPI | Yes | Stable |
| Android | ExoPlayer + MediaCodec | Yes | Stable |
| Web | - | - | Planned |

## Configuration

### Feature Flags

```toml
[dependencies]
egui-vid = { git = "https://github.com/egui-vid/egui-vid", features = ["ffmpeg"] }
```

| Feature | Description | Default |
|---------|-------------|---------|
| `ffmpeg` | FFmpeg-based decoding (desktop) | Yes |
| `macos-native-video` | Native VideoToolbox (no FFmpeg) | No |
| `linux-gstreamer-video` | GStreamer backend for Linux | No |

### Hardware Acceleration Config

```rust
use egui_vid::{FfmpegDecoder, HwAccelConfig};

// Force software decoding
let decoder = FfmpegDecoder::new_with_config(
    "video.mp4",
    HwAccelConfig::software_only()
);

// Prefer specific format
let config = HwAccelConfig {
    preferred_types: vec![HwAccelType::VideoToolbox],
    fallback_to_software: true,
    ..Default::default()
};
```

## Performance

The frame queue uses a producer-consumer pattern with configurable buffer size:

```rust
// Default: 5 frames buffered
let player = VideoPlayer::new(url, render_state);

// Custom buffer size for low-latency
let player = VideoPlayer::with_buffer_size(url, render_state, 2);
```

### Threading Model

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   UI Thread     │     │  Decode Thread  │     │  Audio Thread   │
│                 │     │                 │     │                 │
│  VideoPlayer    │◄────│  FrameQueue     │     │  AudioDecoder   │
│  renders frame  │     │  buffers frames │     │  syncs playback │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## API Reference

### Core Types

- **`VideoPlayer`** - Main widget for rendering video in egui
- **`VideoState`** - Playback state (Loading, Ready, Playing, Paused, Buffering, Error, Ended)
- **`VideoFrame`** - Decoded frame with PTS timestamp
- **`VideoMetadata`** - Video properties (dimensions, duration, frame rate, codec)
- **`VideoError`** - Error types for decoding failures

### VideoPlayer Methods

```rust
impl VideoPlayer {
    // Construction
    fn new(url: &str, render_state: &RenderState) -> Self;
    fn with_buffer_size(url: &str, render_state: &RenderState, buffer_size: usize) -> Self;

    // Playback control
    fn play(&mut self);
    fn pause(&mut self);
    fn toggle_playback(&mut self);
    fn seek(&mut self, position: Duration);
    fn set_muted(&mut self, muted: bool);
    fn set_volume(&mut self, volume: f32);

    // State queries
    fn state(&self) -> &VideoState;
    fn position(&self) -> Option<Duration>;
    fn duration(&self) -> Option<Duration>;
    fn is_playing(&self) -> bool;
    fn buffering_percent(&self) -> i32;

    // Rendering
    fn ui(&mut self, ui: &mut Ui, size: Vec2) -> Response;
    fn ui_no_controls(&mut self, ui: &mut Ui, size: Vec2) -> Response;
}
```

## Development

### Building

```bash
# Desktop (requires FFmpeg)
cargo build --features ffmpeg

# macOS native (no FFmpeg dependency)
cargo build --features macos-native-video

# Android
cargo ndk -t arm64-v8a build --release
```

### Running Tests

```bash
cargo test
```

### FFmpeg Installation

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt install libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
```

**Windows:**
Download from [FFmpeg builds](https://github.com/BtbN/FFmpeg-Builds/releases) and add to PATH.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

GPL-3.0 - see [LICENSE](LICENSE) for details.

## Credits

- Built on [egui](https://github.com/emilk/egui) by Emil Ernerfeldt
- FFmpeg integration via [rust-ffmpeg](https://github.com/zmwangx/rust-ffmpeg)
- Extracted from [Notedeck](https://github.com/damus-io/notedeck) by Damus
