# egui-vid

[![CI](https://github.com/egui-vid/egui-vid/actions/workflows/ci.yml/badge.svg)](https://github.com/egui-vid/egui-vid/actions/workflows/ci.yml)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

Hardware-accelerated video playback for [egui](https://github.com/emilk/egui) applications.

## Features

- **Hardware Acceleration**: Platform-native GPU decoding
  - macOS: VideoToolbox
  - Windows: Media Foundation
  - Linux: GStreamer
  - Android: MediaCodec via ExoPlayer
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
├── android_video.rs   # ExoPlayer/MediaCodec JNI bridge (Android)
├── macos_video.rs     # VideoToolbox native decoder (macOS)
├── linux_video.rs     # GStreamer backend (Linux)
├── windows_video.rs   # Media Foundation decoder (Windows)
└── audio.rs           # Audio playback integration
```

## Platform Support

| Platform | Decoder | HW Decode | Rendering | Status |
|----------|---------|-----------|-----------|--------|
| macOS | VideoToolbox | Yes | CPU→GPU copy | Stable |
| Windows | Media Foundation | Yes | CPU→GPU copy | Stable |
| Linux | GStreamer | Yes | CPU→GPU copy | Stable |
| Android | ExoPlayer + MediaCodec | Yes | CPU→GPU copy | Stable |
| Web | - | - | - | Planned |

> **Note**: All platforms currently decode to CPU memory, then upload to GPU via `wgpu::Queue::write_texture()`. True zero-copy (direct GPU surface binding) is planned for future releases.

### Why Native Decoders Over FFmpeg?

| Aspect | Native Decoder | FFmpeg |
|--------|---------------|--------|
| **HW Integration** | Direct API access to VideoToolbox/MediaCodec/etc. | Abstraction layer adds overhead |
| **Memory Efficiency** | Decoder writes to optimal memory locations | Extra copy through libav buffers |
| **Power Consumption** | OS-optimized for battery life (critical on mobile) | Generic implementation, higher power draw |
| **Binary Size** | Uses system libraries (0 MB added) | +15-30 MB for FFmpeg libs |
| **Codec Updates** | Automatic via OS updates | Must rebuild/redeploy |
| **Latency** | Minimal abstraction overhead | Additional buffering in libav pipeline |

Native decoders (VideoToolbox, MediaCodec, Media Foundation, GStreamer) are tightly integrated with each platform's hardware and driver stack. They're maintained by Apple, Google, and Microsoft specifically for optimal performance on their hardware. FFmpeg is an excellent general-purpose solution, but it adds an abstraction layer between your app and the hardware decoder.

**Hardware acceleration is always enabled by default** — software decoding is not a viable option for video playback. Even modest 720p H.264 content at 30fps requires decoding ~25 MB/s of compressed data. CPU-only decoding would consume entire cores and drain batteries in minutes on mobile. HW decoders offload this work to dedicated silicon designed specifically for video, achieving the same decode with a fraction of the power.

## Configuration

### Feature Flags

```toml
[dependencies]
egui-vid = { git = "https://github.com/egui-vid/egui-vid" }
```

| Feature | Description | Default |
|---------|-------------|---------|
| `macos-native-video` | Native VideoToolbox decoder | Yes (macOS) |
| `linux-gstreamer-video` | GStreamer backend | Yes (Linux) |
| `android` | ExoPlayer/MediaCodec backend | Yes (Android) |
| `ffmpeg` | FFmpeg-based decoding (optional) | No |

Platform-native decoders are used by default. FFmpeg is available as an optional fallback.

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
# macOS (uses VideoToolbox - no external dependencies)
cargo build

# Linux (uses GStreamer)
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
cargo build

# Android
cargo ndk -t arm64-v8a build --release
```

### Running Tests

```bash
cargo test
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

GPL-3.0 - see [LICENSE](LICENSE) for details.

## Credits

- Built on [egui](https://github.com/emilk/egui) by Emil Ernerfeldt
- Extracted from [Notedeck](https://github.com/damus-io/notedeck) by Damus
