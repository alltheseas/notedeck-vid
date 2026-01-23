# egui-vid

*It just works.*

[![CI](https://github.com/egui-vid/egui-vid/actions/workflows/ci.yml/badge.svg)](https://github.com/egui-vid/egui-vid/actions/workflows/ci.yml)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

> **Experimental** - This library is under active development. Earlier prototypes have been tested and working in [Notedeck](https://github.com/damus-io/notedeck) video playback on macOS, Linux, and Android. Testing, feedback, issues, and PRs are all welcome! Testers needed!

Hardware-accelerated video playback for [egui](https://github.com/emilk/egui) applications.

## Features

- **Hardware Acceleration**: Platform-native GPU decoding
  - macOS: VideoToolbox
  - Windows: Media Foundation
  - Linux: GStreamer
  - Android: MediaCodec via ExoPlayer
- **Video Sources**: Both streaming URLs and local file playback supported
  - HTTP/HTTPS streaming (progressive download)
  - HLS streaming
  - Local file paths
- **Audio Sync**: Integrated audio playback with video synchronization
- **No-Panic Design**: Uses `parking_lot` for panic-free mutex operations

> **Working Examples**: Streaming video playback has been tested and working in Notedeck prototypes. See the platform PRs for working implementations:
> [macOS](https://github.com/alltheseas/notedeck-vid/pull/8) |
> [Linux](https://github.com/alltheseas/notedeck-vid/pull/10) |
> [Android](https://github.com/alltheseas/notedeck-vid/pull/9) |
> [Windows](https://github.com/alltheseas/notedeck-vid/pull/18)

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
        self.player = Some(VideoPlayer::new("https://example.com/video.mp4"));
    }

    egui::CentralPanel::default().show(ctx, |ui| {
        if let Some(player) = &mut self.player {
            // Render video with controls
            player.show(ui, [640.0, 360.0].into());
        }
    });
}
```

### Custom Controls

```rust
// Create player without built-in controls (builder pattern)
let mut player = VideoPlayer::new("https://example.com/video.mp4")
    .with_controls(false);

let response = player.show(ui, [640.0, 360.0].into());

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
| macOS | VideoToolbox | Yes | CPU→GPU copy | [Alpha](https://github.com/alltheseas/notedeck-vid/pull/8) |
| Linux | GStreamer | Yes | CPU→GPU copy | [Alpha](https://github.com/alltheseas/notedeck-vid/pull/10) |
| Android | ExoPlayer + MediaCodec | Yes | CPU→GPU copy | [Alpha](https://github.com/alltheseas/notedeck-vid/pull/9) |
| Windows | Media Foundation | Yes | CPU→GPU copy | [Untested](https://github.com/alltheseas/notedeck-vid/pull/18) |
| Web | - | - | - | Planned |

> **Note**: All platforms currently decode to CPU memory, then upload to GPU via `wgpu::Queue::write_texture()`. True zero-copy (direct GPU surface binding) is planned for future releases.

### Supported OS Versions & Hardware

#### macOS
- **OS**: macOS 10.13 (High Sierra) or later
- **Hardware**: Any Mac with Metal support
  - Apple Silicon (M1, M2, M3, M4) — native hardware decode
  - Intel Macs with QuickSync (2012 and later)
- **Codecs**: H.264, HEVC, VP9, AV1 (M3+ for AV1)

#### Windows
- **OS**: Windows 10 version 1803 or later (Windows 11 recommended)
- **Hardware**: Any GPU with DirectX 11 support
  - NVIDIA: Kepler (GTX 600) or newer
  - AMD: GCN 1.0 (HD 7000) or newer
  - Intel: Haswell (4th gen) or newer
- **Codecs**: H.264, HEVC (requires [HEVC Extensions](https://apps.microsoft.com/detail/9nmzlz57r3t7)), AV1 (with AV1 Extensions)

#### Linux
- **OS**: Ubuntu 20.04+, Fedora 34+, or any distro with GStreamer 1.16+
- **Hardware**: VA-API compatible GPU with appropriate drivers
  - Intel: Haswell (4th gen) or newer with `intel-media-va-driver`
  - AMD: GCN 1.0 or newer with Mesa VA-API
  - NVIDIA: With proprietary drivers and `gstreamer1.0-vaapi` (or NVDEC via `nvcodec` plugin)
- **Runtime packages**: `gstreamer1.0-plugins-good`, `gstreamer1.0-plugins-bad`, `gstreamer1.0-libav`, `gstreamer1.0-vaapi`

#### Android
- **OS**: Android 5.0 (API 21) or later
- **Tested devices**:
  - Google Pixel 4a, 6, 7, 8
  - Samsung Galaxy S10+, S21, S23
  - OnePlus 8T, 9 Pro
- **Hardware**: Any device with hardware MediaCodec support (virtually all Android 5.0+ devices)
- **Codecs**: H.264, HEVC, VP8, VP9, AV1 (Pixel 8+, recent Samsung flagships)

> **Note**: Hardware acceleration availability varies by device manufacturer and Android version. H.264 is universally supported; HEVC/VP9 support is widespread on 2018+ devices.

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

### Packaging Recommendations

**End users should never need to install separate video dependencies.** Video playback should "just work" when users install your app. Here's the recommended approach per platform:

| Platform | Recommendation |
|----------|---------------|
| **macOS** | VideoToolbox is a system framework — no additional dependencies needed. Ship your `.app` bundle as-is. |
| **Windows** | Media Foundation is built into Windows. No additional dependencies for H.264/AAC. For HEVC, document that users may need the [HEVC Extensions](https://apps.microsoft.com/detail/9nmzlz57r3t7). |
| **Linux** | Declare GStreamer plugins as package dependencies in your `.deb`/`.rpm` metadata. Package managers auto-install them when users install your app. |
| **Android** | MediaCodec is part of Android — no additional dependencies. ExoPlayer is bundled in your APK. |

The goal is a frictionless install experience: users install your app, video works immediately.

### Supported Formats

Format support depends on the native platform decoder. Common formats work across all platforms:

| Format | macOS | Linux | Windows | Android |
|--------|-------|-------|---------|---------|
| **Video** |||||
| H.264/AVC | Yes | Yes | Yes | Yes |
| H.265/HEVC | Yes | Yes | Yes* | Yes |
| VP8 | Yes | Yes | No | Yes |
| VP9 | Yes | Yes | No | Yes |
| AV1 | Yes (M3+) | Yes** | Yes** | Yes (newer devices) |
| **Audio** |||||
| AAC | Yes | Yes | Yes | Yes |
| MP3 | Yes | Yes | Yes | Yes |
| Opus | Yes | Yes | Yes | Yes |
| Vorbis | Yes | Yes | No | Yes |
| FLAC | Yes | Yes | Yes | Yes |
| **Containers** |||||
| MP4/M4V | Yes | Yes | Yes | Yes |
| WebM | Yes | Yes | No | Yes |
| MKV | Yes | Yes | Partial | Yes |
| MOV | Yes | Yes | Yes | Yes |

\* Windows HEVC requires the [HEVC Video Extensions](https://apps.microsoft.com/detail/9nmzlz57r3t7) from Microsoft Store
\*\* AV1 support requires appropriate system codecs/plugins installed

> **Note**: Actual codec availability depends on OS version, installed plugins (Linux GStreamer), and hardware capabilities. When in doubt, H.264 + AAC in MP4 container has the broadest compatibility.

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
let player = VideoPlayer::new(url);

// With wgpu render state for immediate texture creation
let player = VideoPlayer::with_wgpu(url, render_state);
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
    fn new(url: impl Into<String>) -> Self;
    fn with_wgpu(url: impl Into<String>, render_state: &RenderState) -> Self;

    // Playback control
    fn play(&mut self);
    fn pause(&mut self);
    fn toggle_playback(&mut self);
    fn seek(&mut self, position: Duration);
    fn set_muted(&mut self, muted: bool);
    fn set_volume(&mut self, volume: f32);
    fn with_controls(self, show: bool) -> Self;  // Builder pattern

    // State queries
    fn state(&self) -> &VideoState;
    fn position(&self) -> Option<Duration>;
    fn duration(&self) -> Option<Duration>;
    fn is_playing(&self) -> bool;
    fn buffering_percent(&self) -> i32;

    // Rendering
    fn show(&mut self, ui: &mut Ui, size: Vec2) -> VideoPlayerResponse;
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

## Future Improvements

| Improvement | Description | Benefit |
|-------------|-------------|---------|
| **Zero-Copy Rendering** | Direct GPU surface binding instead of CPU→GPU copy | ~2-3x lower latency, reduced memory bandwidth |
| **Lock-Free Frame Queue** | Replace Mutex with `poll_promise` or triple buffering | Eliminates potential UI thread blocking |
| **Web Support** | WebCodecs API for browser-based playback | Cross-platform web applications |
| **HDR Support** | HDR10/Dolby Vision tone mapping | High dynamic range content |
| **Adaptive Streaming** | HLS/DASH quality switching based on bandwidth | Better streaming experience |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

GPL-3.0 - see [LICENSE](LICENSE) for details.

## Credits

- Built on [egui](https://github.com/emilk/egui) by Emil Ernerfeldt
- Extracted from [Notedeck](https://github.com/damus-io/notedeck) by Damus
