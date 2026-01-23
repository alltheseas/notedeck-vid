# egui-vid

Cross-platform hardware-accelerated video playback for [egui](https://github.com/emilk/egui).

## Features

- **Hardware Acceleration**: Native decoders with GPU-accelerated decoding on all platforms
- **Cross-Platform**: macOS, Linux, Windows, and Android support
- **Streaming**: HTTP/HTTPS video streaming with buffering
- **Controls**: Built-in play/pause, seek, volume, and fullscreen controls
- **Lock-Free**: Triple-buffered frame passing for smooth playback
- **No-Panic**: Uses `parking_lot` for panic-free mutex operations

## Platform Support

| Platform | Decoder | Hardware Acceleration | Feature Flag |
|----------|---------|----------------------|--------------|
| macOS | AVFoundation | VideoToolbox | `macos-native-video` |
| Linux | GStreamer | VA-API, NVDEC | `linux-gstreamer-video` |
| Windows | FFmpeg | DXVA2, D3D11VA | `ffmpeg` |
| Android | MediaCodec | Hardware codecs | (automatic) |

## Quick Start

```toml
# Cargo.toml
[dependencies]
egui-vid = { version = "0.1", features = ["macos-native-video"] }  # macOS
# egui-vid = { version = "0.1", features = ["linux-gstreamer-video"] }  # Linux
# egui-vid = { version = "0.1", features = ["ffmpeg"] }  # Windows/fallback
```

```rust
use egui_vid::{VideoPlayer, VideoPlayerExt};

// Create a video player
let mut player = VideoPlayer::new("https://example.com/video.mp4");

// In your egui update loop:
player.show(ui, available_size);
```

## Installation

### macOS

No additional dependencies required. The native AVFoundation decoder uses VideoToolbox for hardware acceleration.

```bash
cargo build --features macos-native-video
```

### Linux

Install GStreamer development libraries:

```bash
# Ubuntu/Debian
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

# Fedora
sudo dnf install gstreamer1-devel gstreamer1-plugins-base-devel
```

```bash
cargo build --features linux-gstreamer-video
```

### Windows

Install FFmpeg and set `FFMPEG_DIR` environment variable:

```bash
cargo build --features ffmpeg
```

## Architecture

```
+-----------------------------------------------------+
|                    VideoPlayer                       |
|  +-----------+  +-----------+  +-----------+        |
|  |  Decoder  |--|FrameQueue |--|  Renderer |        |
|  | (Native/  |  | (Triple   |  |  (wgpu)   |        |
|  |  FFmpeg)  |  | Buffered) |  |           |        |
|  +-----------+  +-----------+  +-----------+        |
|                                                      |
|  +-----------+  +-----------+                       |
|  |   Audio   |  | Controls  |                       |
|  |  (rodio)  |  |  (egui)   |                       |
|  +-----------+  +-----------+                       |
+-----------------------------------------------------+
```

## Known Issues

### macOS: objc2 Version Coexistence

The native macOS decoder uses `objc2 0.6.x` for AVFoundation bindings, while `winit` (used by egui) uses `objc2 0.5.x`.

**This is a known working configuration.** Both versions coexist safely because they bind to different Objective-C classes:

- `objc2 0.5.x`: winit's window management classes (NSWindow, NSView, etc.)
- `objc2 0.6.x`: AVFoundation media classes (AVPlayer, AVAsset, etc.)

If you encounter issues:
1. Ensure you're using the [damus egui/eframe fork](https://github.com/damus-io/egui) which has been tested with this setup
2. Check that your `Cargo.toml` patches egui correctly (see workspace Cargo.toml)

### FFmpeg Fallback

If native decoders aren't available, FFmpeg can be used as a fallback on all platforms. FFmpeg still uses hardware acceleration where available:

- macOS: VideoToolbox via FFmpeg
- Linux: VA-API/VDPAU via FFmpeg
- Windows: DXVA2/D3D11VA via FFmpeg

## Decoder Philosophy

We prioritize native platform decoders over FFmpeg for several reasons:

- **Better hardware integration**: Native APIs have first-class access to GPU decoders
- **Lower memory footprint**: No need to bundle FFmpeg libraries
- **Reduced power consumption**: Hardware decoders are more efficient
- **Smaller binary size**: Native frameworks are already on the system
- **Automatic codec updates**: System updates bring new codec support

## License

GPL-3.0 - see license information in individual crates.

## Contributing

Contributions welcome! Please open an issue or PR on [GitHub](https://github.com/egui-vid/egui-vid).
