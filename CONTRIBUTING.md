# Contributing to egui-vid

Thank you for your interest in contributing to egui-vid! This document provides guidelines and information for contributors.

## Code of Conduct

Be respectful and constructive. We're all here to build great software.

## Getting Started

### Prerequisites

- Rust 1.70+ (stable)
- Platform-specific dependencies (see below)

### Setting Up Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/egui-vid/egui-vid.git
   cd egui-vid
   ```

2. **Install platform dependencies:**

   **macOS** - No additional dependencies (uses system AVFoundation/VideoToolbox)

   **Linux** - GStreamer for native video:
   ```bash
   # Ubuntu/Debian
   sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-good

   # Fedora
   sudo dnf install gstreamer1-devel gstreamer1-plugins-base-devel gstreamer1-plugins-good
   ```

   **Windows** - No additional dependencies (uses system Media Foundation)

   **Optional: FFmpeg fallback** (if native decoders are unavailable):
   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu/Debian
   sudo apt install libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
   ```

3. **Build and test:**
   ```bash
   # Platform-specific native decoders (recommended)
   cargo build --features macos-native-video    # macOS - VideoToolbox
   cargo build --features linux-gstreamer-video # Linux - GStreamer
   cargo build --features windows-native-video  # Windows - Media Foundation

   # FFmpeg fallback (optional, requires FFmpeg installed)
   cargo build --features ffmpeg

   # Run tests
   cargo test
   ```

   > **Note**: No features are enabled by default. You must explicitly enable a platform feature.

## Development Guidelines

### Code Style

We follow the standard Rust style guidelines with some project-specific rules:

1. **No `unwrap()` or `expect()` in production code**
   - Use `let ... else { return }` guards
   - Use proper error handling with `Result`
   - Test code may use `panic!()` with descriptive messages

   ```rust
   // Good
   let Some(frame) = queue.pop() else {
       return None;
   };

   // Bad
   let frame = queue.pop().unwrap();
   ```

2. **Use `parking_lot` instead of `std::sync`**
   - `parking_lot::Mutex` instead of `std::sync::Mutex`
   - `parking_lot::Condvar` instead of `std::sync::Condvar`
   - This eliminates `.lock().unwrap()` patterns

3. **Avoid over-engineering**
   - Only add features that are explicitly requested
   - Keep solutions simple and focused
   - Don't add abstractions for one-time operations

4. **Platform-specific code**
   - Use `#[cfg(target_os = "...")]` attributes
   - Keep platform code in separate files (e.g., `macos_video.rs`, `android_video.rs`)
   - Provide fallback implementations where possible

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(android): add ExoPlayer JNI bridge for video decoding

fix(frame_queue): prevent deadlock in Condvar wait

docs: update README with hardware acceleration info
```

### Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Run tests:**
   ```bash
   cargo test
   cargo clippy -- -D warnings
   cargo fmt --check
   ```

4. **Push and create PR:**
   ```bash
   git push origin feat/your-feature-name
   ```

5. **PR Description should include:**
   - Summary of changes
   - Related issue numbers
   - Test plan
   - Screenshots/videos for UI changes

### Testing

- Write tests for new functionality
- Ensure existing tests pass
- Test on multiple platforms when possible

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_frame_queue

# Run tests with output
cargo test -- --nocapture
```

## Architecture Overview

### Module Structure

```
crates/egui-vid/src/media/
├── video.rs             # Core types and traits
├── video_player.rs      # egui widget implementation
├── video_texture.rs     # GPU texture and shader management
├── frame_queue.rs       # Frame buffering and decode thread
├── triple_buffer.rs     # Lock-free triple buffering
├── network.rs           # HTTP streaming
├── audio.rs             # Audio playback
│
├── macos_video.rs       # macOS AVFoundation/VideoToolbox (native)
├── linux_video_gst.rs   # Linux GStreamer/VA-API (native)
├── windows_video.rs     # Windows Media Foundation (native)
├── android_video.rs     # Android MediaCodec (native)
│
└── video_decoder.rs     # FFmpeg decoder (optional fallback)
```

### Key Abstractions

1. **`VideoDecoderBackend` trait** - Platform-agnostic decoder interface
2. **`FrameQueue`** - Thread-safe frame buffer
3. **`VideoPlayer`** - Main egui widget
4. **`VideoTexture`** - GPU texture management

### Threading Model

- **UI Thread**: Renders frames, handles user input
- **Decode Thread**: Decodes video frames, fills queue
- **Audio Thread**: Plays audio in sync with video

Communication uses `parking_lot` primitives and channels.

## Platform-Specific Development

### macOS

VideoToolbox integration requires:
- macOS 10.13+
- Xcode command line tools

### Windows

D3D11VA/DXVA2 integration requires:
- Windows 10+
- Visual Studio Build Tools

### Linux

VAAPI integration requires:
- Mesa with VA-API support
- `libva-dev` package

### Android

ExoPlayer integration requires:
- Android NDK
- JNI knowledge
- Kotlin/Java for Android-side code

## Reporting Issues

### Bug Reports

Include:
1. Platform and version
2. Steps to reproduce
3. Expected vs actual behavior
4. Relevant logs or error messages
5. Video format/codec information

### Feature Requests

Include:
1. Use case description
2. Proposed API (if applicable)
3. Platform considerations

## License

By contributing, you agree that your contributions will be licensed under GPL-3.0.

## Questions?

Open a discussion or issue on GitHub.
