//! Audio playback for video player.
//!
//! This module handles audio decoding and playback for video files.
//! It provides A/V synchronization, volume control, and mute toggle.
//!
//! # Architecture
//!
//! The audio system consists of:
//! - `AudioDecoder`: Extracts audio frames from video stream via FFmpeg
//! - `AudioPlayer`: Handles audio output via rodio
//! - `AudioSync`: Synchronizes audio playback with video presentation
//!
//! Audio serves as the master clock for A/V sync - video frames are
//! presented relative to the audio playback position.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Audio playback state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioState {
    /// Audio is not initialized
    Uninitialized,
    /// Audio is playing
    Playing,
    /// Audio is paused
    Paused,
    /// Audio playback error
    Error,
}

/// Configuration for audio playback.
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u16,
    /// Buffer size in samples
    pub buffer_size: u32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            channels: 2,
            buffer_size: 1024,
        }
    }
}

/// Audio player handle for volume and mute control.
///
/// This is a lightweight handle that can be cloned and shared
/// between the video player and UI controls.
#[derive(Clone)]
pub struct AudioHandle {
    inner: Arc<AudioHandleInner>,
}

struct AudioHandleInner {
    /// Volume level (0-100)
    volume: AtomicU32,
    /// Whether audio is muted
    muted: AtomicBool,
    /// Current playback position in microseconds (u64 to handle videos >71 minutes)
    position_us: AtomicU64,
    /// Whether audio is available for this video
    available: AtomicBool,
}

impl AudioHandle {
    /// Creates a new audio handle.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(AudioHandleInner {
                volume: AtomicU32::new(100),
                muted: AtomicBool::new(false),
                position_us: AtomicU64::new(0),
                available: AtomicBool::new(false),
            }),
        }
    }

    /// Returns the current volume (0-100).
    pub fn volume(&self) -> u32 {
        self.inner.volume.load(Ordering::Relaxed)
    }

    /// Sets the volume (0-100).
    pub fn set_volume(&self, volume: u32) {
        self.inner.volume.store(volume.min(100), Ordering::Relaxed);
    }

    /// Returns whether audio is muted.
    pub fn is_muted(&self) -> bool {
        self.inner.muted.load(Ordering::Relaxed)
    }

    /// Sets the mute state.
    pub fn set_muted(&self, muted: bool) {
        self.inner.muted.store(muted, Ordering::Relaxed);
    }

    /// Toggles the mute state.
    pub fn toggle_mute(&self) {
        // Use fetch_xor for atomic toggle to avoid TOCTOU race condition
        self.inner.muted.fetch_xor(true, Ordering::Relaxed);
    }

    /// Returns the effective volume (0.0-1.0) accounting for mute.
    pub fn effective_volume(&self) -> f32 {
        if self.is_muted() {
            0.0
        } else {
            self.volume() as f32 / 100.0
        }
    }

    /// Returns the current playback position.
    pub fn position(&self) -> Duration {
        Duration::from_micros(self.inner.position_us.load(Ordering::Relaxed))
    }

    /// Updates the playback position (internal use).
    pub fn set_position(&self, position: Duration) {
        let us = position.as_micros() as u64;
        self.inner.position_us.store(us, Ordering::Relaxed);
    }

    /// Returns whether audio is available for this video.
    pub fn is_available(&self) -> bool {
        self.inner.available.load(Ordering::Relaxed)
    }

    /// Sets whether audio is available (internal use).
    pub fn set_available(&self, available: bool) {
        self.inner.available.store(available, Ordering::Relaxed);
    }
}

impl Default for AudioHandle {
    fn default() -> Self {
        Self::new()
    }
}

/// Audio synchronization helper.
///
/// Uses audio playback position as the master clock for video frame timing.
/// When audio is not available, falls back to wall-clock time.
pub struct AudioSync {
    /// Audio handle for getting playback position
    audio: AudioHandle,
    /// Whether to use audio as master clock
    use_audio_clock: bool,
    /// Fallback start time when audio is not available
    #[allow(dead_code)]
    fallback_start: std::time::Instant,
}

impl AudioSync {
    /// Creates a new audio sync helper.
    pub fn new(audio: AudioHandle) -> Self {
        Self {
            audio,
            use_audio_clock: true,
            fallback_start: std::time::Instant::now(),
        }
    }

    /// Returns the current playback position for frame timing.
    pub fn position(&self) -> Duration {
        if self.use_audio_clock && self.audio.is_available() {
            self.audio.position()
        } else {
            // Fallback to scheduler's time
            Duration::ZERO
        }
    }

    /// Sets whether to use audio as the master clock.
    pub fn set_use_audio_clock(&mut self, use_audio: bool) {
        self.use_audio_clock = use_audio;
    }

    /// Returns whether audio clock is being used.
    pub fn using_audio_clock(&self) -> bool {
        self.use_audio_clock && self.audio.is_available()
    }
}

/// Decoded audio samples ready for playback.
#[derive(Clone)]
pub struct AudioSamples {
    /// Interleaved samples (f32, -1.0 to 1.0)
    pub data: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Presentation timestamp
    pub pts: Duration,
}

// ============================================================================
// Rodio-based audio player (when ffmpeg feature is enabled)
// ============================================================================

#[cfg(all(feature = "ffmpeg", not(target_os = "android")))]
mod rodio_impl {
    use super::*;
    use crossbeam_channel::{bounded, Receiver, Sender};
    use rodio::{buffer::SamplesBuffer, OutputStream, OutputStreamBuilder, Sink};
    use std::sync::Mutex;

    /// Rodio-based audio player.
    pub struct AudioPlayer {
        /// Audio handle for control
        handle: AudioHandle,
        /// Sender for audio samples
        sender: Sender<AudioSamples>,
        /// Receiver (held to keep channel alive, actual receiving done in thread)
        receiver: Receiver<AudioSamples>,
        /// Rodio output stream (must be kept alive)
        _stream: OutputStream,
        /// Rodio sink for playback control
        sink: Arc<Mutex<Sink>>,
        /// Current state
        state: AudioState,
        /// Device sample rate
        device_sample_rate: u32,
        /// Samples played for position tracking
        samples_played: Arc<std::sync::atomic::AtomicU64>,
    }

    impl AudioPlayer {
        /// Creates a new audio player.
        ///
        /// If `external_handle` is provided, the player will use it for volume/mute control.
        /// Otherwise, it creates its own handle.
        pub fn new_with_handle(
            _config: AudioConfig,
            external_handle: Option<AudioHandle>,
        ) -> Result<Self, String> {
            // Create audio output stream (rodio 0.21 API)
            let stream = OutputStreamBuilder::open_default_stream()
                .map_err(|e| format!("Failed to create audio output: {}", e))?;

            // Get the device sample rate from the stream config
            let device_sample_rate = stream.config().sample_rate();

            tracing::info!("Audio device sample rate: {}Hz", device_sample_rate);

            // Create channel for samples
            let (sender, receiver) = bounded(32);

            // Use external handle or create our own
            let handle = external_handle.unwrap_or_else(AudioHandle::new);
            handle.set_available(true);

            // Create sink connected to the stream's mixer (rodio 0.21 API)
            let sink = Sink::connect_new(&stream.mixer());
            sink.pause(); // Start paused

            tracing::info!(
                "Audio player initialized at {}Hz stereo",
                device_sample_rate
            );

            Ok(Self {
                handle,
                sender,
                receiver,
                _stream: stream,
                sink: Arc::new(Mutex::new(sink)),
                state: AudioState::Paused,
                device_sample_rate,
                samples_played: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            })
        }

        /// Creates a new audio player with its own handle.
        pub fn new(config: AudioConfig) -> Result<Self, String> {
            Self::new_with_handle(config, None)
        }

        /// Returns the device sample rate.
        pub fn device_sample_rate(&self) -> u32 {
            self.device_sample_rate
        }

        /// Returns the audio handle for control.
        pub fn handle(&self) -> AudioHandle {
            self.handle.clone()
        }

        /// Queues audio samples for playback using SamplesBuffer.
        pub fn queue_samples(&mut self, samples: AudioSamples) {
            if samples.data.is_empty() {
                return;
            }

            // Create a SamplesBuffer (known working rodio source)
            // Don't apply volume here - use Sink::set_volume() for dynamic control
            let buffer =
                SamplesBuffer::new(samples.channels, samples.sample_rate, samples.data.clone());

            // Append to sink and update volume dynamically
            if let Ok(sink) = self.sink.lock() {
                // Apply current volume/mute state to sink (dynamic, affects all queued audio)
                sink.set_volume(self.handle.effective_volume());
                sink.append(buffer);
            }

            // Update position
            let played = self.samples_played.fetch_add(
                samples.data.len() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            let seconds = played as f64 / (samples.sample_rate as f64 * samples.channels as f64);
            self.handle.set_position(Duration::from_secs_f64(seconds));
        }

        /// Starts audio playback.
        pub fn play(&mut self) {
            if let Ok(sink) = self.sink.lock() {
                sink.play();
                self.state = AudioState::Playing;
            }
        }

        /// Pauses audio playback.
        pub fn pause(&mut self) {
            if let Ok(sink) = self.sink.lock() {
                sink.pause();
                self.state = AudioState::Paused;
            }
        }

        /// Returns the current state.
        pub fn state(&self) -> AudioState {
            self.state
        }

        /// Clears the audio buffer (for seeking).
        pub fn clear(&mut self) {
            if let Ok(sink) = self.sink.lock() {
                sink.clear();
            }
            self.samples_played
                .store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

// ============================================================================
// Placeholder implementation (when ffmpeg feature is disabled)
// ============================================================================

#[cfg(any(not(feature = "ffmpeg"), target_os = "android"))]
mod placeholder_impl {
    use super::*;

    /// Placeholder audio player.
    pub struct AudioPlayer {
        /// Audio handle for control
        handle: AudioHandle,
        /// Current state
        #[allow(dead_code)]
        state: AudioState,
    }

    impl AudioPlayer {
        /// Creates a new audio player (placeholder).
        pub fn new(_config: AudioConfig) -> Result<Self, String> {
            Ok(Self {
                handle: AudioHandle::new(),
                state: AudioState::Uninitialized,
            })
        }

        /// Returns the audio handle for control.
        pub fn handle(&self) -> AudioHandle {
            self.handle.clone()
        }

        /// Queues audio samples for playback (no-op).
        pub fn queue_samples(&mut self, _samples: AudioSamples) {
            // No-op
        }

        /// Starts audio playback (no-op).
        pub fn play(&mut self) {}

        /// Pauses audio playback (no-op).
        pub fn pause(&mut self) {}

        /// Returns the current state.
        pub fn state(&self) -> AudioState {
            AudioState::Uninitialized
        }

        /// Clears the audio buffer (no-op).
        pub fn clear(&mut self) {}
    }
}

// Re-export the appropriate implementation
#[cfg(all(feature = "ffmpeg", not(target_os = "android")))]
pub use rodio_impl::AudioPlayer;

#[cfg(any(not(feature = "ffmpeg"), target_os = "android"))]
pub use placeholder_impl::AudioPlayer;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_handle_volume() {
        let handle = AudioHandle::new();
        assert_eq!(handle.volume(), 100);

        handle.set_volume(50);
        assert_eq!(handle.volume(), 50);

        handle.set_volume(150); // Should clamp to 100
        assert_eq!(handle.volume(), 100);
    }

    #[test]
    fn test_audio_handle_mute() {
        let handle = AudioHandle::new();
        assert!(!handle.is_muted());
        assert_eq!(handle.effective_volume(), 1.0);

        handle.set_muted(true);
        assert!(handle.is_muted());
        assert_eq!(handle.effective_volume(), 0.0);

        handle.toggle_mute();
        assert!(!handle.is_muted());
    }
}
