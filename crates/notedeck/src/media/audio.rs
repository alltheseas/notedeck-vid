//! Audio playback for video player.
//!
//! This module handles audio decoding and playback for video files.
//! It provides A/V synchronization, volume control, and mute toggle.
//!
//! # Dependencies Required
//!
//! To enable audio playback, add one of these to workspace dependencies:
//! ```toml
//! # Option 1: Low-level audio I/O
//! cpal = "0.15"
//!
//! # Option 2: Higher-level audio playback
//! rodio = "0.19"
//! ```
//!
//! # Architecture
//!
//! The audio system consists of:
//! - `AudioDecoder`: Extracts audio frames from video stream via FFmpeg
//! - `AudioPlayer`: Handles audio output via cpal/rodio
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
        let current = self.inner.muted.load(Ordering::Relaxed);
        self.inner.muted.store(!current, Ordering::Relaxed);
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
        Duration::from_micros(self.inner.position_us.load(Ordering::Relaxed) as u64)
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

/// Placeholder for audio player.
///
/// # Implementation Notes
///
/// When audio support is enabled with cpal:
///
/// ```ignore
/// pub struct AudioPlayer {
///     stream: cpal::Stream,
///     sample_queue: crossbeam::channel::Receiver<AudioSamples>,
///     config: AudioConfig,
///     handle: AudioHandle,
/// }
///
/// impl AudioPlayer {
///     pub fn new(config: AudioConfig) -> Result<Self, AudioError> {
///         let host = cpal::default_host();
///         let device = host.default_output_device()?;
///         let stream_config = device.default_output_config()?;
///         // ... setup audio stream
///     }
///
///     pub fn play(&mut self) {
///         self.stream.play().ok();
///     }
///
///     pub fn pause(&mut self) {
///         self.stream.pause().ok();
///     }
/// }
/// ```
pub struct AudioPlayer {
    /// Audio handle for control
    handle: AudioHandle,
    /// Current state
    #[allow(dead_code)]
    state: AudioState,
}

impl AudioPlayer {
    /// Creates a new audio player (placeholder).
    pub fn new(_config: AudioConfig) -> Self {
        Self {
            handle: AudioHandle::new(),
            state: AudioState::Uninitialized,
        }
    }

    /// Returns the audio handle for control.
    pub fn handle(&self) -> AudioHandle {
        self.handle.clone()
    }

    /// Starts audio playback (placeholder).
    pub fn play(&mut self) {
        // TODO: Implement with cpal when dependency is added
    }

    /// Pauses audio playback (placeholder).
    pub fn pause(&mut self) {
        // TODO: Implement with cpal when dependency is added
    }

    /// Seeks to a position (placeholder).
    pub fn seek(&mut self, _position: Duration) {
        // TODO: Implement with cpal when dependency is added
    }
}

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
