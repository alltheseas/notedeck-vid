//! FFmpeg-based audio decoder for extracting audio from video files.
//!
//! This module provides audio decoding using FFmpeg, converting compressed
//! audio to PCM samples that can be played back via rodio.

use std::time::Duration;

use super::audio::AudioSamples;

/// Audio decoder error types.
#[derive(Debug, Clone)]
pub enum AudioError {
    /// Failed to open the audio stream
    OpenFailed(String),
    /// No audio stream found
    NoAudioStream,
    /// Decoder initialization failed
    DecoderInit(String),
    /// Decoding failed
    DecodeFailed(String),
    /// Seek failed
    SeekFailed(String),
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenFailed(s) => write!(f, "Open failed: {s}"),
            Self::NoAudioStream => write!(f, "No audio stream found"),
            Self::DecoderInit(s) => write!(f, "Decoder init failed: {s}"),
            Self::DecodeFailed(s) => write!(f, "Decode failed: {s}"),
            Self::SeekFailed(s) => write!(f, "Seek failed: {s}"),
        }
    }
}

impl std::error::Error for AudioError {}

/// Audio metadata from the stream.
#[derive(Debug, Clone)]
pub struct AudioMetadata {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Audio codec name
    pub codec: String,
    /// Duration if known
    pub duration: Option<Duration>,
}

// ============================================================================
// Real FFmpeg implementation (when feature is enabled)
// ============================================================================

#[cfg(all(feature = "ffmpeg", not(target_os = "android")))]
mod real_impl {
    use super::*;
    use ffmpeg_next as ffmpeg;
    use ffmpeg_next::ffi;
    use std::sync::Once;

    static FFMPEG_INIT: Once = Once::new();

    fn init_ffmpeg() {
        FFMPEG_INIT.call_once(|| {
            ffmpeg::init().expect("Failed to initialize FFmpeg");
        });
    }

    /// FFmpeg-based audio decoder.
    pub struct AudioDecoder {
        /// Input format context
        input: ffmpeg::format::context::Input,
        /// Audio stream index
        audio_stream_index: usize,
        /// Audio decoder
        decoder: ffmpeg::decoder::Audio,
        /// Audio resampler for format conversion
        resampler: Option<ffmpeg::software::resampling::Context>,
        /// Audio metadata
        metadata: AudioMetadata,
        /// Stream time base (numerator, denominator)
        time_base: (i32, i32),
        /// Whether EOF has been reached
        eof_reached: bool,
        /// Packet iterator state
        packet_iter_finished: bool,
        /// Target output sample rate
        target_sample_rate: u32,
    }

    impl AudioDecoder {
        /// Creates a new audio decoder for the given URL or file path.
        pub fn new(url: &str, target_sample_rate: u32) -> Result<Self, AudioError> {
            init_ffmpeg();

            // Open input file/stream
            let input = ffmpeg::format::input(&url)
                .map_err(|e| AudioError::OpenFailed(format!("Failed to open {url}: {e}")))?;

            // Find audio stream
            let audio_stream = input
                .streams()
                .best(ffmpeg::media::Type::Audio)
                .ok_or(AudioError::NoAudioStream)?;

            let audio_stream_index = audio_stream.index();
            let time_base = audio_stream.time_base();

            // Get codec parameters
            let codec_params = audio_stream.parameters();

            // Create decoder context from parameters
            let context =
                ffmpeg::codec::context::Context::from_parameters(codec_params).map_err(|e| {
                    AudioError::DecoderInit(format!("Failed to create codec context: {e}"))
                })?;

            // Open decoder
            let decoder = context
                .decoder()
                .audio()
                .map_err(|e| AudioError::DecoderInit(format!("Failed to open decoder: {e}")))?;

            // Extract metadata
            let duration = if input.duration() > 0 {
                Some(Duration::from_micros(
                    (input.duration() as f64 * 1_000_000.0 / ffi::AV_TIME_BASE as f64) as u64,
                ))
            } else {
                None
            };

            let metadata = AudioMetadata {
                sample_rate: decoder.rate(),
                channels: decoder.channels() as u16,
                codec: decoder
                    .codec()
                    .map(|c| c.name().to_string())
                    .unwrap_or_else(|| "unknown".to_string()),
                duration,
            };

            tracing::info!(
                "Audio: {}Hz -> {}Hz, {} channels, codec: {}, duration: {:?}",
                metadata.sample_rate,
                target_sample_rate,
                metadata.channels,
                metadata.codec,
                metadata.duration
            );

            Ok(Self {
                input,
                audio_stream_index,
                decoder,
                resampler: None,
                metadata,
                time_base: (time_base.0, time_base.1),
                eof_reached: false,
                packet_iter_finished: false,
                target_sample_rate,
            })
        }

        /// Returns the audio metadata.
        pub fn metadata(&self) -> &AudioMetadata {
            &self.metadata
        }

        fn pts_to_duration(&self, pts: i64) -> Duration {
            if pts < 0 || self.time_base.1 == 0 {
                return Duration::ZERO;
            }
            let seconds = (pts as f64) * (self.time_base.0 as f64) / (self.time_base.1 as f64);
            Duration::from_secs_f64(seconds.max(0.0))
        }

        fn ensure_resampler(&mut self, frame: &ffmpeg::frame::Audio) -> Result<(), AudioError> {
            let src_format = frame.format();
            let src_rate = frame.rate();
            let src_layout = frame.channel_layout();

            // Target format: stereo, device sample rate, f32 packed (interleaved)
            let dst_format = ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed);
            let dst_rate = self.target_sample_rate;
            let dst_layout = ffmpeg::ChannelLayout::STEREO;

            // Check if we need to recreate the resampler (format/rate/layout changed)
            let needs_recreate = match &self.resampler {
                None => true,
                Some(resampler) => {
                    let input = resampler.input();
                    input.format != src_format
                        || input.rate != src_rate
                        || input.channel_layout != src_layout
                }
            };

            if needs_recreate {
                let resampler = ffmpeg::software::resampling::Context::get(
                    src_format, src_layout, src_rate, dst_format, dst_layout, dst_rate,
                )
                .map_err(|e| {
                    AudioError::DecodeFailed(format!("Failed to create resampler: {e}"))
                })?;

                self.resampler = Some(resampler);
            }

            Ok(())
        }

        fn frame_to_samples(
            &mut self,
            frame: &ffmpeg::frame::Audio,
        ) -> Result<AudioSamples, AudioError> {
            self.ensure_resampler(frame)?;

            let resampler = self.resampler.as_mut().unwrap();

            // Create output frame
            let mut output = ffmpeg::frame::Audio::empty();

            // Run resampler
            let _delay = resampler
                .run(frame, &mut output)
                .map_err(|e| AudioError::DecodeFailed(format!("Resampling failed: {e}")))?;

            // Get the actual number of samples from the frame
            let num_samples = output.samples();

            if num_samples == 0 {
                // No samples produced yet (resampler buffering)
                return Ok(AudioSamples {
                    data: vec![],
                    sample_rate: self.target_sample_rate,
                    channels: 2,
                    pts: Duration::ZERO,
                });
            }

            // Get raw byte data and interpret as f32
            let raw_data = output.data(0);
            let bytes_per_sample = 4; // f32
            let channels = 2;
            let expected_bytes = num_samples * channels * bytes_per_sample;

            // Debug: log first frame info
            static LOGGED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::info!(
                    "Audio frame: {} samples, raw data {} bytes, expected {} bytes, format {:?}, rate {}",
                    num_samples,
                    raw_data.len(),
                    expected_bytes,
                    output.format(),
                    output.rate()
                );
            }

            // Convert raw bytes to f32 samples
            let num_floats = (raw_data.len() / 4).min(num_samples * channels);
            let mut samples = Vec::with_capacity(num_floats);

            for i in 0..num_floats {
                let offset = i * 4;
                if offset + 4 <= raw_data.len() {
                    let sample = f32::from_ne_bytes([
                        raw_data[offset],
                        raw_data[offset + 1],
                        raw_data[offset + 2],
                        raw_data[offset + 3],
                    ]);
                    samples.push(sample);
                }
            }

            let pts = frame.pts().unwrap_or(0);

            // Use actual output rate from resampler
            let actual_rate = output.rate();

            Ok(AudioSamples {
                data: samples,
                sample_rate: actual_rate,
                channels: 2,
                pts: self.pts_to_duration(pts),
            })
        }

        /// Decodes the next audio samples.
        pub fn decode_next(&mut self) -> Result<Option<AudioSamples>, AudioError> {
            if self.eof_reached {
                return Ok(None);
            }

            let mut decoded_frame = ffmpeg::frame::Audio::empty();

            loop {
                // Try to receive a frame from the decoder
                match self.decoder.receive_frame(&mut decoded_frame) {
                    Ok(()) => {
                        // Got a frame
                        let samples = self.frame_to_samples(&decoded_frame)?;
                        return Ok(Some(samples));
                    }
                    Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::error::EAGAIN => {
                        // Need more packets
                        if self.packet_iter_finished {
                            // Send EOF to decoder
                            self.decoder.send_eof().ok();
                            self.packet_iter_finished = false;
                            continue;
                        }

                        // Read next packet
                        let mut found_audio_packet = false;
                        for (stream, packet) in self.input.packets() {
                            if stream.index() == self.audio_stream_index {
                                self.decoder.send_packet(&packet).map_err(|e| {
                                    AudioError::DecodeFailed(format!("Send packet failed: {e}"))
                                })?;
                                found_audio_packet = true;
                                break;
                            }
                        }

                        if !found_audio_packet {
                            self.packet_iter_finished = true;
                        }
                    }
                    Err(ffmpeg::Error::Eof) => {
                        self.eof_reached = true;
                        return Ok(None);
                    }
                    Err(e) => {
                        return Err(AudioError::DecodeFailed(format!("Decode error: {e}")));
                    }
                }
            }
        }

        /// Seeks to a specific position.
        pub fn seek(&mut self, position: Duration) -> Result<(), AudioError> {
            let timestamp =
                (position.as_secs_f64() * self.time_base.1 as f64 / self.time_base.0 as f64) as i64;

            self.input
                .seek(timestamp, ..timestamp)
                .map_err(|e| AudioError::SeekFailed(format!("Seek failed: {e}")))?;

            // Flush decoder
            self.decoder.flush();
            self.eof_reached = false;
            self.packet_iter_finished = false;

            Ok(())
        }
    }

    // SAFETY: AudioDecoder is only accessed from a single thread (the audio decode thread).
    unsafe impl Send for AudioDecoder {}
}

// ============================================================================
// Placeholder implementation (when feature is disabled)
// ============================================================================

#[cfg(any(not(feature = "ffmpeg"), target_os = "android"))]
mod placeholder_impl {
    use super::*;

    /// Placeholder audio decoder.
    pub struct AudioDecoder {
        metadata: AudioMetadata,
    }

    impl AudioDecoder {
        /// Creates a new audio decoder (placeholder).
        pub fn new(_url: &str, _target_sample_rate: u32) -> Result<Self, AudioError> {
            Err(AudioError::NoAudioStream)
        }

        /// Returns the audio metadata.
        pub fn metadata(&self) -> &AudioMetadata {
            &self.metadata
        }

        /// Decodes the next audio samples (placeholder).
        pub fn decode_next(&mut self) -> Result<Option<AudioSamples>, AudioError> {
            Ok(None)
        }

        /// Seeks to a specific position (placeholder).
        pub fn seek(&mut self, _position: Duration) -> Result<(), AudioError> {
            Ok(())
        }
    }
}

// Re-export the appropriate implementation
#[cfg(all(feature = "ffmpeg", not(target_os = "android")))]
pub use real_impl::AudioDecoder;

#[cfg(any(not(feature = "ffmpeg"), target_os = "android"))]
pub use placeholder_impl::AudioDecoder;
