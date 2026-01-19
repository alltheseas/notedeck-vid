//! Android video decoder stub.
//!
//! This is a placeholder module for Android video decoding.
//! The actual implementation is in the video-android branch.

use std::time::Duration;

use super::video::{VideoDecoderBackend, VideoError, VideoFrame, VideoMetadata};

/// Android video decoder (stub).
///
/// This placeholder returns an error on construction.
/// The actual ExoPlayer-based implementation is in the video-android branch.
pub struct AndroidVideoDecoder;

impl AndroidVideoDecoder {
    /// Creates a new Android video decoder.
    ///
    /// This stub always returns an error.
    pub fn new(_url: &str) -> Result<Self, VideoError> {
        Err(VideoError::DecoderInit(
            "Android video decoder not available in this build".to_string(),
        ))
    }
}

impl VideoDecoderBackend for AndroidVideoDecoder {
    fn open(_url: &str) -> Result<Self, VideoError>
    where
        Self: Sized,
    {
        Err(VideoError::DecoderInit(
            "Android video decoder not available in this build".to_string(),
        ))
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        Err(VideoError::DecoderInit(
            "Android video decoder not available".to_string(),
        ))
    }

    fn seek(&mut self, _position: Duration) -> Result<(), VideoError> {
        Err(VideoError::SeekFailed(
            "Android video decoder not available".to_string(),
        ))
    }

    fn metadata(&self) -> &VideoMetadata {
        // This should never be called since construction fails
        unreachable!("AndroidVideoDecoder stub should not be constructed")
    }
}
