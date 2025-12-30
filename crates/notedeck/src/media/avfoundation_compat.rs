//! Minimal AVFoundation bindings for objc2 0.6.x.
//!
//! These bindings provide only what we need for streaming video playback.
//! Uses raw `msg_send!` calls to AVFoundation/CoreMedia/CoreVideo.

use std::ffi::c_void;

use objc2::encode::{Encode, Encoding};
use objc2::ffi::NSInteger;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObject};
use objc2::{class, msg_send, msg_send_id};
use objc2_foundation::{MainThreadMarker, NSString};

// ============================================================================
// Core Media types (opaque)
// ============================================================================

/// CMTime structure matching CoreMedia's definition.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CMTime {
    pub value: i64,
    pub timescale: i32,
    pub flags: u32,
    pub epoch: i64,
}

impl CMTime {
    pub const FLAGS_VALID: u32 = 1 << 0;

    pub fn new(value: i64, timescale: i32) -> Self {
        Self {
            value,
            timescale,
            flags: Self::FLAGS_VALID,
            epoch: 0,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.flags & Self::FLAGS_VALID != 0
    }
}

impl Default for CMTime {
    fn default() -> Self {
        Self::new(0, 1)
    }
}

// CMTime encoding: {value=q, timescale=i, flags=I, epoch=q} = "{CMTime=qiIq}"
unsafe impl Encode for CMTime {
    const ENCODING: Encoding = Encoding::Struct(
        "CMTime",
        &[
            Encoding::LongLong,  // value: i64
            Encoding::Int,       // timescale: i32
            Encoding::UInt,      // flags: u32
            Encoding::LongLong,  // epoch: i64
        ],
    );
}

// ============================================================================
// Core Video types
// ============================================================================

/// Opaque CVPixelBuffer type.
#[repr(C)]
pub struct CVPixelBuffer {
    _private: [u8; 0],
}

pub type CVPixelBufferRef = *mut CVPixelBuffer;

/// Pixel format constants.
pub const K_CV_PIXEL_FORMAT_TYPE_32_BGRA: u32 = 0x42475241; // 'BGRA'

/// Lock flags for CVPixelBuffer.
pub const K_CV_PIXEL_BUFFER_LOCK_READ_ONLY: u64 = 0x00000001;

// CVPixelBuffer functions (from CoreVideo.framework)
#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    pub fn CVPixelBufferGetWidth(pixelBuffer: CVPixelBufferRef) -> usize;
    pub fn CVPixelBufferGetHeight(pixelBuffer: CVPixelBufferRef) -> usize;
    pub fn CVPixelBufferGetBytesPerRow(pixelBuffer: CVPixelBufferRef) -> usize;
    pub fn CVPixelBufferGetPixelFormatType(pixelBuffer: CVPixelBufferRef) -> u32;
    pub fn CVPixelBufferLockBaseAddress(pixelBuffer: CVPixelBufferRef, lockFlags: u64) -> i32;
    pub fn CVPixelBufferUnlockBaseAddress(pixelBuffer: CVPixelBufferRef, lockFlags: u64) -> i32;
    pub fn CVPixelBufferGetBaseAddress(pixelBuffer: CVPixelBufferRef) -> *mut c_void;
    pub fn CVPixelBufferRetain(pixelBuffer: CVPixelBufferRef) -> CVPixelBufferRef;
    pub fn CVPixelBufferRelease(pixelBuffer: CVPixelBufferRef);
}

/// CoreFoundation string constant for pixel format key.
#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    pub static kCVPixelBufferPixelFormatTypeKey: *const c_void;
}

// ============================================================================
// AVFoundation wrappers
// ============================================================================

/// AVPlayerItemStatus enum values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(isize)]
pub enum AVPlayerItemStatus {
    Unknown = 0,
    ReadyToPlay = 1,
    Failed = 2,
}

impl From<NSInteger> for AVPlayerItemStatus {
    fn from(value: NSInteger) -> Self {
        match value {
            1 => AVPlayerItemStatus::ReadyToPlay,
            2 => AVPlayerItemStatus::Failed,
            _ => AVPlayerItemStatus::Unknown,
        }
    }
}

/// AVPlayerTimeControlStatus enum values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(isize)]
pub enum AVPlayerTimeControlStatus {
    Paused = 0,
    WaitingToPlayAtSpecifiedRate = 1,
    Playing = 2,
}

impl From<NSInteger> for AVPlayerTimeControlStatus {
    fn from(value: NSInteger) -> Self {
        match value {
            1 => AVPlayerTimeControlStatus::WaitingToPlayAtSpecifiedRate,
            2 => AVPlayerTimeControlStatus::Playing,
            _ => AVPlayerTimeControlStatus::Paused,
        }
    }
}

// ============================================================================
// NSURL wrapper
// ============================================================================

/// Creates an NSURL from a string (for HTTP/HTTPS URLs).
pub fn nsurl_with_string(url_string: &str) -> Option<Retained<NSObject>> {
    unsafe {
        let ns_string = NSString::from_str(url_string);
        let cls = class!(NSURL);
        msg_send_id![cls, URLWithString: &*ns_string]
    }
}

/// Creates an NSURL from a file path.
pub fn nsurl_file_url_with_path(path: &str) -> Retained<NSObject> {
    unsafe {
        let ns_string = NSString::from_str(path);
        let cls = class!(NSURL);
        msg_send_id![cls, fileURLWithPath: &*ns_string]
    }
}

// ============================================================================
// AVPlayerItem wrapper
// ============================================================================

/// Creates an AVPlayerItem with the given URL.
///
/// **Requires main thread.**
pub fn avplayeritem_with_url(_mtm: MainThreadMarker, url: &NSObject) -> Retained<NSObject> {
    unsafe {
        let cls = class!(AVPlayerItem);
        msg_send_id![cls, playerItemWithURL: url]
    }
}

/// Gets the status of an AVPlayerItem.
pub fn avplayeritem_status(item: &NSObject) -> AVPlayerItemStatus {
    unsafe {
        let status: NSInteger = msg_send![item, status];
        AVPlayerItemStatus::from(status)
    }
}

/// Gets the error from an AVPlayerItem (if status is Failed).
pub fn avplayeritem_error(item: &NSObject) -> Option<Retained<NSObject>> {
    unsafe { msg_send_id![item, error] }
}

/// Gets the duration of an AVPlayerItem.
pub fn avplayeritem_duration(item: &NSObject) -> CMTime {
    unsafe { msg_send![item, duration] }
}

/// Gets the asset from an AVPlayerItem.
pub fn avplayeritem_asset(item: &NSObject) -> Retained<NSObject> {
    unsafe { msg_send_id![item, asset] }
}

/// Checks if playback is likely to keep up.
pub fn avplayeritem_is_playback_likely_to_keep_up(item: &NSObject) -> bool {
    unsafe { msg_send![item, isPlaybackLikelyToKeepUp] }
}

/// Adds an output to an AVPlayerItem.
pub fn avplayeritem_add_output(item: &NSObject, output: &NSObject) {
    unsafe { msg_send![item, addOutput: output] }
}

// ============================================================================
// AVPlayer wrapper
// ============================================================================

/// Creates an AVPlayer with the given player item.
///
/// **Requires main thread.**
pub fn avplayer_with_player_item(_mtm: MainThreadMarker, item: &NSObject) -> Retained<NSObject> {
    unsafe {
        let cls = class!(AVPlayer);
        msg_send_id![cls, playerWithPlayerItem: item]
    }
}

/// Starts playback.
pub fn avplayer_play(player: &NSObject) {
    unsafe { msg_send![player, play] }
}

/// Pauses playback.
pub fn avplayer_pause(player: &NSObject) {
    unsafe { msg_send![player, pause] }
}

/// Gets the current playback time.
pub fn avplayer_current_time(player: &NSObject) -> CMTime {
    unsafe { msg_send![player, currentTime] }
}

/// Gets the time control status.
pub fn avplayer_time_control_status(player: &NSObject) -> AVPlayerTimeControlStatus {
    unsafe {
        let status: NSInteger = msg_send![player, timeControlStatus];
        AVPlayerTimeControlStatus::from(status)
    }
}

/// Seeks to the specified time with zero tolerance.
pub fn avplayer_seek_to_time(player: &NSObject, time: CMTime) {
    let zero = CMTime::new(0, 1);
    unsafe {
        msg_send![player, seekToTime: time toleranceBefore: zero toleranceAfter: zero]
    }
}

// ============================================================================
// AVPlayerItemVideoOutput wrapper
// ============================================================================

/// Creates an AVPlayerItemVideoOutput with BGRA pixel format.
pub fn avplayeritemvideooutput_new() -> Retained<NSObject> {
    unsafe {
        // Create dictionary with pixel format
        let dict_cls = class!(NSMutableDictionary);
        let dict: Retained<NSObject> = msg_send_id![dict_cls, new];

        // Create NSNumber with BGRA format
        let num_cls = class!(NSNumber);
        let format_num: Retained<NSObject> =
            msg_send_id![num_cls, numberWithUnsignedInt: K_CV_PIXEL_FORMAT_TYPE_32_BGRA];

        // Get the key (CFString -> NSString bridged)
        let key = kCVPixelBufferPixelFormatTypeKey as *const NSObject;
        let key_ref: &NSObject = &*key;

        // Set the value
        let _: () = msg_send![&*dict, setObject: &*format_num forKey: key_ref];

        // Create output with settings
        let output_cls = class!(AVPlayerItemVideoOutput);
        msg_send_id![output_cls, alloc]
            .map(|obj: Retained<NSObject>| {
                let initialized: Retained<NSObject> =
                    msg_send_id![&*obj, initWithPixelBufferAttributes: &*dict];
                initialized
            })
            .expect("Failed to create AVPlayerItemVideoOutput")
    }
}

/// Checks if a new pixel buffer is available for the given time.
pub fn avplayeritemvideooutput_has_new_pixel_buffer(output: &NSObject, time: CMTime) -> bool {
    unsafe { msg_send![output, hasNewPixelBufferForItemTime: time] }
}

/// Copies the pixel buffer for the given time.
///
/// Returns the pixel buffer (must be released with CVPixelBufferRelease) and the display time.
pub fn avplayeritemvideooutput_copy_pixel_buffer(
    output: &NSObject,
    time: CMTime,
) -> Option<(CVPixelBufferRef, CMTime)> {
    unsafe {
        let mut display_time = CMTime::default();
        let pixel_buffer: CVPixelBufferRef = msg_send![
            output,
            copyPixelBufferForItemTime: time
            itemTimeForDisplay: &mut display_time as *mut CMTime
        ];

        if pixel_buffer.is_null() {
            None
        } else {
            Some((pixel_buffer, display_time))
        }
    }
}

// ============================================================================
// AVAsset wrapper
// ============================================================================

/// Gets tracks with the specified media type from an asset.
pub fn avasset_tracks_with_media_type(asset: &NSObject, media_type: &str) -> Retained<NSObject> {
    unsafe {
        let media_type_ns = NSString::from_str(media_type);
        msg_send_id![asset, tracksWithMediaType: &*media_type_ns]
    }
}

// ============================================================================
// AVAssetTrack wrapper
// ============================================================================

/// CGSize structure.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CGSize {
    pub width: f64,
    pub height: f64,
}

// CGSize encoding: {width=d, height=d} = "{CGSize=dd}"
unsafe impl Encode for CGSize {
    const ENCODING: Encoding = Encoding::Struct("CGSize", &[Encoding::Double, Encoding::Double]);
}

/// Gets the natural size of a track.
pub fn avassettrack_natural_size(track: &NSObject) -> CGSize {
    unsafe { msg_send![track, naturalSize] }
}

/// Gets the nominal frame rate of a track.
pub fn avassettrack_nominal_frame_rate(track: &NSObject) -> f32 {
    unsafe { msg_send![track, nominalFrameRate] }
}

// ============================================================================
// NSArray helpers
// ============================================================================

/// Gets the count of an NSArray.
pub fn nsarray_count(array: &NSObject) -> usize {
    unsafe { msg_send![array, count] }
}

/// Gets an object at the specified index from an NSArray.
pub fn nsarray_object_at_index(array: &NSObject, index: usize) -> Retained<NSObject> {
    unsafe { msg_send_id![array, objectAtIndex: index] }
}

// ============================================================================
// NSError helper
// ============================================================================

/// Gets the localized description from an NSError.
pub fn nserror_localized_description(error: &NSObject) -> String {
    unsafe {
        let desc: Retained<NSString> = msg_send_id![error, localizedDescription];
        desc.to_string()
    }
}

// ============================================================================
// Media type constant
// ============================================================================

/// AVMediaTypeVideo constant string.
pub const AV_MEDIA_TYPE_VIDEO: &str = "vide";
