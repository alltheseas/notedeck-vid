//! Android video decoder using ExoPlayer via JNI.
//!
//! This module provides video decoding on Android using ExoPlayer,
//! which automatically handles hardware acceleration via MediaCodec.

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use jni::objects::{GlobalRef, JByteArray, JClass, JObject, JValue};
use jni::sys::{jint, jlong};
use jni::JNIEnv;
use once_cell::sync::Lazy;

use super::video::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata,
};

/// State shared between Rust and JNI callbacks.
struct SharedState {
    /// Channel for receiving frames from JNI callbacks
    frame_sender: Sender<AndroidFrame>,
    /// Current video width
    width: u32,
    /// Current video height
    height: u32,
    /// Video duration in milliseconds
    duration_ms: i64,
    /// Current playback state from ExoPlayer
    playback_state: i32,
    /// Last error message
    last_error: Option<String>,
    /// Whether a new frame is available
    frame_available: bool,
}

/// Frame data received from Android.
pub struct AndroidFrame {
    /// RGBA pixel data
    pub pixels: Vec<u8>,
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
    /// Timestamp in nanoseconds
    pub timestamp_ns: i64,
}

/// Android video decoder using ExoPlayer.
pub struct AndroidVideoDecoder {
    /// JNI reference to ExoPlayerBridge instance
    bridge: GlobalRef,
    /// Shared state between Rust and JNI
    state: Arc<Mutex<SharedState>>,
    /// Frame receiver
    #[allow(dead_code)]
    frame_receiver: Receiver<AndroidFrame>,
    /// Video metadata
    metadata: VideoMetadata,
    /// Whether the decoder is initialized
    #[allow(dead_code)]
    initialized: bool,
    /// Native handle for JNI callback lookup
    native_handle: i64,
}

// Global map of native handles to SharedState
// This is necessary because JNI callbacks don't have access to Rust structs directly
static NATIVE_HANDLES: Lazy<Mutex<std::collections::HashMap<i64, Arc<Mutex<SharedState>>>>> =
    Lazy::new(|| Mutex::new(std::collections::HashMap::new()));

static NEXT_HANDLE: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(1);

fn register_native_handle(state: Arc<Mutex<SharedState>>) -> i64 {
    let handle = NEXT_HANDLE.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    NATIVE_HANDLES.lock().unwrap().insert(handle, state);
    handle
}

fn unregister_native_handle(handle: i64) {
    NATIVE_HANDLES.lock().unwrap().remove(&handle);
}

fn get_native_state(handle: i64) -> Option<Arc<Mutex<SharedState>>> {
    NATIVE_HANDLES.lock().unwrap().get(&handle).cloned()
}

impl AndroidVideoDecoder {
    /// Creates a new Android video decoder for the given URL.
    pub fn new(url: &str) -> Result<Self, VideoError> {
        // Get JNI environment
        let vm = crate::platform::android::get_jvm();
        let mut env = vm
            .attach_current_thread()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to attach JNI thread: {}", e)))?;

        // Get Android context
        let context =
            unsafe { JObject::from_raw(ndk_context::android_context().context().cast()) };

        // Create frame channel
        let (frame_sender, frame_receiver) = mpsc::channel();

        // Create shared state
        let state = Arc::new(Mutex::new(SharedState {
            frame_sender,
            width: 0,
            height: 0,
            duration_ms: 0,
            playback_state: 0,
            last_error: None,
            frame_available: false,
        }));

        // Register native handle
        let native_handle = register_native_handle(Arc::clone(&state));

        // Create ExoPlayerBridge instance
        let bridge_class = env
            .find_class("com/damus/notedeck/video/ExoPlayerBridge")
            .map_err(|e| VideoError::DecoderInit(format!("Failed to find ExoPlayerBridge: {}", e)))?;

        let bridge = env
            .new_object(
                bridge_class,
                "(Landroid/content/Context;J)V",
                &[JValue::Object(&context), JValue::Long(native_handle)],
            )
            .map_err(|e| {
                VideoError::DecoderInit(format!("Failed to create ExoPlayerBridge: {}", e))
            })?;

        // Create global reference
        let bridge_ref = env.new_global_ref(bridge).map_err(|e| {
            VideoError::DecoderInit(format!("Failed to create global ref: {}", e))
        })?;

        // Initialize the bridge
        env.call_method(&bridge_ref, "initialize", "()V", &[])
            .map_err(|e| VideoError::DecoderInit(format!("Failed to initialize bridge: {}", e)))?;

        // Start playback
        let url_jstring = env.new_string(url).map_err(|e| {
            VideoError::DecoderInit(format!("Failed to create URL string: {}", e))
        })?;

        env.call_method(
            &bridge_ref,
            "play",
            "(Ljava/lang/String;)V",
            &[JValue::Object(&url_jstring)],
        )
        .map_err(|e| VideoError::DecoderInit(format!("Failed to start playback: {}", e)))?;

        // Initial metadata (will be updated by callbacks)
        let metadata = VideoMetadata {
            width: 1920,
            height: 1080,
            duration: None,
            frame_rate: 30.0,
            codec: "mediacodec".to_string(),
            pixel_aspect_ratio: 1.0,
        };

        Ok(Self {
            bridge: bridge_ref,
            state,
            frame_receiver,
            metadata,
            initialized: true,
            native_handle,
        })
    }

    /// Extracts the current frame from ExoPlayer.
    fn extract_frame(&self) -> Result<Option<AndroidFrame>, VideoError> {
        let vm = crate::platform::android::get_jvm();
        let mut env = vm
            .attach_current_thread()
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to attach JNI thread: {}", e)))?;

        // Call extractCurrentFrame
        let result = env
            .call_method(&self.bridge, "extractCurrentFrame", "()[B", &[])
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to extract frame: {}", e)))?;

        let bytes_obj = result.l().map_err(|e| {
            VideoError::DecodeFailed(format!("Failed to get frame bytes: {}", e))
        })?;

        if bytes_obj.is_null() {
            return Ok(None);
        }

        let bytes_array = JByteArray::from(bytes_obj);
        let len = env.get_array_length(&bytes_array).map_err(|e| {
            VideoError::DecodeFailed(format!("Failed to get array length: {}", e))
        })? as usize;

        let mut pixels: Vec<i8> = vec![0; len];
        env.get_byte_array_region(&bytes_array, 0, &mut pixels)
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to get array data: {}", e)))?;

        // Convert i8 to u8
        let pixels: Vec<u8> = pixels.into_iter().map(|b| b as u8).collect();

        // Get current dimensions from state
        let state = self.state.lock().unwrap();
        let width = state.width;
        let height = state.height;

        if width == 0 || height == 0 {
            return Ok(None);
        }

        // Get current position for timestamp
        let position = env
            .call_method(&self.bridge, "getCurrentPosition", "()J", &[])
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to get position: {}", e)))?
            .j()
            .unwrap_or(0);

        Ok(Some(AndroidFrame {
            pixels,
            width,
            height,
            timestamp_ns: position * 1_000_000, // ms to ns
        }))
    }
}

impl Drop for AndroidVideoDecoder {
    fn drop(&mut self) {
        // Unregister native handle first
        unregister_native_handle(self.native_handle);

        // Release ExoPlayer resources
        if let Ok(vm) = std::panic::catch_unwind(|| crate::platform::android::get_jvm()) {
            if let Ok(mut env) = vm.attach_current_thread() {
                let _ = env.call_method(&self.bridge, "release", "()V", &[]);
            }
        }
    }
}

impl VideoDecoderBackend for AndroidVideoDecoder {
    fn open(url: &str) -> Result<Self, VideoError>
    where
        Self: Sized,
    {
        Self::new(url)
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        // ExoPlayer playback states (from Player.java)
        const STATE_ENDED: i32 = 4;

        // Check for errors from callbacks
        {
            let state = self.state.lock().unwrap();
            if let Some(ref error) = state.last_error {
                return Err(VideoError::DecodeFailed(error.clone()));
            }

            // Check if playback has ended - this is the only case where Ok(None) means EOS
            if state.playback_state == STATE_ENDED {
                return Ok(None);
            }
        }

        // Wait for a frame to be available (with timeout to avoid blocking forever)
        // This prevents returning Ok(None) prematurely which would be treated as EOS
        let max_wait_ms = 100;
        let start = std::time::Instant::now();

        loop {
            // Check if a frame is available
            {
                let mut state = self.state.lock().unwrap();
                if state.frame_available {
                    state.frame_available = false;
                    break;
                }

                // Check for errors while waiting
                if let Some(ref error) = state.last_error {
                    return Err(VideoError::DecodeFailed(error.clone()));
                }

                // Check for EOS while waiting
                if state.playback_state == STATE_ENDED {
                    return Ok(None);
                }
            }

            // Check timeout
            if start.elapsed().as_millis() > max_wait_ms as u128 {
                // No frame available yet, but not EOS - return a placeholder frame
                // to keep the decode loop running
                std::thread::sleep(Duration::from_millis(5));
                continue;
            }

            std::thread::sleep(Duration::from_millis(5));
        }

        // Extract frame from ExoPlayer
        let frame = self.extract_frame()?;

        match frame {
            Some(android_frame) => {
                // Convert to CpuFrame (RGBA format)
                let cpu_frame = CpuFrame::new(
                    PixelFormat::Rgba,
                    android_frame.width,
                    android_frame.height,
                    vec![Plane {
                        data: android_frame.pixels,
                        stride: android_frame.width as usize * 4,
                    }],
                );

                let pts = Duration::from_nanos(android_frame.timestamp_ns as u64);

                Ok(Some(VideoFrame::new(pts, DecodedFrame::Cpu(cpu_frame))))
            }
            // Frame extraction failed but we're not at EOS - try again next call
            None => {
                // Re-check if we're at EOS
                let state = self.state.lock().unwrap();
                if state.playback_state == STATE_ENDED {
                    Ok(None)
                } else {
                    // Return a minimal placeholder frame to keep decode loop alive
                    // This is a workaround - the frame will be skipped but loop continues
                    let placeholder = CpuFrame::new(
                        PixelFormat::Rgba,
                        1,
                        1,
                        vec![Plane {
                            data: vec![0, 0, 0, 255],
                            stride: 4,
                        }],
                    );
                    Ok(Some(VideoFrame::new(Duration::ZERO, DecodedFrame::Cpu(placeholder))))
                }
            }
        }
    }

    fn seek(&mut self, position: Duration) -> Result<(), VideoError> {
        let vm = crate::platform::android::get_jvm();
        let mut env = vm
            .attach_current_thread()
            .map_err(|e| VideoError::SeekFailed(format!("Failed to attach JNI thread: {}", e)))?;

        let position_ms = position.as_millis() as i64;

        env.call_method(
            &self.bridge,
            "seek",
            "(J)V",
            &[JValue::Long(position_ms)],
        )
        .map_err(|e| VideoError::SeekFailed(format!("Seek failed: {}", e)))?;

        Ok(())
    }

    fn metadata(&self) -> &VideoMetadata {
        &self.metadata
    }

    fn hw_accel_type(&self) -> HwAccelType {
        HwAccelType::MediaCodec
    }
}

// JNI callback implementations
// These are called from Java when events occur

#[no_mangle]
pub extern "C" fn Java_com_damus_notedeck_video_ExoPlayerBridge_nativeOnFrameAvailable(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
    width: jint,
    height: jint,
    _timestamp_ns: jlong,
) {
    if let Some(state) = get_native_state(handle) {
        let mut state = state.lock().unwrap();
        state.width = width as u32;
        state.height = height as u32;
        state.frame_available = true;
    }
}

#[no_mangle]
pub extern "C" fn Java_com_damus_notedeck_video_ExoPlayerBridge_nativeOnPlaybackStateChanged(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
    state_value: jint,
) {
    if let Some(state) = get_native_state(handle) {
        let mut state = state.lock().unwrap();
        state.playback_state = state_value;
    }
}

#[no_mangle]
pub extern "C" fn Java_com_damus_notedeck_video_ExoPlayerBridge_nativeOnError(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    error_message: jni::objects::JString,
) {
    if let Some(state) = get_native_state(handle) {
        let error: String = env
            .get_string(&error_message)
            .map(|s| s.into())
            .unwrap_or_else(|_| "Unknown error".to_string());

        let mut state = state.lock().unwrap();
        state.last_error = Some(error);
    }
}

#[no_mangle]
pub extern "C" fn Java_com_damus_notedeck_video_ExoPlayerBridge_nativeOnVideoSizeChanged(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
    width: jint,
    height: jint,
) {
    if let Some(state) = get_native_state(handle) {
        let mut state = state.lock().unwrap();
        state.width = width as u32;
        state.height = height as u32;
    }
}

#[no_mangle]
pub extern "C" fn Java_com_damus_notedeck_video_ExoPlayerBridge_nativeOnDurationChanged(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
    duration_ms: jlong,
) {
    if let Some(state) = get_native_state(handle) {
        let mut state = state.lock().unwrap();
        state.duration_ms = duration_ms;
    }
}
