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
    /// Last known playback position (for placeholder frames)
    last_position: Duration,
    /// Video URL (for deferred playback start)
    url: String,
    /// Whether playback has been started
    started: bool,
}

/// Converts an Arc<Mutex<SharedState>> into a raw pointer handle for JNI.
/// The Arc's reference count is incremented, so the caller must call
/// `release_native_handle` to avoid leaking memory.
fn create_native_handle(state: Arc<Mutex<SharedState>>) -> i64 {
    // Clone to increment refcount, then convert to raw pointer
    let ptr = Arc::into_raw(state);
    ptr as i64
}

/// Releases a native handle, decrementing the Arc's reference count.
/// # Safety
/// The handle must have been created by `create_native_handle` and must not
/// have been released before.
fn release_native_handle(handle: i64) {
    if handle == 0 {
        return;
    }
    // Convert back to Arc and let it drop (decrements refcount)
    let ptr = handle as *const Mutex<SharedState>;
    unsafe {
        let _ = Arc::from_raw(ptr);
    }
}

/// Gets a clone of the SharedState Arc from a native handle.
/// Returns None if the handle is null (0).
/// # Safety
/// The handle must be valid (created by `create_native_handle` and not yet released).
fn get_native_state(handle: i64) -> Option<Arc<Mutex<SharedState>>> {
    if handle == 0 {
        return None;
    }
    let ptr = handle as *const Mutex<SharedState>;
    // Reconstruct Arc, clone it, then forget the original to avoid double-free
    let arc = unsafe { Arc::from_raw(ptr) };
    let cloned = Arc::clone(&arc);
    std::mem::forget(arc);
    Some(cloned)
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
        let context = unsafe { JObject::from_raw(ndk_context::android_context().context().cast()) };

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

        // Create native handle (stores raw pointer to Arc for JNI callbacks)
        let native_handle = create_native_handle(Arc::clone(&state));

        // Helper to release handle on error - prevents Arc leak if initialization fails
        let release_on_error = |e: VideoError| {
            release_native_handle(native_handle);
            e
        };

        // Get the app's class loader from the context (needed for native threads)
        // Native threads don't have access to app classes via env.find_class()
        let class_loader = env
            .call_method(&context, "getClassLoader", "()Ljava/lang/ClassLoader;", &[])
            .map_err(|e| {
                release_on_error(VideoError::DecoderInit(format!(
                    "Failed to get class loader: {}",
                    e
                )))
            })?
            .l()
            .map_err(|e| {
                release_on_error(VideoError::DecoderInit(format!(
                    "Failed to get class loader object: {}",
                    e
                )))
            })?;

        // Load ExoPlayerBridge class using the app's class loader
        let class_name = env
            .new_string("com.damus.notedeck.video.ExoPlayerBridge")
            .map_err(|e| {
                release_on_error(VideoError::DecoderInit(format!(
                    "Failed to create class name string: {}",
                    e
                )))
            })?;

        let bridge_class = env
            .call_method(
                &class_loader,
                "loadClass",
                "(Ljava/lang/String;)Ljava/lang/Class;",
                &[JValue::Object(&class_name)],
            )
            .map_err(|e| {
                release_on_error(VideoError::DecoderInit(format!(
                    "Failed to load ExoPlayerBridge class: {}",
                    e
                )))
            })?
            .l()
            .map_err(|e| {
                release_on_error(VideoError::DecoderInit(format!(
                    "Failed to get ExoPlayerBridge class: {}",
                    e
                )))
            })?;

        let bridge_class = JClass::from(bridge_class);

        let bridge = env
            .new_object(
                bridge_class,
                "(Landroid/content/Context;J)V",
                &[JValue::Object(&context), JValue::Long(native_handle)],
            )
            .map_err(|e| {
                release_on_error(VideoError::DecoderInit(format!(
                    "Failed to create ExoPlayerBridge: {}",
                    e
                )))
            })?;

        // Create global reference
        let bridge_ref = env.new_global_ref(bridge).map_err(|e| {
            release_on_error(VideoError::DecoderInit(format!(
                "Failed to create global ref: {}",
                e
            )))
        })?;

        // Initialize the bridge (but don't start playback yet)
        env.call_method(&bridge_ref, "initialize", "()V", &[])
            .map_err(|e| {
                release_on_error(VideoError::DecoderInit(format!(
                    "Failed to initialize bridge: {}",
                    e
                )))
            })?;

        // Don't call play() here - playback will start when decode_next() is first called
        // This prevents auto-play on app start

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
            last_position: Duration::ZERO,
            url: url.to_string(),
            started: false,
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

        let bytes_obj = result
            .l()
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to get frame bytes: {}", e)))?;

        if bytes_obj.is_null() {
            tracing::debug!("extractCurrentFrame returned null");
            return Ok(None);
        }

        let bytes_array = JByteArray::from(bytes_obj);
        let len = env
            .get_array_length(&bytes_array)
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to get array length: {}", e)))?
            as usize;

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
            tracing::debug!("Frame dimensions are 0x0");
            return Ok(None);
        }

        // Debug: Check if we got actual pixel data
        let non_zero_count = pixels.iter().filter(|&&b| b != 0).count();
        tracing::trace!(
            "Extracted frame: {}x{}, {} bytes, {} non-zero bytes",
            width,
            height,
            len,
            non_zero_count
        );

        // Use last_position for timestamp - don't call getCurrentPosition() here
        // because ExoPlayer requires main thread access and we're on a decode thread
        let position_ms = self.last_position.as_millis() as i64;

        Ok(Some(AndroidFrame {
            pixels,
            width,
            height,
            timestamp_ns: position_ms * 1_000_000, // ms to ns
        }))
    }

    /// Creates a minimal placeholder frame with the last known playback position.
    /// This keeps the decode loop alive without resetting playback position.
    fn create_placeholder_frame(&self) -> VideoFrame {
        let placeholder = CpuFrame::new(
            PixelFormat::Rgba,
            1,
            1,
            vec![Plane {
                data: vec![0, 0, 0, 255],
                stride: 4,
            }],
        );
        VideoFrame::new(self.last_position, DecodedFrame::Cpu(placeholder))
    }

    /// Starts playback if not already started.
    fn start_playback(&mut self) -> Result<(), VideoError> {
        if self.started {
            return Ok(());
        }

        let vm = crate::platform::android::get_jvm();
        let mut env = vm
            .attach_current_thread()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to attach JNI thread: {}", e)))?;

        let url_jstring = env
            .new_string(&self.url)
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create URL string: {}", e)))?;

        env.call_method(
            &self.bridge,
            "play",
            "(Ljava/lang/String;)V",
            &[JValue::Object(&url_jstring)],
        )
        .map_err(|e| VideoError::DecoderInit(format!("Failed to start playback: {}", e)))?;

        self.started = true;
        tracing::info!("Started ExoPlayer playback for {}", self.url);
        Ok(())
    }

    /// Checks shared state for errors or EOS.
    /// Returns Some(result) if decode_next should return early, None to continue.
    fn check_state_for_early_return(&self) -> Option<Result<Option<VideoFrame>, VideoError>> {
        const STATE_ENDED: i32 = 4;

        let state = self.state.lock().unwrap();

        if let Some(ref error) = state.last_error {
            return Some(Err(VideoError::DecodeFailed(error.clone())));
        }

        if state.playback_state == STATE_ENDED {
            return Some(Ok(None));
        }

        None
    }

    /// Waits for a frame to be available with timeout.
    /// Returns true if a frame is ready, false on timeout.
    fn wait_for_frame(&self, max_wait_ms: u64) -> Result<bool, VideoError> {
        const STATE_ENDED: i32 = 4;

        let start = std::time::Instant::now();

        while start.elapsed().as_millis() < max_wait_ms as u128 {
            {
                let mut state = self.state.lock().unwrap();

                if state.frame_available {
                    state.frame_available = false;
                    return Ok(true);
                }

                if let Some(ref error) = state.last_error {
                    return Err(VideoError::DecodeFailed(error.clone()));
                }

                if state.playback_state == STATE_ENDED {
                    return Ok(false);
                }
            }

            std::thread::sleep(Duration::from_millis(5));
        }

        Ok(false)
    }

    /// Converts an AndroidFrame to a VideoFrame, updating last_position.
    fn android_frame_to_video_frame(&mut self, android_frame: AndroidFrame) -> VideoFrame {
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
        self.last_position = pts;

        VideoFrame::new(pts, DecodedFrame::Cpu(cpu_frame))
    }

    /// Checks if playback has ended.
    fn is_playback_ended(&self) -> bool {
        const STATE_ENDED: i32 = 4;
        let state = self.state.lock().unwrap();
        state.playback_state == STATE_ENDED
    }
}

impl Drop for AndroidVideoDecoder {
    fn drop(&mut self) {
        // Release the native handle (decrements Arc refcount)
        release_native_handle(self.native_handle);

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

    fn pause(&mut self) -> Result<(), VideoError> {
        if !self.started {
            return Ok(());
        }

        let vm = crate::platform::android::get_jvm();
        let mut env = vm
            .attach_current_thread()
            .map_err(|e| VideoError::Generic(format!("Failed to attach JNI thread: {}", e)))?;

        env.call_method(&self.bridge, "pause", "()V", &[])
            .map_err(|e| VideoError::Generic(format!("Failed to pause ExoPlayer: {}", e)))?;

        tracing::info!("Paused ExoPlayer playback");
        Ok(())
    }

    fn resume(&mut self) -> Result<(), VideoError> {
        if !self.started {
            return Ok(());
        }

        let vm = crate::platform::android::get_jvm();
        let mut env = vm
            .attach_current_thread()
            .map_err(|e| VideoError::Generic(format!("Failed to attach JNI thread: {}", e)))?;

        env.call_method(&self.bridge, "resume", "()V", &[])
            .map_err(|e| VideoError::Generic(format!("Failed to resume ExoPlayer: {}", e)))?;

        tracing::info!("Resumed ExoPlayer playback");
        Ok(())
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        // Start playback on first decode_next call
        self.start_playback()?;
        tracing::debug!("decode_next called, started={}", self.started);

        // Check for errors or EOS from callbacks
        if let Some(result) = self.check_state_for_early_return() {
            return result;
        }

        // Wait for a frame to be available (with timeout)
        let frame_ready = self.wait_for_frame(100)?;

        if !frame_ready {
            // EOS check is handled in wait_for_frame, this is just timeout
            tracing::debug!("No frame ready, returning placeholder");
            return Ok(Some(self.create_placeholder_frame()));
        }

        tracing::debug!("Frame ready, extracting...");
        let frame = self.extract_frame()?;

        let Some(android_frame) = frame else {
            // Frame extraction failed - check if EOS or return placeholder
            if self.is_playback_ended() {
                return Ok(None);
            }
            return Ok(Some(self.create_placeholder_frame()));
        };

        Ok(Some(self.android_frame_to_video_frame(android_frame)))
    }

    fn seek(&mut self, position: Duration) -> Result<(), VideoError> {
        let vm = crate::platform::android::get_jvm();
        let mut env = vm
            .attach_current_thread()
            .map_err(|e| VideoError::SeekFailed(format!("Failed to attach JNI thread: {}", e)))?;

        let position_ms = position.as_millis() as i64;

        env.call_method(&self.bridge, "seek", "(J)V", &[JValue::Long(position_ms)])
            .map_err(|e| VideoError::SeekFailed(format!("Seek failed: {}", e)))?;

        Ok(())
    }

    fn metadata(&self) -> &VideoMetadata {
        // Update duration from shared state if available
        let state = self.state.lock().unwrap();
        if state.duration_ms > 0 {
            // We need to return updated metadata, but can't mutate self here
            // This is a limitation - duration will be returned via get_duration() instead
        }
        drop(state);
        &self.metadata
    }

    fn hw_accel_type(&self) -> HwAccelType {
        HwAccelType::MediaCodec
    }

    fn is_eof(&self) -> bool {
        // ExoPlayer playback states (from Player.java)
        const STATE_ENDED: i32 = 4;
        let state = self.state.lock().unwrap();
        state.playback_state == STATE_ENDED
    }

    fn set_muted(&mut self, muted: bool) -> Result<(), VideoError> {
        // Call the inherent method
        AndroidVideoDecoder::set_muted(self, muted)
    }

    fn set_volume(&mut self, volume: f32) -> Result<(), VideoError> {
        // Call the inherent method
        AndroidVideoDecoder::set_volume(self, volume)
    }

    fn duration(&self) -> Option<Duration> {
        // Use the dynamic get_duration() which reads from shared state (updated by callbacks)
        self.get_duration()
    }

    fn dimensions(&self) -> (u32, u32) {
        // Read dimensions from SharedState (updated by JNI callbacks)
        let state = self.state.lock().unwrap();
        if state.width > 0 && state.height > 0 {
            (state.width, state.height)
        } else {
            // Fall back to placeholder if not yet known
            (self.metadata.width, self.metadata.height)
        }
    }
}

impl AndroidVideoDecoder {
    /// Sets the muted state for audio playback.
    pub fn set_muted(&self, muted: bool) -> Result<(), VideoError> {
        let vm = crate::platform::android::get_jvm();
        let mut env = vm
            .attach_current_thread()
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to attach JNI thread: {}", e)))?;

        env.call_method(
            &self.bridge,
            "setMuted",
            "(Z)V",
            &[JValue::Bool(muted as u8)],
        )
        .map_err(|e| VideoError::DecodeFailed(format!("setMuted failed: {}", e)))?;

        tracing::debug!("Set muted: {}", muted);
        Ok(())
    }

    /// Sets the volume for audio playback.
    pub fn set_volume(&self, volume: f32) -> Result<(), VideoError> {
        let vm = crate::platform::android::get_jvm();
        let mut env = vm
            .attach_current_thread()
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to attach JNI thread: {}", e)))?;

        env.call_method(&self.bridge, "setVolume", "(F)V", &[JValue::Float(volume)])
            .map_err(|e| VideoError::DecodeFailed(format!("setVolume failed: {}", e)))?;

        tracing::debug!("Set volume: {}", volume);
        Ok(())
    }

    /// Gets the current playback position.
    pub fn get_position(&self) -> Result<Duration, VideoError> {
        let vm = crate::platform::android::get_jvm();
        let mut env = vm
            .attach_current_thread()
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to attach JNI thread: {}", e)))?;

        let result = env
            .call_method(&self.bridge, "getCurrentPosition", "()J", &[])
            .map_err(|e| VideoError::DecodeFailed(format!("getCurrentPosition failed: {}", e)))?;

        let position_ms = result.j().unwrap_or(0);
        Ok(Duration::from_millis(position_ms as u64))
    }

    /// Gets the video duration.
    pub fn get_duration(&self) -> Option<Duration> {
        // First check the shared state (updated by callbacks)
        let state = self.state.lock().unwrap();
        if state.duration_ms > 0 {
            return Some(Duration::from_millis(state.duration_ms as u64));
        }
        drop(state);

        // Fall back to querying ExoPlayer directly
        let vm = crate::platform::android::get_jvm();
        let mut env = match vm.attach_current_thread() {
            Ok(env) => env,
            Err(_) => return None,
        };

        let result = match env.call_method(&self.bridge, "getDuration", "()J", &[]) {
            Ok(r) => r,
            Err(_) => return None,
        };

        let duration_ms = result.j().unwrap_or(0);
        if duration_ms > 0 {
            Some(Duration::from_millis(duration_ms as u64))
        } else {
            None
        }
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
    tracing::info!(
        "nativeOnVideoSizeChanged: handle={}, {}x{}",
        handle,
        width,
        height
    );
    if let Some(state) = get_native_state(handle) {
        let mut state = state.lock().unwrap();
        state.width = width as u32;
        state.height = height as u32;
        tracing::info!("Video size updated in SharedState: {}x{}", width, height);
    } else {
        tracing::warn!("nativeOnVideoSizeChanged: handle {} not found!", handle);
    }
}

#[no_mangle]
pub extern "C" fn Java_com_damus_notedeck_video_ExoPlayerBridge_nativeOnDurationChanged(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
    duration_ms: jlong,
) {
    tracing::info!(
        "nativeOnDurationChanged: handle={}, duration_ms={}",
        handle,
        duration_ms
    );
    if let Some(state) = get_native_state(handle) {
        let mut state = state.lock().unwrap();
        state.duration_ms = duration_ms;
        tracing::info!("Duration updated in SharedState: {} ms", duration_ms);
    } else {
        tracing::warn!("nativeOnDurationChanged: handle {} not found!", handle);
    }
}
