package com.damus.notedeck.video;

import android.content.Context;
import android.graphics.SurfaceTexture;
import android.net.Uri;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.Surface;

import androidx.annotation.NonNull;
import androidx.annotation.OptIn;
import androidx.media3.common.MediaItem;
import androidx.media3.common.PlaybackException;
import androidx.media3.common.Player;
import androidx.media3.common.VideoSize;
import androidx.media3.common.util.UnstableApi;
import androidx.media3.exoplayer.ExoPlayer;

/**
 * JNI bridge for ExoPlayer video playback.
 * This class manages ExoPlayer lifecycle and provides frame data to Rust.
 */
public class ExoPlayerBridge implements SurfaceTexture.OnFrameAvailableListener {
    private static final String TAG = "ExoPlayerBridge";

    private final Context context;
    private final Handler mainHandler;
    private final long nativeHandle;

    private ExoPlayer player;
    private SurfaceTexture surfaceTexture;
    private Surface surface;
    private FrameExtractor frameExtractor;

    private int videoWidth;
    private int videoHeight;
    private boolean isInitialized = false;

    // Native callbacks
    private static native void nativeOnFrameAvailable(long handle, int width, int height, long timestampNs);
    private static native void nativeOnPlaybackStateChanged(long handle, int state);
    private static native void nativeOnError(long handle, String errorMessage);
    private static native void nativeOnVideoSizeChanged(long handle, int width, int height);
    private static native void nativeOnDurationChanged(long handle, long durationMs);

    /**
     * Creates a new ExoPlayerBridge.
     * @param context Android context
     * @param nativeHandle Pointer to the Rust AndroidVideoPlayer struct
     */
    public ExoPlayerBridge(Context context, long nativeHandle) {
        this.context = context;
        this.nativeHandle = nativeHandle;
        this.mainHandler = new Handler(Looper.getMainLooper());
    }

    /**
     * Initializes the ExoPlayer and frame extraction pipeline.
     * Must be called from the main thread.
     */
    @OptIn(markerClass = UnstableApi.class)
    public void initialize() {
        Log.d(TAG, "initialize() called on thread: " + Thread.currentThread().getName());
        mainHandler.post(() -> {
            Log.d(TAG, "initialize() running on main handler");
            try {
                // Create ExoPlayer
                Log.d(TAG, "Creating ExoPlayer...");
                player = new ExoPlayer.Builder(context).build();
                Log.d(TAG, "ExoPlayer created");

                // Setup frame extraction via SurfaceTexture
                Log.d(TAG, "Creating FrameExtractor...");
                frameExtractor = new FrameExtractor();
                Log.d(TAG, "FrameExtractor created, getting SurfaceTexture...");
                surfaceTexture = frameExtractor.getSurfaceTexture();
                surfaceTexture.setOnFrameAvailableListener(this);
                surface = new Surface(surfaceTexture);

                // Set the surface for video output
                player.setVideoSurface(surface);

                // Add player listener
                player.addListener(new Player.Listener() {
                    @Override
                    public void onPlaybackStateChanged(int state) {
                        nativeOnPlaybackStateChanged(nativeHandle, state);
                    }

                    @Override
                    public void onPlayerError(@NonNull PlaybackException error) {
                        nativeOnError(nativeHandle, error.getMessage());
                    }

                    @Override
                    public void onVideoSizeChanged(@NonNull VideoSize videoSize) {
                        Log.d(TAG, "onVideoSizeChanged: " + videoSize.width + "x" + videoSize.height);
                        videoWidth = videoSize.width;
                        videoHeight = videoSize.height;
                        if (videoWidth > 0 && videoHeight > 0) {
                            surfaceTexture.setDefaultBufferSize(videoWidth, videoHeight);
                            frameExtractor.updateSize(videoWidth, videoHeight);
                            nativeOnVideoSizeChanged(nativeHandle, videoWidth, videoHeight);
                        }
                    }
                });

                isInitialized = true;
                Log.d(TAG, "ExoPlayerBridge initialized");

            } catch (Exception e) {
                Log.e(TAG, "Failed to initialize ExoPlayerBridge", e);
                nativeOnError(nativeHandle, "Initialization failed: " + e.getMessage());
            }
        });
    }

    /**
     * Starts playback of the given URL.
     * @param url Video URL (http://, https://, file://, content://)
     */
    public void play(String url) {
        mainHandler.post(() -> {
            if (!isInitialized || player == null) {
                Log.e(TAG, "Cannot play: not initialized");
                return;
            }

            try {
                MediaItem mediaItem = MediaItem.fromUri(Uri.parse(url));
                player.setMediaItem(mediaItem);
                player.prepare();
                player.play();

                // Report duration once available
                mainHandler.postDelayed(() -> {
                    if (player != null) {
                        long duration = player.getDuration();
                        if (duration > 0) {
                            nativeOnDurationChanged(nativeHandle, duration);
                        }
                    }
                }, 500);

                Log.d(TAG, "Starting playback: " + url);
            } catch (Exception e) {
                Log.e(TAG, "Failed to start playback", e);
                nativeOnError(nativeHandle, "Playback failed: " + e.getMessage());
            }
        });
    }

    /**
     * Pauses playback.
     */
    public void pause() {
        mainHandler.post(() -> {
            if (player != null) {
                player.pause();
            }
        });
    }

    /**
     * Resumes playback.
     */
    public void resume() {
        mainHandler.post(() -> {
            if (player != null) {
                player.play();
            }
        });
    }

    /**
     * Seeks to the specified position.
     * @param positionMs Position in milliseconds
     */
    public void seek(long positionMs) {
        mainHandler.post(() -> {
            if (player != null) {
                player.seekTo(positionMs);
            }
        });
    }

    /**
     * Returns the current playback position in milliseconds.
     */
    public long getCurrentPosition() {
        if (player != null) {
            return player.getCurrentPosition();
        }
        return 0;
    }

    /**
     * Returns the video duration in milliseconds.
     */
    public long getDuration() {
        if (player != null) {
            return player.getDuration();
        }
        return 0;
    }

    /**
     * Returns true if currently playing.
     */
    public boolean isPlaying() {
        return player != null && player.isPlaying();
    }

    /**
     * Sets the muted state.
     * @param muted true to mute, false to unmute
     */
    public void setMuted(boolean muted) {
        mainHandler.post(() -> {
            if (player != null) {
                player.setVolume(muted ? 0f : 1f);
                Log.d(TAG, "setMuted: " + muted);
            }
        });
    }

    /**
     * Sets the volume.
     * @param volume Volume from 0.0 to 1.0
     */
    public void setVolume(float volume) {
        mainHandler.post(() -> {
            if (player != null) {
                player.setVolume(volume);
                Log.d(TAG, "setVolume: " + volume);
            }
        });
    }

    /**
     * Extracts the current frame as RGBA pixel data.
     * @return RGBA byte array or null if not available
     */
    public byte[] extractCurrentFrame() {
        if (!isInitialized) {
            Log.d(TAG, "extractCurrentFrame: not initialized yet");
            return null;
        }
        if (frameExtractor == null) {
            Log.d(TAG, "extractCurrentFrame: frameExtractor is null");
            return null;
        }
        Log.d(TAG, "extractCurrentFrame: calling extractFrame, width=" + videoWidth + ", height=" + videoHeight);
        byte[] result = frameExtractor.extractFrame();
        if (result == null) {
            Log.d(TAG, "extractCurrentFrame: extractFrame returned null");
        } else {
            Log.d(TAG, "extractCurrentFrame: got " + result.length + " bytes");
        }
        return result;
    }

    /**
     * Releases all resources.
     */
    public void release() {
        mainHandler.post(() -> {
            isInitialized = false;

            if (player != null) {
                player.release();
                player = null;
            }

            if (surface != null) {
                surface.release();
                surface = null;
            }

            if (frameExtractor != null) {
                frameExtractor.release();
                frameExtractor = null;
            }

            surfaceTexture = null;

            Log.d(TAG, "ExoPlayerBridge released");
        });
    }

    @Override
    public void onFrameAvailable(SurfaceTexture surfaceTexture) {
        // Don't call updateTexImage() here - it must be called from the thread
        // that owns the EGL context (the extractFrame thread).
        // Just notify Rust that a new frame is available.
        Log.d(TAG, "onFrameAvailable: width=" + videoWidth + ", height=" + videoHeight);
        if (videoWidth > 0 && videoHeight > 0) {
            nativeOnFrameAvailable(nativeHandle, videoWidth, videoHeight, System.nanoTime());
        } else {
            // Use default size if we haven't received the actual size yet
            Log.w(TAG, "onFrameAvailable: using default size 1920x1080");
            nativeOnFrameAvailable(nativeHandle, 1920, 1080, System.nanoTime());
        }
    }
}
