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
        mainHandler.post(() -> {
            try {
                // Create ExoPlayer
                player = new ExoPlayer.Builder(context).build();

                // Setup frame extraction via SurfaceTexture
                frameExtractor = new FrameExtractor();
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
     * Extracts the current frame as RGBA pixel data.
     * @return RGBA byte array or null if not available
     */
    public byte[] extractCurrentFrame() {
        if (frameExtractor != null && videoWidth > 0 && videoHeight > 0) {
            return frameExtractor.extractFrame();
        }
        return null;
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
        // Update the texture with the new frame
        try {
            surfaceTexture.updateTexImage();
            long timestamp = surfaceTexture.getTimestamp();

            // Notify Rust that a new frame is available
            if (videoWidth > 0 && videoHeight > 0) {
                nativeOnFrameAvailable(nativeHandle, videoWidth, videoHeight, timestamp);
            }
        } catch (Exception e) {
            Log.e(TAG, "Error updating frame", e);
        }
    }
}
