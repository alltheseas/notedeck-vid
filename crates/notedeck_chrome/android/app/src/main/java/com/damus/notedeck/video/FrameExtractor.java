package com.damus.notedeck.video;

import android.graphics.SurfaceTexture;
import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLSurface;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * Extracts video frames from SurfaceTexture as RGBA pixel data.
 *
 * This class creates an offscreen EGL context and renders the external OES texture
 * (from SurfaceTexture) to a framebuffer, then reads back the pixels.
 *
 * THREADING: The SurfaceTexture is created without an EGL context and can receive
 * frames from any thread (e.g., main thread where ExoPlayer runs). The EGL context
 * is created lazily on the first extractFrame() call, ensuring it's created on the
 * same thread that will use it.
 */
public class FrameExtractor {
    private static final String TAG = "FrameExtractor";

    // Vertex shader - applies texture transform matrix from SurfaceTexture
    private static final String VERTEX_SHADER =
            "attribute vec4 aPosition;\n" +
            "attribute vec2 aTexCoord;\n" +
            "uniform mat4 uTexMatrix;\n" +
            "varying vec2 vTexCoord;\n" +
            "void main() {\n" +
            "    gl_Position = aPosition;\n" +
            "    vTexCoord = (uTexMatrix * vec4(aTexCoord, 0.0, 1.0)).xy;\n" +
            "}\n";

    // Fragment shader - converts OES texture to regular RGBA
    private static final String FRAGMENT_SHADER =
            "#extension GL_OES_EGL_image_external : require\n" +
            "precision mediump float;\n" +
            "varying vec2 vTexCoord;\n" +
            "uniform samplerExternalOES sTexture;\n" +
            "void main() {\n" +
            "    gl_FragColor = texture2D(sTexture, vTexCoord);\n" +
            "}\n";

    // Fullscreen quad vertices
    private static final float[] VERTICES = {
            -1.0f, -1.0f,  // Bottom left
             1.0f, -1.0f,  // Bottom right
            -1.0f,  1.0f,  // Top left
             1.0f,  1.0f,  // Top right
    };

    // Texture coordinates (flipped Y to correct OpenGL vs screen coordinate mismatch)
    private static final float[] TEX_COORDS = {
            0.0f, 1.0f,  // Bottom left
            1.0f, 1.0f,  // Bottom right
            0.0f, 0.0f,  // Top left
            1.0f, 0.0f,  // Top right
    };

    private EGLDisplay eglDisplay = EGL14.EGL_NO_DISPLAY;
    private EGLContext eglContext = EGL14.EGL_NO_CONTEXT;
    private EGLSurface eglSurface = EGL14.EGL_NO_SURFACE;
    private EGLConfig eglConfig;

    private int oesTextureId;
    private SurfaceTexture surfaceTexture;

    private int program;
    private int aPositionLoc;
    private int aTexCoordLoc;
    private int sTextureLoc;
    private int uTexMatrixLoc;

    private float[] texMatrix = new float[16];

    private int framebuffer;
    private int renderTexture;

    // Track old GL resources for deferred deletion (can't delete on wrong thread)
    private int pendingDeleteFramebuffer = 0;
    private int pendingDeleteRenderTexture = 0;

    private int width = 1920;
    private int height = 1080;

    private FloatBuffer vertexBuffer;
    private FloatBuffer texCoordBuffer;
    private ByteBuffer pixelBuffer;

    private boolean glInitialized = false;
    private long initThreadId = -1;

    /**
     * Creates a new FrameExtractor.
     * Only creates the SurfaceTexture here - it can receive frames without an EGL context.
     * EGL/GL initialization is deferred to first extractFrame() call.
     */
    public FrameExtractor() {
        Log.d(TAG, "FrameExtractor constructor on thread: " + Thread.currentThread().getName());

        // Create a dummy EGL context just to create the SurfaceTexture
        // This is a temporary context that we'll release immediately
        EGLDisplay tempDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        if (tempDisplay == EGL14.EGL_NO_DISPLAY) {
            throw new RuntimeException("Unable to get EGL display");
        }

        int[] version = new int[2];
        if (!EGL14.eglInitialize(tempDisplay, version, 0, version, 1)) {
            throw new RuntimeException("Unable to initialize EGL");
        }

        int[] configAttribs = {
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 8,
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
                EGL14.EGL_NONE
        };

        EGLConfig[] configs = new EGLConfig[1];
        int[] numConfigs = new int[1];
        if (!EGL14.eglChooseConfig(tempDisplay, configAttribs, 0, configs, 0, 1, numConfigs, 0)) {
            throw new RuntimeException("Unable to choose EGL config");
        }

        int[] contextAttribs = {
                EGL14.EGL_CONTEXT_CLIENT_VERSION, 2,
                EGL14.EGL_NONE
        };
        EGLContext tempContext = EGL14.eglCreateContext(tempDisplay, configs[0], EGL14.EGL_NO_CONTEXT, contextAttribs, 0);
        if (tempContext == EGL14.EGL_NO_CONTEXT) {
            throw new RuntimeException("Unable to create EGL context");
        }

        int[] surfaceAttribs = {
                EGL14.EGL_WIDTH, 1,
                EGL14.EGL_HEIGHT, 1,
                EGL14.EGL_NONE
        };
        EGLSurface tempSurface = EGL14.eglCreatePbufferSurface(tempDisplay, configs[0], surfaceAttribs, 0);
        if (tempSurface == EGL14.EGL_NO_SURFACE) {
            throw new RuntimeException("Unable to create EGL PBuffer surface");
        }

        // Make temp context current
        if (!EGL14.eglMakeCurrent(tempDisplay, tempSurface, tempSurface, tempContext)) {
            throw new RuntimeException("Unable to make EGL context current");
        }

        // Create OES texture
        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        oesTextureId = textures[0];

        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTextureId);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        // Create SurfaceTexture attached to our texture
        surfaceTexture = new SurfaceTexture(oesTextureId);
        surfaceTexture.setDefaultBufferSize(width, height);
        Log.d(TAG, "SurfaceTexture created with texture " + oesTextureId);

        // Now detach the SurfaceTexture so we can reattach on the extract thread
        surfaceTexture.detachFromGLContext();
        Log.d(TAG, "SurfaceTexture detached from temp context");

        // Clean up temp context
        EGL14.eglMakeCurrent(tempDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT);
        EGL14.eglDestroySurface(tempDisplay, tempSurface);
        EGL14.eglDestroyContext(tempDisplay, tempContext);
        EGL14.eglTerminate(tempDisplay);

        Log.d(TAG, "FrameExtractor constructor complete");
    }

    private void initEgl() {
        Log.d(TAG, "initEgl on thread: " + Thread.currentThread().getName());

        // Get default display
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) {
            throw new RuntimeException("Unable to get EGL display");
        }

        // Initialize EGL
        int[] version = new int[2];
        if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) {
            throw new RuntimeException("Unable to initialize EGL");
        }
        Log.d(TAG, "EGL initialized, version: " + version[0] + "." + version[1]);

        // Choose config
        int[] configAttribs = {
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 8,
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
                EGL14.EGL_NONE
        };

        EGLConfig[] configs = new EGLConfig[1];
        int[] numConfigs = new int[1];
        if (!EGL14.eglChooseConfig(eglDisplay, configAttribs, 0, configs, 0, 1, numConfigs, 0)) {
            throw new RuntimeException("Unable to choose EGL config");
        }
        if (numConfigs[0] == 0) {
            throw new RuntimeException("No EGL configs found");
        }
        eglConfig = configs[0];

        // Create context
        int[] contextAttribs = {
                EGL14.EGL_CONTEXT_CLIENT_VERSION, 2,
                EGL14.EGL_NONE
        };
        eglContext = EGL14.eglCreateContext(eglDisplay, eglConfig, EGL14.EGL_NO_CONTEXT, contextAttribs, 0);
        if (eglContext == EGL14.EGL_NO_CONTEXT) {
            throw new RuntimeException("Unable to create EGL context: " + EGL14.eglGetError());
        }

        // Create PBuffer surface
        int[] surfaceAttribs = {
                EGL14.EGL_WIDTH, 1,
                EGL14.EGL_HEIGHT, 1,
                EGL14.EGL_NONE
        };
        eglSurface = EGL14.eglCreatePbufferSurface(eglDisplay, eglConfig, surfaceAttribs, 0);
        if (eglSurface == EGL14.EGL_NO_SURFACE) {
            throw new RuntimeException("Unable to create EGL PBuffer surface: " + EGL14.eglGetError());
        }

        // Make context current
        if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
            throw new RuntimeException("Unable to make EGL context current: " + EGL14.eglGetError());
        }

        Log.d(TAG, "EGL initialized on thread: " + Thread.currentThread().getName());
    }

    private void initGl() {
        Log.d(TAG, "initGl starting...");

        // Create vertex buffer
        vertexBuffer = ByteBuffer.allocateDirect(VERTICES.length * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
                .put(VERTICES);
        vertexBuffer.position(0);

        // Create texture coordinate buffer
        texCoordBuffer = ByteBuffer.allocateDirect(TEX_COORDS.length * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
                .put(TEX_COORDS);
        texCoordBuffer.position(0);

        // Create shader program
        program = createProgram(VERTEX_SHADER, FRAGMENT_SHADER);
        aPositionLoc = GLES20.glGetAttribLocation(program, "aPosition");
        aTexCoordLoc = GLES20.glGetAttribLocation(program, "aTexCoord");
        sTextureLoc = GLES20.glGetUniformLocation(program, "sTexture");
        uTexMatrixLoc = GLES20.glGetUniformLocation(program, "uTexMatrix");
        Log.d(TAG, "Shader program created, locations: pos=" + aPositionLoc +
            " tex=" + aTexCoordLoc + " sampler=" + sTextureLoc + " matrix=" + uTexMatrixLoc);

        // Create new OES texture for this context
        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        int newOesTextureId = textures[0];

        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, newOesTextureId);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        // Attach the SurfaceTexture to this new texture
        try {
            surfaceTexture.attachToGLContext(newOesTextureId);
            oesTextureId = newOesTextureId;
            Log.d(TAG, "SurfaceTexture attached to texture " + oesTextureId);
        } catch (Exception e) {
            Log.e(TAG, "Failed to attach SurfaceTexture", e);
            throw new RuntimeException("Failed to attach SurfaceTexture: " + e.getMessage());
        }

        // Create framebuffer and render texture
        createFramebuffer();

        glInitialized = true;
        initThreadId = Thread.currentThread().getId();
        Log.d(TAG, "initGl complete on thread " + initThreadId);
    }

    private void createFramebuffer() {
        Log.d(TAG, "createFramebuffer: " + width + "x" + height);

        // Delete old resources if they exist
        if (framebuffer != 0) {
            int[] fb = {framebuffer};
            GLES20.glDeleteFramebuffers(1, fb, 0);
        }
        if (renderTexture != 0) {
            int[] tex = {renderTexture};
            GLES20.glDeleteTextures(1, tex, 0);
        }

        // Create render texture
        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        renderTexture = textures[0];

        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, renderTexture);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, width, height, 0, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);

        // Create framebuffer
        int[] framebuffers = new int[1];
        GLES20.glGenFramebuffers(1, framebuffers, 0);
        framebuffer = framebuffers[0];

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, framebuffer);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, renderTexture, 0);

        int status = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER);
        if (status != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            throw new RuntimeException("Framebuffer incomplete: " + status);
        }

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);

        // Create pixel buffer
        pixelBuffer = ByteBuffer.allocateDirect(width * height * 4);
        pixelBuffer.order(ByteOrder.nativeOrder());

        Log.d(TAG, "Framebuffer created successfully");
    }

    /**
     * Returns the SurfaceTexture for video output.
     */
    public SurfaceTexture getSurfaceTexture() {
        return surfaceTexture;
    }

    /**
     * Updates the frame size.
     */
    public synchronized void updateSize(int newWidth, int newHeight) {
        if (newWidth != width || newHeight != height) {
            Log.d(TAG, "updateSize: " + width + "x" + height + " -> " + newWidth + "x" + newHeight);
            width = newWidth;
            height = newHeight;
            surfaceTexture.setDefaultBufferSize(width, height);

            // Mark that framebuffer needs recreation (will happen in extractFrame)
            if (framebuffer != 0 && glInitialized) {
                // We can't delete GL resources here since we might be on wrong thread
                // Save old handles for deferred deletion in extractFrame
                pendingDeleteFramebuffer = framebuffer;
                pendingDeleteRenderTexture = renderTexture;
                framebuffer = 0;
                renderTexture = 0;
                pixelBuffer = null;
            }
        }
    }

    /**
     * Extracts the current frame as RGBA pixel data.
     * @return RGBA byte array with dimensions width x height
     */
    public synchronized byte[] extractFrame() {
        // Initialize EGL/GL on first call (on the calling thread)
        if (!glInitialized) {
            try {
                Log.d(TAG, "extractFrame: first call, initializing GL...");
                initEgl();
                initGl();
            } catch (Exception e) {
                Log.e(TAG, "Failed to initialize EGL/GL", e);
                return null;
            }
        }

        // Verify we're on the same thread that initialized GL
        long currentThread = Thread.currentThread().getId();
        if (currentThread != initThreadId) {
            Log.e(TAG, "extractFrame called from different thread! init=" + initThreadId + " current=" + currentThread);
            // Try to make context current anyway
        }

        // Ensure framebuffer exists (might have been deleted by updateSize)
        if (framebuffer == 0) {
            try {
                // Make sure context is current before creating framebuffer
                if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
                    Log.e(TAG, "Failed to make EGL context current for framebuffer creation");
                    return null;
                }

                // Delete old GL resources that were deferred from updateSize
                if (pendingDeleteFramebuffer != 0) {
                    int[] fb = {pendingDeleteFramebuffer};
                    GLES20.glDeleteFramebuffers(1, fb, 0);
                    pendingDeleteFramebuffer = 0;
                    Log.d(TAG, "Deleted old framebuffer");
                }
                if (pendingDeleteRenderTexture != 0) {
                    int[] tex = {pendingDeleteRenderTexture};
                    GLES20.glDeleteTextures(1, tex, 0);
                    pendingDeleteRenderTexture = 0;
                    Log.d(TAG, "Deleted old render texture");
                }

                createFramebuffer();
            } catch (Exception e) {
                Log.e(TAG, "Failed to create framebuffer", e);
                return null;
            }
        }

        // Make sure our context is current
        if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
            int error = EGL14.eglGetError();
            Log.e(TAG, "Failed to make EGL context current, error: " + error);
            return null;
        }

        try {
            // Update the texture with latest frame
            surfaceTexture.updateTexImage();

            // Get the texture transform matrix - this is required for correct sampling
            surfaceTexture.getTransformMatrix(texMatrix);

            // Bind framebuffer
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, framebuffer);
            GLES20.glViewport(0, 0, width, height);

            // Clear to magenta for debugging (if we see magenta, the shader didn't run)
            GLES20.glClearColor(1.0f, 0.0f, 1.0f, 1.0f);
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);

            // Use shader program
            GLES20.glUseProgram(program);

            // Set texture transform matrix
            GLES20.glUniformMatrix4fv(uTexMatrixLoc, 1, false, texMatrix, 0);

            // Bind OES texture
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTextureId);
            GLES20.glUniform1i(sTextureLoc, 0);

            // Set vertex attributes
            GLES20.glEnableVertexAttribArray(aPositionLoc);
            GLES20.glVertexAttribPointer(aPositionLoc, 2, GLES20.GL_FLOAT, false, 0, vertexBuffer);

            GLES20.glEnableVertexAttribArray(aTexCoordLoc);
            GLES20.glVertexAttribPointer(aTexCoordLoc, 2, GLES20.GL_FLOAT, false, 0, texCoordBuffer);

            // Draw
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

            // Check for GL errors
            int error = GLES20.glGetError();
            if (error != GLES20.GL_NO_ERROR) {
                Log.e(TAG, "GL error after draw: " + error);
            }

            // Finish rendering before reading
            GLES20.glFinish();

            // Read pixels
            pixelBuffer.position(0);
            GLES20.glReadPixels(0, 0, width, height, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, pixelBuffer);

            error = GLES20.glGetError();
            if (error != GLES20.GL_NO_ERROR) {
                Log.e(TAG, "GL error after readPixels: " + error);
            }

            // Unbind
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
            GLES20.glDisableVertexAttribArray(aPositionLoc);
            GLES20.glDisableVertexAttribArray(aTexCoordLoc);

            // Copy to byte array
            byte[] pixels = new byte[width * height * 4];
            pixelBuffer.position(0);
            pixelBuffer.get(pixels);

            // Debug: check if we got actual pixel data
            int nonZeroR = 0, nonZeroG = 0, nonZeroB = 0, nonZeroA = 0;
            int sampleSize = Math.min(pixels.length / 4, 10000);
            for (int i = 0; i < sampleSize; i++) {
                int offset = i * 4;
                if ((pixels[offset] & 0xFF) != 0) nonZeroR++;
                if ((pixels[offset + 1] & 0xFF) != 0) nonZeroG++;
                if ((pixels[offset + 2] & 0xFF) != 0) nonZeroB++;
                if ((pixels[offset + 3] & 0xFF) != 0) nonZeroA++;
            }
            Log.d(TAG, "extractFrame: " + width + "x" + height +
                ", samples=" + sampleSize +
                ", R=" + nonZeroR + " G=" + nonZeroG + " B=" + nonZeroB + " A=" + nonZeroA);

            return pixels;

        } catch (Exception e) {
            Log.e(TAG, "Error extracting frame", e);
            return null;
        }
    }

    /**
     * Releases all resources.
     */
    public synchronized void release() {
        Log.d(TAG, "release called");
        glInitialized = false;

        if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
            EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT);

            if (eglSurface != EGL14.EGL_NO_SURFACE) {
                EGL14.eglDestroySurface(eglDisplay, eglSurface);
            }
            if (eglContext != EGL14.EGL_NO_CONTEXT) {
                EGL14.eglDestroyContext(eglDisplay, eglContext);
            }
            EGL14.eglTerminate(eglDisplay);
        }

        eglDisplay = EGL14.EGL_NO_DISPLAY;
        eglContext = EGL14.EGL_NO_CONTEXT;
        eglSurface = EGL14.EGL_NO_SURFACE;

        if (surfaceTexture != null) {
            surfaceTexture.release();
            surfaceTexture = null;
        }

        Log.d(TAG, "FrameExtractor released");
    }

    private int createProgram(String vertexSource, String fragmentSource) {
        int vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexSource);
        int fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentSource);

        int program = GLES20.glCreateProgram();
        GLES20.glAttachShader(program, vertexShader);
        GLES20.glAttachShader(program, fragmentShader);
        GLES20.glLinkProgram(program);

        int[] linkStatus = new int[1];
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linkStatus, 0);
        if (linkStatus[0] != GLES20.GL_TRUE) {
            String error = GLES20.glGetProgramInfoLog(program);
            GLES20.glDeleteProgram(program);
            throw new RuntimeException("Program link failed: " + error);
        }

        Log.d(TAG, "Shader program linked successfully");
        return program;
    }

    private int loadShader(int type, String source) {
        int shader = GLES20.glCreateShader(type);
        GLES20.glShaderSource(shader, source);
        GLES20.glCompileShader(shader);

        int[] compileStatus = new int[1];
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compileStatus, 0);
        if (compileStatus[0] != GLES20.GL_TRUE) {
            String error = GLES20.glGetShaderInfoLog(shader);
            GLES20.glDeleteShader(shader);
            throw new RuntimeException("Shader compile failed: " + error);
        }

        return shader;
    }
}
