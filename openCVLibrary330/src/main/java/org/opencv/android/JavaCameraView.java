package org.opencv.android;

import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.hardware.Camera.PreviewCallback;
import android.os.Build;
import android.util.AttributeSet;
import android.util.Log;
import android.view.ViewGroup;

import org.opencv.BuildConfig;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.List;

/**
 * This class is an implementation of the Bridge View between OpenCV and Java Camera.
 * This class relays on the functionality available in base class and only implements
 * required functions:
 * connectCamera - opens Java camera and sets the PreviewCallback to be delivered.
 * disconnectCamera - closes the camera and stops preview.
 * When frame is delivered via callback from Camera - it processed via OpenCV to be
 * converted to RGBA32 and then passed to the external callback for modifications if required.
 */
public class JavaCameraView extends CameraBridgeViewBase implements PreviewCallback {

    private static final int MAGIC_TEXTURE_ID = 10;
    private static final String TAG = "JavaCameraView";

    private byte mBuffer[];
    private Mat[] mFrameChain;
    private int mChainIdx = 0;
    private Thread mThread;
    private boolean mStopThread;

    protected Camera mCamera;
    protected JavaCameraFrame[] mCameraFrame;
    private SurfaceTexture mSurfaceTexture;
    private int mPreviewFormat = ImageFormat.NV21;
    private boolean isFrontCamera = false;

    public static class JavaCameraSizeAccessor implements ListItemAccessor {

        @Override
        public int getWidth(Object obj) {
            Camera.Size size = (Camera.Size) obj;
            return size.width;
        }

        @Override
        public int getHeight(Object obj) {
            Camera.Size size = (Camera.Size) obj;
            return size.height;
        }
    }

    public JavaCameraView(Context context, int cameraId) {
        super(context, cameraId);
    }

    public JavaCameraView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public boolean initializeCamera(int width, int height) {
        Log.d(TAG, "Initialize java camera");
        boolean result = true;
        synchronized (this) {
            mCamera = null;
            if (mCameraIndex == CAMERA_ID_ANY) {
                Log.d(TAG, "Trying to open camera with old open()");
                try {
                    mCamera = Camera.open();
                    isFrontCamera = false;
                } catch (Exception e) {
                    Log.e(TAG, "Camera is not available (in use or does not exist): " + e.getLocalizedMessage());
                }

                if (mCamera == null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.GINGERBREAD) {
                    boolean connected = false;
                    for (int camIdx = 0; camIdx < Camera.getNumberOfCameras(); ++camIdx) {
                        Log.d(TAG, "Trying to open camera with new open(" + Integer.valueOf(camIdx) + ")");
                        try {
                            mCamera = Camera.open(camIdx);
                            if (camIdx == CAMERA_ID_BACK) {
                                isFrontCamera = false;
                            } else if (camIdx == CAMERA_ID_FRONT) {
                                isFrontCamera = true;
                            }
                            connected = true;
                        } catch (RuntimeException e) {
                            Log.e(TAG, "Camera #" + camIdx + "failed to open: " + e.getLocalizedMessage());
                        }
                        if (connected) {
                            break;
                        }
                    }
                }
            } else {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.GINGERBREAD) {
                    int localCameraIndex = mCameraIndex;
                    if (mCameraIndex == CAMERA_ID_BACK) {
                        Log.i(TAG, "Trying to open back camera");
                        Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
                        for (int camIdx = 0; camIdx < Camera.getNumberOfCameras(); ++camIdx) {
                            Camera.getCameraInfo(camIdx, cameraInfo);
                            if (cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_BACK) {
                                localCameraIndex = camIdx;
                                isFrontCamera = false;
                                break;
                            }
                        }
                    } else if (mCameraIndex == CAMERA_ID_FRONT) {
                        Log.i(TAG, "Trying to open front camera");
                        Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
                        for (int camIdx = 0; camIdx < Camera.getNumberOfCameras(); ++camIdx) {
                            Camera.getCameraInfo(camIdx, cameraInfo);
                            if (cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                                localCameraIndex = camIdx;
                                isFrontCamera = true;
                                break;
                            }
                        }
                    }
                    if (localCameraIndex == CAMERA_ID_BACK) {
                        Log.e(TAG, "Back camera not found!");
                    } else if (localCameraIndex == CAMERA_ID_FRONT) {
                        Log.e(TAG, "Front camera not found!");
                    } else {
                        Log.d(TAG, "Trying to open camera with new open(" + Integer.valueOf(localCameraIndex) + ")");
                        try {
                            mCamera = Camera.open(localCameraIndex);
                        } catch (RuntimeException e) {
                            Log.e(TAG, "Camera #" + localCameraIndex + "failed to open: " + e.getLocalizedMessage());
                        }
                    }
                }
            }

            if (mCamera == null) {
                return false;
            }
            /* Now set camera parameters */
            try {
                Camera.Parameters params = mCamera.getParameters();
                Log.d(TAG, "getSupportedPreviewSizes()");
                List<Camera.Size> sizes = params.getSupportedPreviewSizes();

                if (sizes != null) {
                    /* Select the size that fits surface considering maximum size allowed */
                    org.opencv.core.Size frameSize = calculateCameraFrameSize(sizes, new JavaCameraSizeAccessor(), width, height);

                    /* Image format NV21 causes issues in the Android emulators */
                    if (Build.FINGERPRINT.startsWith("generic")
                            || Build.FINGERPRINT.startsWith("unknown")
                            || Build.MODEL.contains("google_sdk")
                            || Build.MODEL.contains("Emulator")
                            || Build.MODEL.contains("Android SDK built for x86")
                            || Build.MANUFACTURER.contains("Genymotion")
                            || (Build.BRAND.startsWith("generic") && Build.DEVICE.startsWith("generic"))
                            || "google_sdk".equals(Build.PRODUCT)) {
                        params.setPreviewFormat(ImageFormat.YV12);
                    } // "generic" or "android" = android emulator
                    else {
                        params.setPreviewFormat(ImageFormat.NV21);
                    }
                    mPreviewFormat = params.getPreviewFormat();

                    Log.d(TAG, "Set preview Format to " + mPreviewFormat);

                    Log.d(TAG, "Set preview size to " + Integer.valueOf((int) frameSize.width) + "x" + Integer.valueOf((int) frameSize.height));
                    params.setPreviewSize((int) frameSize.width, (int) frameSize.height);

                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.ICE_CREAM_SANDWICH && !android.os.Build.MODEL.equals("GT-I9100")) {
                        params.setRecordingHint(true);
                    }
                    List<String> FocusModes = params.getSupportedFocusModes();
                    if (FocusModes != null && FocusModes.contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO)) {
                        params.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
                    }
                    mCamera.setParameters(params);
                    params = mCamera.getParameters();

                    mFrameWidth = params.getPreviewSize().width;
                    mFrameHeight = params.getPreviewSize().height;
                    if ((getLayoutParams().width == ViewGroup.LayoutParams.MATCH_PARENT) && (getLayoutParams().height == ViewGroup.LayoutParams.MATCH_PARENT)) {
                        mScale = Math.min(((float) height) / mFrameHeight, ((float) width) / mFrameWidth);
                    } else {
                        mScale = 0;
                    }

                    if (mFpsMeter != null) {
                        mFpsMeter.setResolution(mFrameWidth, mFrameHeight);
                    }

                    int size = mFrameWidth * mFrameHeight;
                    size = size * ImageFormat.getBitsPerPixel(params.getPreviewFormat()) / 8;
                    mBuffer = new byte[size];

                    mCamera.addCallbackBuffer(mBuffer);
                    mCamera.setPreviewCallbackWithBuffer(this);

                    mFrameChain = new Mat[2];
                    mFrameChain[0] = new Mat(mFrameHeight + (mFrameHeight / 2), mFrameWidth, CvType.CV_8UC1);
                    mFrameChain[1] = new Mat(mFrameHeight + (mFrameHeight / 2), mFrameWidth, CvType.CV_8UC1);

                    AllocateCache();

                    mCameraFrame = new JavaCameraView.JavaCameraFrame[2];
                    mCameraFrame[0] = new JavaCameraView.JavaCameraFrame(mFrameChain[0], mFrameWidth, mFrameHeight);
                    mCameraFrame[1] = new JavaCameraView.JavaCameraFrame(mFrameChain[1], mFrameWidth, mFrameHeight);

                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.HONEYCOMB) {
                        mSurfaceTexture = new SurfaceTexture(MAGIC_TEXTURE_ID);
                        mCamera.setPreviewTexture(mSurfaceTexture);
                    } else {
                        mCamera.setPreviewDisplay(null);
                    }
                    /* Finally we are ready to start the preview */
                    Log.d(TAG, "startPreview");
                    mCamera.startPreview();
                } else {
                    result = false;
                }
            } catch (Exception e) {
                result = false;
                e.printStackTrace();
            }
        }
        return result;
    }

    protected void releaseCamera() {
        synchronized (this) {
            if (mCamera != null) {
                try {
                    mCamera.setPreviewDisplay(null);
                    mCamera.setPreviewCallbackWithBuffer(null);
                    mCamera.stopPreview();
                    mCamera.release();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            mCamera = null;
            if (mFrameChain != null) {
                mFrameChain[0].release();
                mFrameChain[1].release();
            }
            if (mCameraFrame != null) {
                mCameraFrame[0].release();
                mCameraFrame[1].release();
            }
        }
    }

    private boolean mCameraFrameReady = false;

    @Override
    protected boolean connectCamera(int width, int height) {

        /* 1. We need to instantiate camera
         * 2. We need to start thread which will be getting frames
         */
        /* First step - initialize camera connection */
        Log.d(TAG, "Connecting to camera");
        if (!initializeCamera(width, height)) {
            return false;
        }
        mCameraFrameReady = false;

        /* now we can start update thread */
        Log.d(TAG, "Starting processing thread");
        mStopThread = false;
        mThread = new Thread(new CameraWorker());
        mThread.start();

        return true;
    }

    @Override
    protected void disconnectCamera() {
        /* 1. We need to stop thread which updating the frames
         * 2. Stop camera and release it
         */
        Log.d(TAG, "Disconnecting from camera");
        try {
            mStopThread = true;
            Log.d(TAG, "Notify thread");
            synchronized (this) {
                this.notify();
            }
            Log.d(TAG, "Waiting for thread");
            if (mThread != null) {
                mThread.join();
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            mThread = null;
        }

        /* Now release camera */
        releaseCamera();

        mCameraFrameReady = false;
    }

    @Override
    public boolean switchCamera() {
        int numberOfCameras = 0;
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.GINGERBREAD) {
            numberOfCameras = Camera.getNumberOfCameras();
        }
        // 2个及以上摄像头
        if (2 > numberOfCameras) {
            return false;
        }
        if (isFrontCamera) {
            mCameraIndex = CAMERA_ID_BACK;
        } else {
            mCameraIndex = CAMERA_ID_FRONT;
        }
        disconnectCamera();
        if (connectCamera(getWidth(), getHeight())) {
            return true;
        } else {
            return false;
        }
    }

    @Override
    public void onPreviewFrame(byte[] frame, Camera arg1) {
        if (BuildConfig.DEBUG) {
            Log.d(TAG, "Preview Frame received. Frame size: " + frame.length);
        }
        synchronized (this) {
            mFrameChain[mChainIdx].put(0, 0, frame);
            mCameraFrameReady = true;
            this.notify();
        }
        if (mCamera != null) {
            mCamera.addCallbackBuffer(mBuffer);
        }
    }

    private class JavaCameraFrame implements CvCameraViewFrame {

        @Override
        public Mat gray() {
            mGray = mYuvFrameData.submat(0, mHeight, 0, mWidth);
            if (isFrontCamera) {
                flip(mGray);
            }
            return mGray;
        }

        @Override
        public Mat rgba() {
            if (mPreviewFormat == ImageFormat.NV21) {
                Imgproc.cvtColor(mYuvFrameData, mRgba, Imgproc.COLOR_YUV2RGBA_NV21, 4);
            } else if (mPreviewFormat == ImageFormat.YV12) {
                Imgproc.cvtColor(mYuvFrameData, mRgba, Imgproc.COLOR_YUV2RGB_I420, 4);  // COLOR_YUV2RGBA_YV12 produces inverted colors
            } else {
                throw new IllegalArgumentException("Preview Format can be NV21 or YV12");
            }
            if (isFrontCamera) {
                flip(mRgba);
            }
            return mRgba;
        }

        public JavaCameraFrame(Mat Yuv420sp, int width, int height) {
            super();
            mWidth = width;
            mHeight = height;
            mYuvFrameData = Yuv420sp;
            mRgba = new Mat();
            mGray = new Mat();
        }

        public void release() {
            mRgba.release();
            mGray.release();
        }

        private Mat mYuvFrameData;
        private Mat mRgba;
        private Mat mGray;
        private int mWidth;
        private int mHeight;
    }

    /**
     * 对图像进行翻转
     *
     * @param mat
     * @return
     */
    @Override
    public Mat flip(Mat mat) {
        //任意正数代表y轴镜像旋转
        Core.flip(mat, mat, 1);
        //0代表x轴镜像旋转
//        Core.flip(mat, mat, 0);
        //任意负数代表x和y轴同时镜像旋转
//        Core.flip(mat, mat, -1);
        return mat;
    }

    /**
     * 对图片进行旋转
     *
     * @param mat
     * @return
     */
    @Override
    public Mat transpose(Mat mat) {
        Mat mat1 = new Mat();
        Core.transpose(mat, mat1);
        Core.flip(mat1, mat1, 1);
        mat1.copyTo(mat);
        mat1.release();
        return mat;
    }

    private class CameraWorker implements Runnable {

        @Override
        public void run() {
            do {
                boolean hasFrame = false;
                synchronized (JavaCameraView.this) {
                    try {
                        while (!mCameraFrameReady && !mStopThread) {
                            JavaCameraView.this.wait();
                        }
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    if (mCameraFrameReady) {
                        mChainIdx = 1 - mChainIdx;
                        mCameraFrameReady = false;
                        hasFrame = true;
                    }
                }

                if (!mStopThread && hasFrame) {
                    if (!mFrameChain[1 - mChainIdx].empty()) {
                        deliverAndDrawFrame(mCameraFrame[1 - mChainIdx]);
                    }
                }
            } while (!mStopThread);
            Log.d(TAG, "Finish processing thread");
        }
    }
}
