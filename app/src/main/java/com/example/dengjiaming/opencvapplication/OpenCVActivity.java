package com.example.dengjiaming.opencvapplication;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Environment;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.dengjiaming.opencvapplication.util.DetectionBasedTracker;
import com.example.dengjiaming.opencvapplication.util.FaceUtil;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

/**
 * @author djm
 * @date 2018/2/1.
 */
public class OpenCVActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static final int JAVA_DETECTOR = 0;
    private static final int NATIVE_DETECTOR = 1;
    private static final String TAG = "OpenCVActivity";
    private static final String FACE1 = "face1";
    private static final String FACE2 = "face2";
    //图片所在文件夹
    private static final String fileName =
            Environment.getExternalStorageDirectory().getPath();
    CameraBridgeViewBase mCameraView;

    private Mat mGray;
    private Mat mRgba;

    private int mDetectorType = NATIVE_DETECTOR;
    private int mAbsoluteFaceSize = 0;
    private float mRelativeFaceSize = 0.2f;
    private DetectionBasedTracker mNativeDetector;
    private CascadeClassifier mJavaDetector;
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);

    private int state = 0;
    private Size mSize0;
    private MatOfInt mChannels[];
    private MatOfInt mHistSize;
    private int mHistSizeNum = 25;
    private float[] mBuff;
    private MatOfFloat mRanges;
    private Point mP1;
    private Point mP2;
    private Scalar mColorsRGB[];
    private Scalar mWhilte;
    private Mat mSepiaKernel;

    private Mat mTmp;
    private Mat mTmp2;

    private File mCascadeFile;

    private boolean isGettingFace = false;
    private Bitmap mBitmapFace1;
    private Bitmap mBitmapFace2;
    private ImageView mImageViewFace1;
    private ImageView mImageViewFace2;
    private TextView mCmpPic;
    private double cmp;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    // OpenCV初始化加载成功，再加载本地so库
                    System.loadLibrary("detection_based_tracker");

                    try {
                        // 加载人脸检测模式文件
                        InputStream is = getResources()
                                .openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir,
                                "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);
                        byte[] buffer = new byte[4096];
                        int byteesRead;
                        while ((byteesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, byteesRead);
                        }
                        is.close();
                        os.close();
                        // 使用模型文件初始化人脸检测引擎

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "加载cascade classifier失败");
                            mJavaDetector = null;
                        } else {
                            Log.d(TAG, "Loaded cascade classifier from "
                                    + mCascadeFile.getAbsolutePath());
                        }
                        mNativeDetector = new
                                DetectionBasedTracker(mCascadeFile.getAbsolutePath(),
                                0);
                        cascadeDir.delete();
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    // 开启渲染Camera
                    mCameraView.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_opencv);

        mSize0 = new Size();
        mChannels = new MatOfInt[]{new MatOfInt(0), new MatOfInt(1), new MatOfInt(2)};
        mBuff = new float[mHistSizeNum];
        mHistSize = new MatOfInt(mHistSizeNum);
        mRanges = new MatOfFloat(0f, 256f);
        mColorsRGB = new Scalar[]{new Scalar(200, 0, 0, 255), new Scalar(0, 200, 0, 255), new Scalar(0, 0, 200, 255)};
        mWhilte = Scalar.all(255);
        mP1 = new Point();
        mP2 = new Point();

        // Fill sepia kernel
        mSepiaKernel = new Mat(4, 4, CvType.CV_32F);
        mSepiaKernel.put(0, 0, /* R */0.189f, 0.769f, 0.393f, 0f);
        mSepiaKernel.put(1, 0, /* G */0.168f, 0.686f, 0.349f, 0f);
        mSepiaKernel.put(2, 0, /* B */0.131f, 0.534f, 0.272f, 0f);
        mSepiaKernel.put(3, 0, /* A */0.000f, 0.000f, 0.000f, 1f);

        //读取照片
        //Mat src = Imgcodecs.imread(fileName + File.separator + "2.jpg");

        mCameraView = (CameraBridgeViewBase) findViewById(R.id.cameraView_face);
        mCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        // 注册Camera渲染事件监听器
        mCameraView.setCvCameraViewListener(this);
        //设置分辨率
        mCameraView.setMaxFrameSize(1280, 720);
        //设置前后摄像头
        mCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);

        findViewById(R.id.switchCamera).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //转换前后摄像头
                if (mCameraView.switchCamera()) {
                    showMsg("success");
                } else {
                    showMsg("fail");
                }
            }
        });
        findViewById(R.id.takePhoto).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //拍照,保存为png格式
                Mat mBgr = new Mat();
                Imgproc.cvtColor(mRgba, mBgr, Imgproc.COLOR_RGBA2BGR, 3);
                if (Imgcodecs.imwrite(fileName + File.separator +
                      /*  System.currentTimeMillis()*/ state + ".png", mBgr)) {
//                    state++;
                    showMsg("success");
                } else {
                    showMsg("fail");
                }
                mBgr.release();
            }
        });
        findViewById(R.id.catchFace).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                isGettingFace = true;
            }
        });
        mImageViewFace1 = (ImageView) findViewById(R.id.face1);
        mImageViewFace2 = (ImageView) findViewById(R.id.face2);
        mCmpPic = (TextView) findViewById(R.id.similarity);
    }

    @Override
    protected void onResume() {
        super.onResume();
        // 静态初始化OpenCV
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "无法加载OpenCV本地库，将使用OpenCV Manager初始化");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_3_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "成功加载OpenCV本地库");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        // 停止渲染Camera
        if (mCameraView != null) {
            mCameraView.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // 停止渲染Camera
        if (mCameraView != null) {
            mCameraView.disableView();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        // 灰度图像
        mGray = new Mat();
        // R、G、B彩色图像
        mRgba = new Mat();

        mTmp = new Mat();

        mTmp2 = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
        mTmp.release();
        mTmp2.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        int cols = (int) mRgba.size().width;
        int rows = (int) mRgba.size().height;

        int left = cols / 8;
        int top = rows / 8;

        int width = cols * 3 / 4;
        int height = rows * 3 / 4;

        switch (state) {
            case 1:
                //旋转图片
                Core.transpose(mRgba, mTmp);
                Imgproc.resize(mTmp, mRgba, mRgba.size());
                break;
            case 2:
                //Canny边缘检测
                Imgproc.Canny(mGray, mTmp, 80, 100);
                Imgproc.cvtColor(mTmp, mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
                break;
            case 3:
                //Hist直方图计算
                Mat hist = new Mat();
                int thikness = (int) (cols / (mHistSizeNum + 10) / 5);
                if (thikness > 5) {
                    thikness = 5;
                }
                int offset = (int) ((cols - (5 * mHistSizeNum + 4 * 10) * thikness) / 2);

                // RGB
                for (int c = 0; c < 3; c++) {
                    Imgproc.calcHist(Arrays.asList(mRgba), mChannels[c], mTmp2, hist, mHistSize, mRanges);
                    Core.normalize(hist, hist, rows / 2, 0, Core.NORM_INF);
                    hist.get(0, 0, mBuff);
                    for (int h = 0; h < mHistSizeNum; h++) {
                        mP1.x = mP2.x = offset + (c * (mHistSizeNum + 10) + h) * thikness;
                        mP1.y = rows - 1;
                        mP2.y = mP1.y - 2 - (int) mBuff[h];
                        Imgproc.line(mRgba, mP1, mP2, mColorsRGB[c], thikness);
                    }
                }
                // Value and Hue
                Imgproc.cvtColor(mRgba, mTmp, Imgproc.COLOR_RGB2HSV_FULL);
                // Value
                Imgproc.calcHist(Arrays.asList(mTmp), mChannels[2], mTmp2, hist, mHistSize, mRanges);
                Core.normalize(hist, hist, rows / 2, 0, Core.NORM_INF);
                hist.get(0, 0, mBuff);
                for (int h = 0; h < mHistSizeNum; h++) {
                    mP1.x = mP2.x = offset + (3 * (mHistSizeNum + 10) + h) * thikness;
                    mP1.y = rows - 1;
                    mP2.y = mP1.y - 2 - (int) mBuff[h];
                    Imgproc.line(mRgba, mP1, mP2, mWhilte, thikness);
                }
                break;
            case 4:
                //Sobel边缘检测
                Imgproc.Sobel(
                        mGray.submat(top, top + height, left, left + width),
                        mTmp, CvType.CV_8U, 1, 1);
                Core.convertScaleAbs(mTmp, mTmp, 10, 0);
                Imgproc.cvtColor(mTmp,
                        mRgba.submat(top, top + height, left, left + width),
                        Imgproc.COLOR_GRAY2BGRA, 4);
                break;
            case 5:
                //SEPIA(色调变换)
                mTmp = mRgba.submat(top, top + height, left, left + width);
                Core.transform(mTmp, mTmp, mSepiaKernel);
                break;
            case 6:
                //ZOOM放大镜
                mTmp = mRgba.submat(0,
                        rows / 2 - rows / 10, 0, cols / 2 - cols / 10);
                mTmp2 = mRgba.submat(rows / 2 - 9 * rows / 100,
                        rows / 2 + 9 * rows / 100,
                        cols / 2 - 9 * cols / 100,
                        cols / 2 + 9 * cols / 100);
                Imgproc.resize(mTmp2, mTmp, mTmp.size());
                Imgproc.rectangle(mTmp2, new Point(1, 1),
                        new Point(mTmp2.size().width - 2, mTmp2.size().height - 2),
                        FACE_RECT_COLOR, 2);
                break;
            case 7:
                //PIXELIZE像素化
                mTmp = mRgba.submat(top, top + height, left, left + width);
                //放大图片
//                float scale = 0.5f;
//                Imgproc.resize(mTmp, mTmp2, new Size(cols * scale, rows * scale));

//                scale = 1.5f;
//                Imgproc.resize(mTmp, mTmp2, new Size(cols * scale, rows * scale));

//                cols = 400;
//                rows = 400;
//                Imgproc.resize(mTmp, mTmp2, new Size(cols, rows));

                //dsize如果等于0时，dsize = Size(round(fx * src.cols), round(fy * src.rows))
                Imgproc.resize(mTmp, mTmp2, mSize0, 0.1, 0.1, Imgproc.INTER_NEAREST);
                Imgproc.resize(mTmp2, mRgba.submat(0,
                        rows / 2 - rows / 10, 0, cols / 2 - cols / 10), mRgba.submat(0,
                        rows / 2 - rows / 10, 0, cols / 2 - cols / 10).size());
//                Imgproc.resize(mTmp2, mTmp, mTmp.size(), 0., 0., Imgproc.INTER_NEAREST);
                break;
            default:
                break;
        }
        // 设置脸部大小
        if (mAbsoluteFaceSize == 0) {
            height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
            mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
        }

        // 获取检测到的脸部数据
        MatOfRect faces = new MatOfRect();
        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null) {
                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2,
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
//                mJavaDetector.detectMultiScale(mGray, // 要检查的灰度图像
//                        faces, // 检测到的人脸
//                        1.1, // 表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%;
//                        10, // 默认是3 控制误检测，表示默认几次重叠检测到人脸，才认为人脸存在
//                        CV_HAAR_FIND_BIGGEST_OBJECT // 返回一张最大的人脸（无效？）
//                                | CV_HAAR_SCALE_IMAGE
//                                | CV_HAAR_DO_ROUGH_SEARCH
//                                | CV_HAAR_DO_CANNY_PRUNING, //CV_HAAR_DO_CANNY_PRUNING ,// CV_HAAR_SCALE_IMAGE, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
//                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize),
//                        new Size(mGray.width(), mGray.height()));
            }
        } else if (mDetectorType == NATIVE_DETECTOR) {
            if (mNativeDetector != null) {
//                Log.v(TAG, "start");
                mNativeDetector.detect(mGray, faces);
//                Log.v(TAG, "end");
            }
        } else {
            Log.e(TAG, "Detection method is not selected!");
        }

        // 绘制检测框
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++) {
            Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
            onFace(mRgba, facesArray[i]);
        }
        if (faces.toArray().length > 0) {
//            Log.d(TAG, "共检测到 " + faces.toArray().length + " 张脸");
        } else {
            if (isGettingFace) {
                isGettingFace = !isGettingFace;
            }
        }
        return mRgba;
    }

    /**
     * 检测到人脸
     *
     * @param mat  Mat
     * @param rect Rect
     */
    public void onFace(Mat mat, Rect rect) {
        if (isGettingFace) {
            if (null == mBitmapFace1 || null != mBitmapFace2) {
                mBitmapFace1 = null;
                mBitmapFace2 = null;

                // 保存人脸信息并显示
                FaceUtil.saveImage(this, mat, rect, FACE1);
                mBitmapFace1 = FaceUtil.getImage(this, FACE1);
                cmp = 0.0d;
            } else {
                FaceUtil.saveImage(this, mat, rect, FACE2);
                mBitmapFace2 = FaceUtil.getImage(this, FACE2);

                // 计算相似度
                cmp = FaceUtil.compare(this, FACE1, FACE2);
                Log.i(TAG, "onFace: 相似度 : " + cmp);
            }
            Log.d(TAG, "onFace");
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    if (null == mBitmapFace1) {
                        mImageViewFace1.setImageResource(R.mipmap.ic_launcher);
                    } else {
                        mImageViewFace1.setImageBitmap(mBitmapFace1);
                    }
                    if (null == mBitmapFace2) {
                        mImageViewFace2.setImageResource(R.mipmap.ic_launcher);
                    } else {
                        mImageViewFace2.setImageBitmap(mBitmapFace2);
                    }
                    mCmpPic.setText(String.format("相似度 :  %.2f", cmp) + "%");
                }
            });

            isGettingFace = false;
        }
    }

    private void showMsg(String msg) {
        Toast.makeText(OpenCVActivity.this, msg, Toast.LENGTH_SHORT).show();
    }
}
