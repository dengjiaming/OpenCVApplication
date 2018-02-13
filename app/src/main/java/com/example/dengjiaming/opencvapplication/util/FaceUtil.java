package com.example.dengjiaming.opencvapplication.util;

import android.content.Context;
import android.os.Environment;
import android.text.TextUtils;
import android.util.Log;

import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.LinkedList;
import java.util.List;
import java.util.Stack;


/**
 * @author djm
 * @date 2018/2/1.
 */
public final class FaceUtil {

    private static final String TAG = "FaceUtil";
    private static final String mFaceModelDir =
            Environment.getExternalStorageDirectory().getPath();

    private FaceUtil() {
    }

    public static boolean fileIsExists(String path) {
        try {
            File f = new File(path);
            if (!f.exists()) {
                return false;
            }
        } catch (Exception e) {
            return false;
        }
        return true;
    }


    //使用CSV文件去读图像和标签
    public static void readCsv(String filename, Stack<Mat> images, Stack<Integer> labels, char separator) {
        if (!fileIsExists(mFaceModelDir + File.separator + filename)) {
            Log.e(TAG, "No File.");
            return;
        }
        try {
            InputStreamReader reader = new InputStreamReader(
                    new FileInputStream(
                            new File(mFaceModelDir + File.separator + filename)));
            BufferedReader br = new BufferedReader(reader);
            String line = "";
            String path = "";
            int label = 0;
            int temp = 0;
            line = br.readLine();
            if (line == null) {
                Log.e(TAG, "No valid input file was given, please check the given filename.");
            }
            while (line != null) {
                line = br.readLine();
                temp = line.lastIndexOf(separator);
                path = line.substring(0, temp - 1);
                label = Integer.parseInt(line.substring(temp + 1));
                images.push(Imgcodecs.imread(path, 0));
                labels.push(label);
            }


        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 获取人脸特征路径
     *
     * @param fileName 人脸特征的图片的名字
     * @return 路径
     */
    public static String getFilePath(Context context, String fileName) {
        if (TextUtils.isEmpty(fileName)) {
            return null;
        }
        // 内存路径
        return context.getApplicationContext().getFilesDir().getPath() + fileName + ".jpg";
        // 内存卡路径 需要SD卡读取权限
        // return Environment.getExternalStorageDirectory() + "/FaceDetect/" + fileName + ".jpg";
    }

    public static Mat FeatureOrbLannbased(Mat src, Mat dst) {
        FeatureDetector fd = FeatureDetector.create(FeatureDetector.ORB);
        DescriptorExtractor de = DescriptorExtractor.create(DescriptorExtractor.ORB);
        DescriptorMatcher Matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_L1);

        MatOfKeyPoint mkp = new MatOfKeyPoint();
        fd.detect(src, mkp);
        Mat desc = new Mat();
        de.compute(src, mkp, desc);
        Features2d.drawKeypoints(src, mkp, src);

        MatOfKeyPoint mkp2 = new MatOfKeyPoint();
        fd.detect(dst, mkp2);
        Mat desc2 = new Mat();
        de.compute(dst, mkp2, desc2);
        Features2d.drawKeypoints(dst, mkp2, dst);


        // Matching features

        MatOfDMatch Matches = new MatOfDMatch();
        Matcher.match(desc, desc2, Matches);

        double maxDist = Double.MIN_VALUE;
        double minDist = Double.MAX_VALUE;

        DMatch[] mats = Matches.toArray();

        for (int i = 0; i < mats.length; i++) {
            double dist = mats[i].distance;
            if (dist < minDist) {
                minDist = dist;
            }
            if (dist > maxDist) {
                maxDist = dist;
            }
        }
        System.out.println("-->>" + mats.length);
        System.out.println("Min Distance:" + minDist);
        System.out.println("Max Distance:" + maxDist);
        List<DMatch> goodMatch = new LinkedList<>();

        for (int i = 0; i < mats.length; i++) {
            double dist = mats[i].distance;
            if (dist < 3 * minDist && dist < 0.2f) {
                goodMatch.add(mats[i]);
            }
        }

        Matches.fromList(goodMatch);
        // Show result
        Mat OutImage = new Mat();
        Features2d.drawMatches(src, mkp, dst, mkp2, Matches, OutImage);

        return OutImage;
    }

}
