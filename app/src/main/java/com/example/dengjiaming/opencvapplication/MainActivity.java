package com.example.dengjiaming.opencvapplication;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
//        System.loadLibrary("opencv_java3");
    }

    private static final int READ_EXTERNAL_STORAGE_CODE = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        if (ContextCompat.checkSelfPermission(this,
                android.Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED
                && ContextCompat.checkSelfPermission(this,
                android.Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED
                && ContextCompat.checkSelfPermission(this,
                android.Manifest.permission.MOUNT_UNMOUNT_FILESYSTEMS)
                != PackageManager.PERMISSION_GRANTED
                && ContextCompat.checkSelfPermission(this,
                android.Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED
                && ContextCompat.checkSelfPermission(this,
                android.Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{android.Manifest.permission.READ_EXTERNAL_STORAGE
                            , android.Manifest.permission.WRITE_EXTERNAL_STORAGE
                            , android.Manifest.permission.MOUNT_UNMOUNT_FILESYSTEMS
                            , android.Manifest.permission.CAMERA
                            , android.Manifest.permission.RECORD_AUDIO},
                    READ_EXTERNAL_STORAGE_CODE);
        }
        // Example of a call to a native method
        Button tv = (Button) findViewById(R.id.sample_text);
        tv.setText(stringFromJNI());
        Intent intentFd = new Intent(MainActivity.this, OpenCVActivity.class);
        startActivity(intentFd);
        tv.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intentFd = new Intent(MainActivity.this, OpenCVActivity.class);
                startActivity(intentFd);
            }
        });
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (READ_EXTERNAL_STORAGE_CODE == requestCode) {
            for (String string : permissions) {
                Log.e("requestPermissions", string);
            }
        }
    }
}
