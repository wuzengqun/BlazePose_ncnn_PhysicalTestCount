// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.tencent.blazeposencnn;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.Spinner;

import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;

import android.os.Handler;      // Handler 类
import android.widget.TextView; // TextView 类

public class MainActivity extends Activity implements SurfaceHolder.Callback
{
    public static final int REQUEST_CAMERA = 100;

    private BlazePoseNcnn blazeposencnn = new BlazePoseNcnn();
    private int facing = 1;

    private Spinner spinnerModel;
    private Spinner spinnerCPUGPU;
    private int current_model = 2;
    private int current_cpugpu = 0;

    private SurfaceView cameraView;
    int selectmode;  //传入的模式选择

    // 定时器相关
    private Handler countHandler = new Handler();
    private TextView ShowCount;

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        cameraView = (SurfaceView) findViewById(R.id.cameraview);

        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
        cameraView.getHolder().addCallback(this);

        ShowCount = (TextView)findViewById(R.id.count);

        //按键控制引体计数
        Button buttonSelectPullupcount = (Button) findViewById(R.id.buttonSelectPullup);
        buttonSelectPullupcount.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                Log.d("FitnessUI", "引体按钮被点击了！"); // 添加这一行
                selectmode = 0;
                blazeposencnn.setmode(selectmode);
                updateActiveButton((Button) arg0);
            }
        });

        //按键控制俯卧撑计数
        Button buttonSelectPushupcount = (Button) findViewById(R.id.buttonSelectPushup);
        buttonSelectPushupcount.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                selectmode = 1;
                blazeposencnn.setmode(selectmode);
                updateActiveButton((Button) arg0);
            }
        });

        //按键控制深蹲计数
        Button buttonSelectSitupcount = (Button) findViewById(R.id.buttonSelectSitup);
        buttonSelectSitupcount.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                selectmode = 2;
                blazeposencnn.setmode(selectmode);
                updateActiveButton((Button) arg0);
            }
        });

        //按键控制俯卧撑计数
        Button buttonSelectCrushcount = (Button) findViewById(R.id.buttonSelectCrush);
        buttonSelectCrushcount.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                selectmode = 3;
                blazeposencnn.setmode(selectmode);
                updateActiveButton((Button) arg0);
            }
        });

        //按键控制停止计数
        Button buttonSelectStopcount = (Button) findViewById(R.id.buttonSelectStopcount);
        buttonSelectStopcount.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                selectmode = 4;
                blazeposencnn.setmode(selectmode);
                updateActiveButton((Button) arg0);
            }
        });

//        Button buttonSwitchCamera = (Button) findViewById(R.id.buttonSwitchCamera);
//        buttonSwitchCamera.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View arg0) {
//
//                int new_facing = 1 - facing;
//
//                blazeposencnn.closeCamera();
//
//                blazeposencnn.openCamera(new_facing);
//
//                facing = new_facing;
//            }
//        });

        spinnerModel = (Spinner) findViewById(R.id.spinnerModel);
        spinnerModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_model)
                {
                    current_model = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        spinnerCPUGPU = (Spinner) findViewById(R.id.spinnerCPUGPU);
        spinnerCPUGPU.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_cpugpu)
                {
                    current_cpugpu = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        // 强制将 UI 容器置于 SurfaceView 之上
        findViewById(R.id.ui_container).bringToFront();

        reload();
    }

    // 定义更新任务
    private Runnable updateCountsTask = new Runnable() {
        @Override
        public void run() {
            // 1. 从 JNI 获取最新数据
            int result = blazeposencnn.getCounts();

            ShowCount.setText(String.valueOf(result));

            // 3. 每 200 毫秒执行一次（一秒更新5次，足够流畅）
            countHandler.postDelayed(this, 200);
        }
    };

    private void updateActiveButton(Button selectedBtn) {
        // 1. 定义颜色值
        int activeColor = 0xFF4CAF50; // 选中的颜色：亮绿色
        int idleColor = 0x66000000;   // 默认的颜色：半透明黑 (对应你 XML 里的背景)
        int stopBtnColor = 0xFFFF4B4B; // 停止按钮保持红色

        // 2. 找到所有按钮并重置颜色
        // 注意：这里要确保 ID 和你的 XML 一致
        Button btnPullup = (Button)findViewById(R.id.buttonSelectPullup);
        Button btnPushup = (Button)findViewById(R.id.buttonSelectPushup);
        Button btnSitup = (Button)findViewById(R.id.buttonSelectSitup);
        Button btnCrush = (Button)findViewById(R.id.buttonSelectCrush);
        Button btnStop = (Button)findViewById(R.id.buttonSelectStopcount);

        // 将所有模式按钮恢复为默认半透明黑
        btnPullup.getBackground().setTint(idleColor);
        btnPushup.getBackground().setTint(idleColor);
        btnSitup.getBackground().setTint(idleColor);
        btnCrush.getBackground().setTint(idleColor);

        // 停止按钮通常保持红色，除非你也想让它参与变色
        btnStop.getBackground().setTint(stopBtnColor);

        // 3. 将当前点击的按钮染成绿色
        if (selectedBtn.getId() != R.id.buttonSelectStopcount) {
            selectedBtn.getBackground().setTint(activeColor);
        }
    }
    private void reload()
    {
        boolean ret_init = blazeposencnn.loadModel(getAssets(), current_model, current_cpugpu);
        if (!ret_init)
        {
            Log.e("MainActivity", "blazeposencnn loadModel failed");
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height)
    {
        blazeposencnn.setOutputWindow(holder.getSurface());
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder)
    {
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder)
    {
    }

    @Override
    public void onResume()
    {
        super.onResume();
        countHandler.post(updateCountsTask);

        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED)
        {
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }

        blazeposencnn.openCamera(facing);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        countHandler.removeCallbacks(updateCountsTask);

        blazeposencnn.closeCamera();
    }
}
