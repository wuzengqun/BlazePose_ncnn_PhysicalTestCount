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

#include "blazepose.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cpu.h"

#include <android/log.h>
#define LOG_TAG "BlazePose"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)


//value_result inference_result1;
value_result inference_result1{
        std::vector<int>(5, 0),  // 提前分配
        0
};

const std::vector<std::pair<int,int>> lines = {
        {0, 1},
        {0, 4},
        {1, 2},
        {2, 3},
        {3, 7},
        {4, 5},
        {5, 6},
        {6, 8},
        {9, 10},
        {11,12},
        {11,13},
        {11,23},
        {12,14},
        {12,24},
        {13,15},
        {14,16},
        {15,17},
        {15,19},
        {15,21},
        {16,18},
        {16,20},
        {16,22},
        {17,19},
        {18,20},
        {23,24}
};

const std::vector<std::pair<int,int>> extended_lines_fb = {
    {23, 25},
    {24, 26},
    {25, 27},
    {26, 28},
    {27, 29},
    {27, 31},
    {28, 30},
    {28, 32},
    {29, 31},
    {30, 32}
};
const std::vector<std::pair<int, int>> left_body = {
        {1, 2},
        {2,	3},
        {3, 4},
        {3,7},
        {11,13},
        {13,15},
        {15,17},
        {17,19},
        {19,15},
        {15,21},
        {11,23},
        {23,25},
        {25,27},
        {27,29},
        {29,31},
        {31,27}
};
const std::vector<std::pair<int, int>> right_body = {
        { 4, 5 },
        { 5, 6 },
        { 6 ,8 },
        { 12 ,14 },
        { 14 ,16 },
        { 16 ,18 },
        { 18 ,20 },
        { 20 ,16 },
        { 16 ,22 },
        { 12 ,24 },
        {24,26},
        {26,28},
        {28,30},
        {30,32},
        {32,28},
};

static float calculateScale(float min_scale, float max_scale, int stride_index, int num_strides) 
{
    if (num_strides == 1)
        return (min_scale + max_scale) * 0.5f;
    else
        return min_scale + (max_scale - min_scale) * 1.0 * stride_index / (num_strides - 1.0f);
}

static void generateAnchors(std::vector<Anchor>& anchors, const AnchorsParams& anchor_params)
{
    int layer_id = 0;
    for(int layer_id = 0; layer_id < anchor_params.strides.size();)
    {
        std::vector<float> anchor_height;
        std::vector<float> anchor_width;
        std::vector<float> aspect_ratios;
        std::vector<float> scales;
        
        int last_same_stride_layer = layer_id;
        while (last_same_stride_layer < (int)anchor_params.strides.size() &&
            anchor_params.strides[last_same_stride_layer] == anchor_params.strides[layer_id])
        {
            const float scale = calculateScale(anchor_params.min_scale, anchor_params.max_scale,last_same_stride_layer, anchor_params.strides.size());
            {
                for (int aspect_ratio_id = 0; aspect_ratio_id < (int)anchor_params.aspect_ratios.size(); aspect_ratio_id++)
                {
                    aspect_ratios.push_back(anchor_params.aspect_ratios[aspect_ratio_id]);
                    scales.push_back(scale);
                }
              
                const float scale_next =last_same_stride_layer == (int)anchor_params.strides.size() - 1? 1.0f : calculateScale(anchor_params.min_scale, anchor_params.max_scale,last_same_stride_layer + 1,anchor_params.strides.size());
                scales.push_back(std::sqrt(scale * scale_next));
                aspect_ratios.push_back(1.0);
            }
            last_same_stride_layer++;
        }

        for (int i = 0; i < (int)aspect_ratios.size(); ++i) 
        {
            const float ratio_sqrts = std::sqrt(aspect_ratios[i]);
            anchor_height.push_back(scales[i] / ratio_sqrts);
            anchor_width.push_back(scales[i] * ratio_sqrts);
        }

        int feature_map_height = 0;
        int feature_map_width = 0;
        const int stride = anchor_params.strides[layer_id];
        feature_map_height = std::ceil(1.0f * anchor_params.input_size_height / stride);
        feature_map_width = std::ceil(1.0f * anchor_params.input_size_width / stride);

        for (int y = 0; y < feature_map_height; ++y) 
        {
            for (int x = 0; x < feature_map_width; ++x) 
            {
                for (int anchor_id = 0; anchor_id < (int)anchor_height.size(); ++anchor_id) 
                {
                    const float x_center = (x + anchor_params.anchor_offset_x) * 1.0f / feature_map_width;
                    const float y_center = (y + anchor_params.anchor_offset_y) * 1.0f / feature_map_height;

                    Anchor new_anchor;
                    new_anchor.x_center = x_center;
                    new_anchor.y_center = y_center;

                    new_anchor.w = 1.0f;
                    new_anchor.h = 1.0f;

                    anchors.push_back(new_anchor);
                }
            }
        }
        layer_id = last_same_stride_layer;
    }
}

static void createAnchors(int input_w, int input_h, std::vector<Anchor> &anchors) 
{
    AnchorsParams anchor_options;
    anchor_options.num_layers        = 5;
    anchor_options.min_scale         = 0.1484375;
    anchor_options.max_scale         = 0.75;
    anchor_options.input_size_height = 224;
    anchor_options.input_size_width  = 224;
    anchor_options.anchor_offset_x   = 0.5f;
    anchor_options.anchor_offset_y   = 0.5f;
    anchor_options.strides.push_back(8);
    anchor_options.strides.push_back(16);
    anchor_options.strides.push_back(32);
    anchor_options.strides.push_back(32);
    anchor_options.strides.push_back(32);
    anchor_options.aspect_ratios.push_back(1.0);
    generateAnchors(anchors, anchor_options);
}
static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static int decodeBBox(std::list<DetectRegion>& region_list, float score_thresh, int input_img_w,
        int input_img_h, const ncnn::Mat& cls, const ncnn::Mat& reg, std::vector<Anchor>& anchors)
{
    DetectRegion region;
    size_t i = 0;
    const float* scores_ptr = (float*)cls.data;
    const float* bboxes_ptr = (float*)reg.data;

    for (auto &anchor : anchors) 
    {
        float score = sigmoid(scores_ptr[i]);
        
        if (score > score_thresh)
        {
            const float* p = bboxes_ptr + (i * 12);

            float cx = p[0] / input_img_w + anchor.x_center;
            float cy = p[1] / input_img_h + anchor.y_center;
            float w  = p[2] / input_img_w;
            float h  = p[3] / input_img_h;

            cv::Point2f topleft, btmright;
            topleft.x  = cx - w * 0.5f;
            topleft.y  = cy - h * 0.5f;
            btmright.x = cx + w * 0.5f;
            btmright.y = cy + h * 0.5f;

            region.score    = score;
            region.topleft  = topleft;
            region.btmright = btmright;

            for (size_t j = 0; j < 4; j++)
            {
                float lx = p[4 + (2 * j) + 0];
                float ly = p[4 + (2 * j) + 1];
                lx += anchor.x_center * input_img_w;
                ly += anchor.y_center * input_img_h;
                lx /= (float)input_img_w;
                ly /= (float)input_img_h;

                region.landmarks[j].x = lx;
                region.landmarks[j].y = ly;
            }

            region_list.push_back(region);
        }
        i++;
    }

    return 0;
}

static float calcIOU(DetectRegion& region0, DetectRegion& region1)
{
    float sx0 = region0.topleft.x;
    float sy0 = region0.topleft.y;
    float ex0 = region0.btmright.x;
    float ey0 = region0.btmright.y;
    float sx1 = region1.topleft.x;
    float sy1 = region1.topleft.y;
    float ex1 = region1.btmright.x;
    float ey1 = region1.btmright.y;

    float xmin0 = std::min(sx0, ex0);
    float ymin0 = std::min(sy0, ey0);
    float xmax0 = std::max(sx0, ex0);
    float ymax0 = std::max(sy0, ey0);
    float xmin1 = std::min(sx1, ex1);
    float ymin1 = std::min(sy1, ey1);
    float xmax1 = std::max(sx1, ex1);
    float ymax1 = std::max(sy1, ey1);

    float area0 = (ymax0 - ymin0) * (xmax0 - xmin0);
    float area1 = (ymax1 - ymin1) * (xmax1 - xmin1);
    if (area0 <= 0 || area1 <= 0)
        return 0.0f;

    float intersect_xmin = std::max(xmin0, xmin1);
    float intersect_ymin = std::max(ymin0, ymin1);
    float intersect_xmax = std::min(xmax0, xmax1);
    float intersect_ymax = std::min(ymax0, ymax1);

    float intersect_area = std::max(intersect_ymax - intersect_ymin, 0.0f) *
        std::max(intersect_xmax - intersect_xmin, 0.0f);

    return intersect_area / (area0 + area1 - intersect_area);
}


static int nms(std::list<DetectRegion>& region_list, std::list<DetectRegion>& region_nms_list, float iou_thresh) 
{
    region_list.sort([](DetectRegion& v1, DetectRegion& v2) { return v1.score > v2.score ? true : false; });

    for (auto itr = region_list.begin(); itr != region_list.end(); itr++)
    {
        DetectRegion region_candidate = *itr;

        int ignore_candidate = false;
        for (auto itr_nms = region_nms_list.rbegin(); itr_nms != region_nms_list.rend(); itr_nms++)
        {
            DetectRegion region_nms = *itr_nms;

            float iou = calcIOU(region_candidate, region_nms);
            if (iou >= iou_thresh)
            {
                ignore_candidate = true;
                break;
            }
        }

        if (!ignore_candidate)
        {
            region_nms_list.push_back(region_candidate);
            if (region_nms_list.size() >= 5)
                break;
        }
    }
    return 0;
}

static float normalizeRadians(float angle)
{
    return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
}

static void computeRotation(DetectRegion& region) 
{
    float x0 = region.landmarks[0].x;
    float y0 = region.landmarks[0].y;
    float x1 = region.landmarks[2].x;
    float y1 = region.landmarks[2].y;

    float target_angle = M_PI * 0.5f;
    float rotation = target_angle - std::atan2(-(y1 - y0), x1 - x0);

    region.rotation = normalizeRadians(rotation);
}

void rotVector(cv::Point2f& vec, float rotation)
{
    float sx = vec.x;
    float sy = vec.y;
    vec.x = sx * std::cos(rotation) - sy * std::sin(rotation);
    vec.y = sx * std::sin(rotation) + sy * std::cos(rotation);
}
static float getDistance(const cv::Point2f& p1, const cv::Point2f& p2)
{
    return std::sqrt((p1.x - p2.x)* (p1.x - p2.x) + (p1.y - p2.y)* (p1.y - p2.y));
}

void compute_detect_to_roi(DetectRegion& region, int target_size,int wpad,int hpad,float scale, Object& obj)
{
    float width = region.btmright.x - region.topleft.x;
    float height = region.btmright.y - region.topleft.y;
    float center_x = region.landmarks[0].x;
    float center_y = region.landmarks[0].y;
    
    float cx;
    float cy;
    float rotation = region.rotation;
    float shift_x = 0.0f;
    float shift_y = 0.0f;

    if (rotation == 0.0f)
    {
        cx = center_x + (width * shift_x);
        cy = center_y + (height * shift_y);
    }
    else
    {
        float dx = (width * shift_x) * std::cos(rotation) -
            (height * shift_y) * std::sin(rotation);
        float dy = (width * shift_x) * std::sin(rotation) +
            (height * shift_y) * std::cos(rotation);
        cx = center_x + dx;
        cy = center_y + dy;
    }

    float long_side = 2.0 * getDistance(region.landmarks[0],region.landmarks[1]);
    width = long_side;
    height = long_side;
    float w = width * 1.25f;
    float h = height * 1.25f;

    obj.cx = cx;
    obj.cy = cy;
    obj.w = w;
    obj.h = h;

    float dx = w * 0.5f;
    float dy = h * 0.5f;

    obj.points[0].x = -dx;  obj.points[0].y = -dy;
    obj.points[1].x = +dx;  obj.points[1].y = -dy;
    obj.points[2].x = +dx;  obj.points[2].y = +dy;
    obj.points[3].x = -dx;  obj.points[3].y = +dy;

    for (size_t i = 0; i < 4; i++)
    {
        rotVector(obj.points[i], rotation);
        obj.points[i].x = ((obj.points[i].x + cx) * target_size - (wpad / 2)) / scale;
        obj.points[i].y = ((obj.points[i].y + cy) * target_size - (hpad / 2)) / scale;
    }

    for (size_t i = 0; i < 4; i++)
    {
        obj.landmarks[i].x = (region.landmarks[i].x * target_size - (wpad / 2)) / scale;
        obj.landmarks[i].y = (region.landmarks[i].y * target_size - (hpad / 2)) / scale;
    }

    obj.score = region.score;
}


static void getRotatedBBox(std::list<DetectRegion>& region_list,int target_size,int wpad,int hpad,float scale,std::vector<Object>& objects)
{
    for (auto& region : region_list) 
    {
        computeRotation(region);
        Object obj;
        compute_detect_to_roi(region, target_size, wpad, hpad, scale, obj);
        objects.push_back(obj);
    }
}
static void preProcess(const cv::Mat& image, int target_size, ncnn::Mat& img_tensor, int& wpad, int& hpad, float& scale)
{

    int width = image.cols;
    int height = image.rows;

    int w = width;
    int h = height;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_RGB, width, height, w, h);

    wpad = target_size - w;
    hpad = target_size - h;

    ncnn::copy_make_border(in, img_tensor, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);
    const float norm_vals[3] = { 1 / 127.5f, 1 / 127.5f, 1 / 127.5f };
    const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
    img_tensor.substract_mean_normalize(mean_vals, norm_vals);
    
}
static void getTransMatrix(const cv::Point2f* org_pts, int target_size, cv::Mat& trans_mat)
{
    cv::Point2f pt1, pt2, pt3, pt4;
    pt1.x = org_pts[0].x;
    pt1.y = org_pts[0].y;
    pt2.x = org_pts[1].x;
    pt2.y = org_pts[1].y;
    pt3.x = org_pts[2].x;
    pt3.y = org_pts[2].y;
    pt4.x = org_pts[3].x;
    pt4.y = org_pts[3].y;

    cv::Point2f srcPts[4];
    srcPts[0] = pt1;
    srcPts[1] = pt2;
    srcPts[2] = pt3;
    srcPts[3] = pt4;
    cv::Point2f dstPts[4];
    dstPts[0] = cv::Point2f(0, 0);
    dstPts[1] = cv::Point2f(target_size, 0);
    dstPts[2] = cv::Point2f(target_size, target_size);
    dstPts[3] = cv::Point2f(0, target_size);

    trans_mat = cv::getAffineTransform(srcPts, dstPts);
}
void BlazePose::smoothingLandmarks(std::vector<Keypoint>& detects, int img_w,int img_h)
{
    Normalized2DLandmarkList landmarks;
    for (auto& kpt : detects)
    {
        Normalized2DLandmark pt;
        pt.first = kpt.x / img_w;
        pt.second = kpt.y / img_h;
        landmarks.push_back(pt);
    }

    auto image_size = std::make_pair(img_h, img_w);
    Normalized2DLandmarkList out_landmarks;
    velocity_landmark_filter->Apply2D(landmarks, image_size, Now(), &out_landmarks);
    //one_euro_landmark_filter->Apply2D(landmarks, image_size, Now(), &out_landmarks);
    if (out_landmarks.size() > 0)
    {
        for (size_t i = 0; i < out_landmarks.size(); i++)
        {
            detects[i].x = out_landmarks[i].first * img_w;
            detects[i].y = out_landmarks[i].second * img_h;
        }
    }
}

void printSkeleton(const std::vector<Object>& objects)
{
    for (size_t j = 0; j < objects.size(); j++)
    {
        const auto& obj = objects[j];
        LOGD("===== Object %zu =====", j);
        LOGD("Skeleton size: %zu", obj.skeleton.size());
        for (size_t i = 0; i < obj.skeleton.size(); i++)
        {
            const auto& kp = obj.skeleton[i];
            LOGD("Keypoint %zu: (%.2f, %.2f)", i, kp.x, kp.y);
        }
        LOGD("=====================");
    }
}

int BlazePose::detect(const cv::Mat& rgb, std::vector<Object>& objects)
{
    ncnn::Mat img_tensor;
    int wpad = 0, hpad = 0;
    float scale = 0.f;
    preProcess(rgb, 224, img_tensor,wpad,hpad,scale);
    ncnn::Extractor ex = pose_detection.create_extractor();

    ex.input("input", img_tensor);
    ncnn::Mat cls, reg;
    ex.extract("cls", cls);
    ex.extract("reg", reg);

    std::list<DetectRegion> region_list, region_nms_list;
    decodeBBox(region_list, 0.5f, 224, 224, cls, reg, anchors);
    nms(region_list, region_nms_list, 0.3f);

    getRotatedBBox(region_nms_list, 224, wpad, hpad, scale, objects);

    for(auto& obj : objects)
    {
        cv::Mat trans_mat;
        getTransMatrix(obj.points, 256, trans_mat);
        cv::warpAffine(rgb, obj.trans_image, trans_mat, cv::Size(256, 256), 1, 0);
        float prob = pose_landmark.detect(obj.trans_image, trans_mat, obj.skeleton);
        obj.score = prob;
        // smoothingLandmarks(obj.skeleton, rgb.cols, rgb.rows);
    }
    return 0;
}

BlazePose::BlazePose()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}
int BlazePose::load(AAssetManager* mgr, const char* modeltype, bool use_gpu)
{
    anchors.clear();
    pose_detection.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    pose_detection.opt = ncnn::Option();
#if NCNN_VULKAN
    pose_detection.opt.use_vulkan_compute = use_gpu;
#endif

    pose_detection.opt.num_threads = ncnn::get_big_cpu_count();
    pose_detection.opt.blob_allocator = &blob_pool_allocator;
    pose_detection.opt.workspace_allocator = &workspace_pool_allocator;

    pose_detection.load_param(mgr, "detection.param");
    pose_detection.load_model(mgr, "detection.bin");

    pose_landmark.load(mgr, modeltype);

    createAnchors(224, 224, anchors);

    velocity_landmark_filter = std::make_shared<VelocityFilter>(window_size, velocity_scale, min_allowed_object_scale, 30);
    //one_euro_landmark_filter = std::make_shared<OneEuroFilterImpl>(frequency, min_cutoff, beta, derivate_cutoff, min_allowed_object_scale, disable_value_scaling);

    return 0;
}

//计算两点角度
static int calculateangle(int x1, int y1, int x2, int y2) {
    // 计算两点之间的差值
    double dx = static_cast<double>(x2 - x1); // 使用 double 保持精度
    double dy = static_cast<double>(y2 - y1);

    // 使用atan2计算角度，结果为弧度
    double angle_radians = atan2(dy, dx);

    // 将弧度转换为度
    double angle_degrees = angle_radians * (180.0 / M_PI);

    // 确保角度为正值
    if (angle_degrees < 0) {
        angle_degrees += 360.0;
    }

    // 将结果转换为 int
    return static_cast<int>(angle_degrees + 0.5); // 四舍五入
}

static inline float calc_angle_3pts(
        float x1, float y1,   // shoulder
        float x2, float y2,   // elbow (vertex)
        float x3, float y3    // wrist
)
{
    float v1x = x1 - x2;
    float v1y = y1 - y2;
    float v2x = x3 - x2;
    float v2y = y3 - y2;

    float dot = v1x * v2x + v1y * v2y;
    float n1 = sqrtf(v1x * v1x + v1y * v1y);
    float n2 = sqrtf(v2x * v2x + v2y * v2y);

    if (n1 < 1e-6f || n2 < 1e-6f)
        return 180.f; // 防止除 0

    float cos_theta = dot / (n1 * n2);

    // ★ 关键：防止 acos NaN
    cos_theta = std::max(-1.f, std::min(1.f, cos_theta));

    return acosf(cos_theta) * 180.f / M_PI;
}

//函数：检查深蹲时角度是否合适
static bool is_situp_bodyangle(float angle1,float angle2,float angle3)
{
    return ((angle1 >= 95) && (angle1 <= 125)) && ((angle2 >= 175) && (angle2 <= 205)) && ((angle3 >= 112) && (angle3 <= 130));
}

//仰卧起坐计数
static bool is_crunch_standard(const std::vector<Object>& objects)
{
    static enum {
        STATE_DOWN,        // 背部贴地 / 初始态
        STATE_UP,          // 卷腹到位
        STATE_COMPLETED    // 完成一次
    } state = STATE_DOWN;

    // 1. 至少检测到一个人
    if (objects.empty()) {
        return false;
    }

    const Object& obj = objects[0];
    const auto& skel = obj.skeleton;

    // 2. 关键点数量检查（用到 16）
    if (skel.size() < 17) {
        LOGD("Invalid skeleton size: %zu", skel.size());
        return false;
    }

    const Keypoint& p6  = skel[11];
    const Keypoint& p12 = skel[23];
    const Keypoint& p14 = skel[25];
    const Keypoint& p16 = skel[27];

    // 4. 可见性过滤（非常重要）
    if (!p6.visibility || !p12.visibility ||
        !p14.visibility || !p16.visibility) {
        return false;
    }

    // 5. 计算角度（完全对应你原来的逻辑）
    int tui_angle = calc_angle_3pts(
            p12.x, p12.y,
            p14.x, p14.y,
            p16.x, p16.y
    );

    bool tui_angle_ok = (tui_angle >= 70 && tui_angle <= 130);

    int bei_angle = calculateangle(
            p6.x, p6.y,
            p12.x, p12.y
    );

    bool bei_angle_ok = (bei_angle >= 80 && bei_angle <= 100);

    bool body_ok = tui_angle_ok && bei_angle_ok;

    // 6. 状态机
    switch (state) {

        // 背部贴地 → 卷腹到位
        case STATE_DOWN:
            if (body_ok) {
                state = STATE_UP;
            }
            break;

            // 卷腹到位 → 完成一次
        case STATE_UP:
            // 背放平才算完成
            if (!body_ok && bei_angle < 20) {
                state = STATE_COMPLETED;
                return true;   // ✔ 计数 +1
            }
            break;

            // 防止重复计数，等待完全躺平
        case STATE_COMPLETED:
            if (!body_ok && bei_angle < 20) {
                state = STATE_DOWN;
            }
            break;
    }

    return false;
}


static bool is_pushup_standard(const std::vector<Object>& objects)
{
    static bool in_down = false;   // 是否已经下去
    static bool in_up   = true;    // 是否已回到上

    if (objects.empty())
        return false;

    const Object& obj = objects[0];
    const auto& skel = obj.skeleton;

    // 2. BlazePose 的关键点数应为 39
    if (skel.size() < 16) {  // 至少用到 13
        LOGD("Invalid skeleton size: %zu", skel.size());
        return false;
    }

    // 关键点索引（按 BlazePose）
    const Keypoint& shoulder = skel[11]; // 右肩
    const Keypoint& elbow    = skel[13]; // 右肘
    const Keypoint& wrist    = skel[15]; // 右腕

    if (!shoulder.visibility || !elbow.visibility || !wrist.visibility)
        return false;

    float angle = calc_angle_3pts(
            shoulder.x, shoulder.y,
            elbow.x, elbow.y,
            wrist.x, wrist.y
    );

    LOGD("elbow angle = %.1f", angle);
    // 下去（手臂弯曲）
    if (angle < 90.f && in_up)
    {
        in_down = true;
        in_up = false;
    }

    // 上来（手臂伸直）→ 完成一次
    if (angle > 140.f && in_down)
    {
        in_down = false;
        in_up = true;
        return true;  // ✔ 完成一次俯卧撑
    }
    return false;
}

static bool isPullupStandard(std::vector<Object> objects)
{
    static enum {
        STATE_DOWN,
        STATE_UP,
        STATE_COMPLETED
    } state = STATE_DOWN;

    // 1. 至少要有一个人
    if (objects.empty()) {
        return false;
    }

    const Object& obj = objects[0];  // 只看第 0 个人
    const auto& skel = obj.skeleton;

    // 2. BlazePose 的关键点数应为 39
    if (skel.size() < 15) {  // 至少用到 13
        LOGD("Invalid skeleton size: %zu", skel.size());
        return false;
    }

    // 3. 取你需要的关键点
    const int k1 = 12;
    const int k2 = 14;

    const Keypoint& p1 = skel[k1];
    const Keypoint& p2 = skel[k2];

    // 4. 可选：过滤不可见点
    if (!p1.visibility || !p2.visibility) {
        return false;
    }

    float angle = calculateangle(p1.x, p1.y, p2.x, p2.y);

    // 5. 状态机
    switch (state) {
        case STATE_DOWN:
            if (angle < 170.f) {
                state = STATE_UP;
            }
            break;

        case STATE_UP:
            if (angle > 210.f) {
                state = STATE_COMPLETED;
                return true;
            }
            break;

        case STATE_COMPLETED:
            if (angle > 223.f) {
                state = STATE_DOWN;
            }
            break;
    }
    return false;
}

//函数：计算深蹲是否标准
static bool is_situp_standard(const std::vector<Object>& objects)
{
    static enum {
        STATE_DOWN,        // 躺平 / 初始态
        STATE_UP,          // 起身完成
        STATE_COMPLETED    // 完成一次
    } state = STATE_DOWN;

    // 1. 至少有一个人
    if (objects.empty()) {
        return false;
    }

    const Object& obj = objects[0];
    const auto& skel = obj.skeleton;

    // 2. 关键点数量检查（用到 16）
    if (skel.size() < 28) {
        LOGD("Invalid skeleton size: %zu", skel.size());
        return false;
    }

    const Keypoint& p6  = skel[11];
    const Keypoint& p12 = skel[23];
    const Keypoint& p14 = skel[25];
    const Keypoint& p16 = skel[27];

    // 4. 可见性过滤（非常重要）
    if (!p6.visibility || !p12.visibility ||
        !p14.visibility || !p16.visibility) {
        return false;
    }

    // 5. 计算角度（完全对应你原来的 angle1/2/3）
    float angle1 = calculateangle(p6.x,  p6.y,  p12.x, p12.y);
    float angle2 = calculateangle(p14.x, p14.y, p12.x, p12.y);
    float angle3 = calculateangle(p14.x, p14.y, p16.x, p16.y);

    bool body_ok = is_situp_bodyangle(angle1, angle2, angle3);

    // 6. 状态机
    switch (state) {

        // 躺平 → 起身
        case STATE_DOWN:
            if (body_ok) {
                state = STATE_UP;
            }
            break;

            // 起身 → 完成一次
        case STATE_UP:
            // 背挺直才算完成
            if (!body_ok && angle1 < 95.f) {
                state = STATE_COMPLETED;
                return true;   // ✔ 计数 +1
            }
            break;

            // 防止重复计数，等待回到躺平
        case STATE_COMPLETED:
            if (!body_ok && angle1 < 95.f) {
                state = STATE_DOWN;
            }
            break;
    }
    return false;
}



struct value_result BlazePose::draw(cv::Mat& rgb, const std::vector<Object>& objects,int selectmode)
{
    static bool isStandardPullup = 0;
    static int id = 0;

    for(const auto& obj : objects)
    {
        if(obj.score < 0)
            continue;
        if (obj.skeleton.size() < 33) {
            LOGD("skip draw: skeleton size=%zu", obj.skeleton.size());
            continue;
        }
        if (rgb.empty() || rgb.data == nullptr) {
            LOGD("rgb invalid, skip draw");
            return inference_result1;
        }
        for (size_t i = 0; i < obj.skeleton.size(); i++)
        {
            const Keypoint& kpt = obj.skeleton[i];
            if (kpt.visibility)
                cv::circle(rgb, cv::Point(kpt.x, kpt.y), 2,
                           cv::Scalar(255, 255, 255), -1);
        }
        for (const auto& line : lines)
        {
            if (line.first >= obj.skeleton.size() ||
                line.second >= obj.skeleton.size())
                continue;

            if (obj.skeleton[line.first].visibility &&
                obj.skeleton[line.second].visibility)
            {
                cv::line(rgb, cv::Point(obj.skeleton[line.first].x, obj.skeleton[line.first].y),
                    cv::Point(obj.skeleton[line.second].x, obj.skeleton[line.second].y),
                    cv::Scalar(255, 255, 255), 2, 8, 0);
            }
        }
        auto safe = [&](int idx) {
            return idx >= 0 && idx < obj.skeleton.size();
        };
        for (const auto& line : extended_lines_fb)
        {
            if (!safe(line.first) || !safe(line.second))
                continue;

            if (obj.skeleton[line.first].visibility &&
                obj.skeleton[line.second].visibility)
            {
                cv::line(rgb, cv::Point(obj.skeleton[line.first].x, obj.skeleton[line.first].y),
                         cv::Point(obj.skeleton[line.second].x, obj.skeleton[line.second].y),
                         cv::Scalar(255, 255, 255), 2, 8, 0);
            }
        }

        for (const auto& line : left_body)
        {
            if(obj.skeleton[line.first].visibility && obj.skeleton[line.second].visibility)
            {
                cv::line(rgb, cv::Point(obj.skeleton[line.first].x, obj.skeleton[line.first].y),
                         cv::Point(obj.skeleton[line.second].x, obj.skeleton[line.second].y),
                         cv::Scalar(255, 138, 0), 1, cv::LINE_AA, 0);
                cv::circle(rgb, cv::Point(obj.skeleton[line.first].x, obj.skeleton[line.first].y), 3,
                        cv::Scalar(255, 138, 0), 1, cv::LINE_AA, 0);
                cv::circle(rgb, cv::Point(obj.skeleton[line.second].x, obj.skeleton[line.second].y), 3,
                        cv::Scalar(255, 138, 0), 1, cv::LINE_AA, 0);
            }

        }
        for (const auto& line : right_body)
        {
            if(obj.skeleton[line.first].visibility && obj.skeleton[line.second].visibility)
            {
                cv::line(rgb, cv::Point(obj.skeleton[line.first].x, obj.skeleton[line.first].y),
                         cv::Point(obj.skeleton[line.second].x, obj.skeleton[line.second].y),
                         cv::Scalar(0, 217, 231), 1, cv::LINE_AA, 0);
                cv::circle(rgb, cv::Point(obj.skeleton[line.first].x, obj.skeleton[line.first].y), 3,
                        cv::Scalar(0, 217, 231), 1, cv::LINE_AA, 0);
                cv::circle(rgb, cv::Point(obj.skeleton[line.second].x, obj.skeleton[line.second].y), 3,
                        cv::Scalar(0, 217, 231), 1, cv::LINE_AA, 0);
            }

        }
        /*cv::circle(rgb, obj.landmarks[0], 2, cv::Scalar(0, 0, 255), -1);
        cv::circle(rgb, obj.landmarks[1], 2, cv::Scalar(0, 255, 0), -1);
        cv::circle(rgb, obj.landmarks[2], 2, cv::Scalar(255, 0, 0), -1);
        cv::circle(rgb, obj.landmarks[3], 2, cv::Scalar(0, 255, 255), -1);
        cv::line(rgb, obj.points[0], obj.points[1], cv::Scalar(0, 0, 255), 2, 8, 0);
        cv::line(rgb, obj.points[1], obj.points[2], cv::Scalar(0, 255, 0), 2, 8, 0);
        cv::line(rgb, obj.points[2], obj.points[3], cv::Scalar(255, 0, 0), 2, 8, 0);
        cv::line(rgb, obj.points[3], obj.points[0], cv::Scalar(0, 255, 255), 2, 8, 0);*/
    }
    if(objects.empty())
    {
        //如果没检测到目标啥也不干
        //LOGD("no target");
    }
    else {
        //LOGD("objects.size(): %d", objects.size());

        //选择模式判断
        switch (selectmode) {
            case 0:
                isStandardPullup = isPullupStandard(objects);   //引体
                if (isStandardPullup) {
                    inference_result1.count[0]++;
                }
                break;
            case 1:
                if(is_pushup_standard(objects))
                {
                    inference_result1.count[1]++;
                }
                break;
            case 2:
                if(is_situp_standard(objects))
                {
                    inference_result1.count[2]++;
                }
                break;
            case 3:
                if(is_crunch_standard(objects))
                {
                    inference_result1.count[3]++;
                }
                break;
            case 4:
                break;
            default:
                break;
        }
    }
    for (int i = 0; i < 5; ++i) {
        if (i != selectmode)
            inference_result1.count[i] = 0;
    }
    return inference_result1;
}
