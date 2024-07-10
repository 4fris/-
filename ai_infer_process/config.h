#ifndef FD_CONFIG
#define FD_CONFIG

#include <stdio.h>
//#include"list.h"
#include<math.h>
#include "sample_comm_nnie.h"

#define QUANT_BASE 4096.0f

#define yolo_layer_num 3 // yolo layer 层数

float confidence_threshold = 0.5f;
float iou_threshold = 0.25f;

#ifndef YOLO_MIN
#define YOLO_MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef YOLO_MAX
#define YOLO_MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

#define IMAGE_W 320.0f // 输入图片大小

#define IMAGE_H 320.0f

typedef struct anchor_w_h
{
	float anchor_w;
	
	float anchor_h;
}anchor_w_h;

typedef struct yolo_result
{
	float left_up_x;
	
	float left_up_y;

	float right_down_x;

	float right_down_y;

	int class_index;

	float score;

	struct yolo_result* next;
}yolo_result;

anchor_w_h  anchor_grids[3][3] = {{{10.0f, 13.0f}, {16.0f, 30.0f}, {33.0f, 23.0f}}, // small yolo layer 层 anchor

									{{30.0f, 61.0f}, {62.0f, 45.0f}, {59.0f, 119.0f}}, // middle yolo layer 层 anchor

									{{116.0f, 90.0f}, {156.0f, 198.0f}, {373.0f, 326.0f}}}; // large yolo layer 层 anchor
	
float strides[3] = {8.0f, 16.0f, 32.0f}; // 每个 yolo 层，grid 大小，与上面顺序对应

int map_size[3] = {40, 20, 10}; // 每个 yolo 层，feature map size 大小，与上面顺序对应


/*下是yolov5 pose的处理代码*/
#define MAX_KEYPOINTS 17  //每个结果有最多17个关键点

//coco2017数据集的关键点的对应位置
enum Keypoints {
    //face,0-4
    NOSE = 0,
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_EAR,
    RIGHT_EAR,
    //upper body,5-10
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    //lower body,11-16
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE
};

typedef struct keypoint {
    float x;
    float y;
    float score;
} keypoint;

typedef struct yolo_pose_result
{
    float left_up_x;
    float left_up_y;
    float right_down_x;
    float right_down_y;
    int class_index;
    float score;
    keypoint keypoints[MAX_KEYPOINTS];  // 关键点信息
    struct yolo_pose_result* next;
} yolo_pose_result;






anchor_w_h anchor_grids_pose = {{{19.0f, 27.0f}, {44.0f, 40.0f}, {38.0f, 94.0f}},  // P3/8
                                {{96.0f, 68.0f}, {86.0f, 152.0f}, {180.0f, 137.0f}},  // P4/16
                                {{140.0f, 301.0f}, {303.0f, 264.0f}, {238.0f, 542.0f}},  // P5/32
                                {{436.0f, 615.0f}, {739.0f, 380.0f}, {925.0f, 792.0f}}};  // P6/64

float strides_pose[4] = {8.0f, 16.0f, 32.0f, 64.0f}; //

int map_size_pose[4] = {80, 40, 20, 10}; //


#endif // FD_CONFIG
