#pragma once
// 定义 MSVC 特有的宏以兼容 MinGW 编译器
#ifndef _Frees_ptr_opt_
#define _Frees_ptr_opt_
#endif

#ifndef _Outptr_result_buffer_maybenull_
#define _Outptr_result_buffer_maybenull_(...)
#endif

#ifndef _Inout_
#define _Inout_
#endif

#ifndef _Out_
#define _Out_
#endif

#ifndef _Check_return_
#define _Check_return_
#endif

#ifndef _Ret_maybenull_
#define _Ret_maybenull_
#endif

#ifndef ORT_MUST_USE_RESULT
#define ORT_MUST_USE_RESULT
#endif

#include <iostream>
#include <memory>
#include <numeric>
#include <math.h>
#include <dirent.h>
#include <fstream>  
#include <opencv2/opencv.hpp>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <sys/time.h>

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace Ort;

#define ORT_OLD_VISON 13  //ort1.12.0

struct MaskParams {
	int netWidth = 640;
	int netHeight = 640;
	float maskThreshold = 0.5;
	cv::Size srcImgShape;
	cv::Vec4d params;
};

struct PoseKeyPoint {
	float x = 0;
	float y = 0;
	float confidence = 0;
};

struct PoseParams {
	float kptThreshold = 0.5;
	int kptRadius = 5;
	bool isDrawKptLine = true;  // If True, the function will draw lines connecting keypoint for human pose.Default is True.
	std::vector<std::vector<int>> skeleton = {
		{16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, 
        {6, 12}, {7, 13}, {6, 7}, {6, 8}, {7, 9}, {8, 10}, {9, 11}, 
        {2, 3}, {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}};

	std::vector<cv::Scalar> posePalette = {
        cv::Scalar(255, 128, 0), cv::Scalar(255, 153, 51), cv::Scalar(255, 178, 102),
        cv::Scalar(230, 230, 0), cv::Scalar(255, 153, 255), cv::Scalar(153, 204, 255),
        cv::Scalar(255, 102, 255), cv::Scalar(255, 51, 255), cv::Scalar(102, 178, 255),
        cv::Scalar(51, 153, 255), cv::Scalar(255, 153, 153), cv::Scalar(255, 102, 102),
        cv::Scalar(255, 51, 51), cv::Scalar(153, 255, 153), cv::Scalar(102, 255, 102),
        cv::Scalar(51, 255, 51), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
        cv::Scalar(255, 0, 0), cv::Scalar(255, 255, 255)};

	std::vector<int> limbColor = { 9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16 };
	std::vector<int> kptColor = { 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9 };
	std::map<unsigned int, std::string> kptBodyNames {
		{0,"Nose"},
		{1,	"left_eye"},		{2,	"right_eye"},
		{3,	"left_ear"},		{4,	"right_ear"},
		{5,	"left_shoulder"},	{6,	"right_shoulder"},
		{7,	"left_elbow"},		{8,	"right_elbow"},
		{9,	"left_wrist"},		{10,"right_wrist"},
		{11,"left_hip"},		{12,"right_hip"},
		{13,"left_knee"},		{14,"right_knee"},
		{15,"left_ankle"},		{16,"right_ankle"}};
};


struct YolovOutput {
	int id;             
	float confidence;  
	cv::Rect box;                // hbb结果矩形框
    cv::RotatedRect rotatedBox;  // obb结果矩形框
	cv::Mat boxMask;             // 掩码图
	std::vector<PoseKeyPoint> keyPoints; // 位姿
    int track_id = -1;
};


class YolovOnnx {
public:
	YolovOnnx() :_OrtMemoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPUOutput)) {};
	~YolovOnnx() {
		if (_OrtSession != nullptr)
			delete _OrtSession;
	}; // delete _OrtMemoryInfo;


public:
	/** \brief Read onnx-model
	*   \param[in] modelPath:onnx-model path
	*   \param[in] isCuda:if true,use Ort-GPU,else run it on cpu.
	*   \param[in] cudaID:if isCuda==true,run Ort-GPU on cudaID.
	*   \param[in] warmUp:if isCuda==true,warm up GPU-model.
	*/
	bool ReadModel(const std::string& modelPath, std::string task, bool isCuda = false, int cudaID = 0, bool warmUp = true);
	/** \brief  detect.
	*   \param[in] srcImg:a 3-channels image.
	*   \param[out] output:detection results of input image.
    *   \param[out] isobb: rotated bbox
	*/
	bool OnnxDetect(cv::Mat& srcImg, std::vector<YolovOutput>& output);

	/** \brief  detect, batch size= _batchSize
	*   \param[in] srcImg:A batch of images.
	*   \param[out] output:detection results of input images.
    *   \param[out] isobb: rotated bbox
	*/
	bool OnnxBatchDetect(std::vector<cv::Mat>& srcImg, std::vector<std::vector<YolovOutput>>& output);
    void DrawPred(cv::Mat& img, std::vector<YolovOutput> result);
    void GetMask(const cv::Mat& maskProposals, const cv::Mat& maskProtos, YolovOutput& output, const MaskParams& maskParams);
    bool ReadLines(const std::string& label_file);

private:

	void LetterBox(const cv::Mat& img, cv::Mat& outImage,
		cv::Vec4d& params, //[ratio_x,ratio_y, dw, dh]
		const cv::Size& newShape = cv::Size(640, 640),
		const cv::Scalar& color = cv::Scalar(114, 114, 114));

	template <typename T>
	T VectorProduct(const std::vector<T>& v)
	{
		return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
	};
	int Preprocessing(const std::vector<cv::Mat>& srcImgs, std::vector<cv::Mat>& outSrcImgs, std::vector<cv::Vec4d>& params);

	int _netWidth = 640;   // ONNX-net-input-width
	int _netHeight = 640;  // ONNX-net-input-height

	bool _isDynamicShape = false;// onnx support dynamic shape
	float _classThreshold = 0.6;
	float _nmsThreshold = 0.6;
	float _maskThreshold = 0.5;
    int key_point_num = 17;  //_bodyKeyPoints.size(), shape[x, y, confidence]
	int class_num = 80;
    std::string task;
	PoseParams poseparams;

	// ONNXRUNTIME	
	Ort::Env _OrtEnv = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Yolov");
	Ort::SessionOptions _OrtSessionOptions = Ort::SessionOptions();
	Ort::Session* _OrtSession = nullptr;
	Ort::MemoryInfo _OrtMemoryInfo;

#if ORT_API_VERSION < ORT_OLD_VISON
    char* _inputName, * _output_name0, * _output_name1;
#else
    std::shared_ptr<char> _inputName, _output_name0, _output_name1;
#endif

	std::vector<char*> _inputNodeNames; 
	std::vector<char*> _outputNodeNames;

	size_t _inputNodesNum = 0;      
	size_t _outputNodesNum = 0;    

	ONNXTensorElementDataType _inputNodeDataType; 
	ONNXTensorElementDataType _outputNodeDataType;
	std::vector<int64_t> _inputTensorShape;
	std::vector<int64_t> _outputTensorShape;

    // seg
    std::vector<int64_t> _outputMaskTensorShape;
    std::vector<int> mask_protos_shape;
    int mask_protos_length;

public:

	int _batchSize = 1;          // if multi-batch, set this
    // std::vector<std::string> _className = {"blur", "fog", "night", "normal", "rain", "snow"};
	// std::vector<std::string> _className = {"NorRider", "NoHelmet", "HasPassenger", "Head", "ReRider", "SunShade"};
    // std::vector<std::string> _className = {"Li", "Heng", "Zong", "Truck"};
    // std::vector<std::string> _className = {"SCar", "MCar", "BCar", "Rider", "Person"};
    // std::vector<std::string> _className = {"vertical", "horizon", "lengthways", "Truck"};
    //  std::vector<std::string> _className = {"LaoTouLe", "LaoTouLe", "Car", "ECar"};
    // std::vector<std::string> _className = {"scar", "bcar"};
    // std::vector<std::string> _className = {"light", "leftwait"};
    // std::vector<std::string> _className = {"red", "yellow", "green", "black", "lvbo", "lvbono"};
    // std::vector<std::string> _className = {"person"};
    std::vector<std::string> _className = {"drone"};
    // std::vector<std::string> _className = {
    //     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    //     "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    //     "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    //     "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    //     "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    //     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    //     "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    //     "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    //     "hair drier", "toothbrush"
    // };

    // std::vector<std::string> _className = {
	// 	"plane", "baseball-diamond", "bridge", "ground-track-field", 
	// 	"small-vehicle", "large-vehicle", "ship", "tennis-court", 
	// 	"basketball-court", "storage-tank", "soccer-ball-field", "roundabout",
	// 	"harbor", "swimming-pool", "helicopter", "container-crane"};
};
