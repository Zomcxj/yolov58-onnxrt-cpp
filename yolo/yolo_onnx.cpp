#include "yolo_onnx.h"

// 辅助函数：将 const char* 转换为 const wchar_t*
inline const wchar_t* char_to_wchar(const char* str) {
    size_t len = strlen(str) + 1;
    wchar_t* wstr = new wchar_t[len];
    mbstowcs(wstr, str, len);
    return wstr;
}

// 辅助函数：释放 wchar_t* 内存
inline void free_wchar(const wchar_t* wstr) {
    delete[] wstr;
}

bool YolovOnnx::ReadLines(const std::string& label_file)
{
    std::ifstream ifs(label_file);
	std::string line;
	_className.clear();

	while (getline(ifs, line)) {
		if (!line.empty() && line.back() == '\r') line.pop_back();   // 兼容 CRLF
		if (!line.empty()) _className.push_back(line);
	}
    std::cout << "load " << label_file << " success !" << std::endl;      	
    return true;
}

// 类别名配置路径优先级：labelFile（--labels / 代码直接赋值）> <模型目录>/<模型名>.txt > class_0..N
bool YolovOnnx::LoadClassNames(const std::string& modelPath) {
    std::string label_file = labelFile;
    bool explicit_file = !label_file.empty();
    if (!explicit_file) {
        size_t slash = modelPath.find_last_of("/\\");
        std::string dir = (slash == std::string::npos) ? "" : modelPath.substr(0, slash + 1);
        std::string name = (slash == std::string::npos) ? modelPath : modelPath.substr(slash + 1);
        size_t dot = name.find_last_of('.');
        std::string stem = (dot == std::string::npos) ? name : name.substr(0, dot);
        label_file = dir + stem + ".txt";
    }

    std::ifstream ifs(label_file);
    if (!ifs.good()) {
        if (explicit_file) {
		std::cout << "label file not found: " << label_file << std::endl;
		return false;
        }
        _className.clear();
        for (int i = 0; i < class_num; ++i)
		_className.push_back("class_" + std::to_string(i));
        std::cout << "no label file, fallback to class_0.." << std::endl;
        return true;
    }

    if (!ReadLines(label_file)) return false;
    if ((int)_className.size() != class_num) {
        std::cout << "label_num (" << _className.size() << ") != class_num (" << class_num << ")!" << std::endl;
        return false;
    }
    return true;
}


void YolovOnnx::LetterBox(const cv::Mat& img, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape, const cv::Scalar& color)
{
	cv::Size shape = img.size();
	float r = std::min((float)newShape.height / (float)shape.height, (float)newShape.width / (float)shape.width);

	float ratio[2]{r, r};																					
	int new_un_pad[2] = {(int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r)};

	auto dw = (float)(newShape.width - new_un_pad[0]);
	auto dh = (float)(newShape.height - new_un_pad[1]);

	dw /= 2.0f;
	dh /= 2.0f;

	if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
		cv::resize(img, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
	else
		outImage = img.clone();

	int top = int(std::round(dh - 0.1f));
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(dw - 0.1f));
	int right = int(std::round(dw + 0.1f));

	params[0] = ratio[0];
	params[1] = ratio[1];
	params[2] = left;
	params[3] = top;

	cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

bool YolovOnnx::ReadModel(const std::string& modelPath, std::string task, bool isCuda, int cudaID, bool warmUp) {
    this->task = task;
	if (_batchSize < 1) _batchSize = 1;
	try
	{
		std::vector<std::string> available_providers = GetAvailableProviders();
		auto cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider");

		if (isCuda && (cuda_available == available_providers.end()))
		{
			std::cout << "Your ORT build without GPU. Change to CPU." << std::endl;
			std::cout << "************* Infer model on CPU! *************" << std::endl;
		}
		else if (isCuda && (cuda_available != available_providers.end()))
		{
			std::cout << "************* Infer model on GPU! *************" << std::endl;
			// 1.14 封装内部走 OrtApi 函数表成员（非全局符号）：CPU/GPU 包均可链接；
			// GPU 包但缺驱动/缺 CUDA dll 时 append 返回错误 → 拦截后回退 CPU，不再中断加载
			try {
				OrtCUDAProviderOptions cudaOption;
				cudaOption.device_id = cudaID;
				_OrtSessionOptions.AppendExecutionProvider_CUDA(cudaOption);
			} catch (const Ort::Exception& e) {
				std::cout << "[warn] CUDA EP unavailable, fallback to CPU: " << e.what() << std::endl;
			}
		}
		else
		{
			std::cout << "************* Infer model on CPU! *************" << std::endl;
		}

		_OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
		// _OrtSession = new Ort::Session(_OrtEnv, modelPath.c_str(), _OrtSessionOptions);

        // Windows 下 ORTCHAR_T 为 wchar_t，Linux 下为 char
#ifdef _WIN32
        const wchar_t* w_model_path = char_to_wchar(modelPath.c_str());
        _OrtSession = new Ort::Session(_OrtEnv, w_model_path, _OrtSessionOptions);
        free_wchar(w_model_path);
#else
        _OrtSession = new Ort::Session(_OrtEnv, modelPath.c_str(), _OrtSessionOptions);
#endif

		Ort::AllocatorWithDefaultOptions allocator;
		//init input
		_inputNodesNum = _OrtSession->GetInputCount();

		_inputName = std::move(_OrtSession->GetInputNameAllocated(0, allocator));
		_inputNodeNames.push_back(_inputName.get());

		// std::cout << _inputNodeNames[0] << std::endl;
		Ort::TypeInfo inputTypeInfo = _OrtSession->GetInputTypeInfo(0);
		auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
		_inputNodeDataType = input_tensor_info.GetElementType();
		_inputTensorShape = input_tensor_info.GetShape();

		if (_inputTensorShape[0] == -1)
		{
			_isDynamicShape = true;
			_inputTensorShape[0] = _batchSize;

		}
		if (_inputTensorShape[2] == -1 || _inputTensorShape[3] == -1) {
			_isDynamicShape = true;
			_inputTensorShape[2] = _netHeight;
			_inputTensorShape[3] = _netWidth;
		}
        _netHeight = _inputTensorShape[2];
		_netWidth = _inputTensorShape[3];
		//init output
		_outputNodesNum = _OrtSession->GetOutputCount();

        if (task == "seg") {
			_output_name0 = std::move(_OrtSession->GetOutputNameAllocated(0, allocator));
			_output_name1 = std::move(_OrtSession->GetOutputNameAllocated(1, allocator));
			Ort::TypeInfo type_info_output0(nullptr);
			Ort::TypeInfo type_info_output1(nullptr);
			bool flag = false;
			flag = strcmp(_output_name0.get(), _output_name1.get()) < 0;

		if (flag)  //make sure "output0" is in front of  "output1"
		{
			type_info_output0 = _OrtSession->GetOutputTypeInfo(0);  //output0
			type_info_output1 = _OrtSession->GetOutputTypeInfo(1);  //output1
			_outputNodeNames.push_back(_output_name0.get());
			_outputNodeNames.push_back(_output_name1.get());
		}
		else {
			type_info_output0 = _OrtSession->GetOutputTypeInfo(1);  //output0
			type_info_output1 = _OrtSession->GetOutputTypeInfo(0);  //output1
			_outputNodeNames.push_back(_output_name1.get());
			_outputNodeNames.push_back(_output_name0.get());
		}

		auto tensor_info_output0 = type_info_output0.GetTensorTypeAndShapeInfo();
		_outputNodeDataType = tensor_info_output0.GetElementType();
		_outputTensorShape = tensor_info_output0.GetShape();
		auto tensor_info_output1 = type_info_output1.GetTensorTypeAndShapeInfo();
		std::vector<int64_t> mask_shape = tensor_info_output1.GetShape();
		{   // seg: output0 通道 = 4+nc+maskC（v8）/ 5+nc+maskC（v5）
			bool v8seg = _outputTensorShape[1] < _outputTensorShape[2];
			int dims = v8seg ? (int)_outputTensorShape[1] : (int)_outputTensorShape[2];
			this->class_num = dims - (v8seg ? 4 : 5) - (int)mask_shape[1];
		}
        }        
        else if (task == "cls") {
		if (_outputNodesNum != 1) {
			std::cout << "Classification model should have 1 output node." << std::endl;
			return false;
		}
			_output_name0 = std::move(_OrtSession->GetOutputNameAllocated(0, allocator));
			_outputNodeNames.push_back(_output_name0.get());

		Ort::TypeInfo type_info_output0 = _OrtSession->GetOutputTypeInfo(0);
		auto tensor_info_output0 = type_info_output0.GetTensorTypeAndShapeInfo();
		_outputNodeDataType = tensor_info_output0.GetElementType();
		_outputTensorShape = tensor_info_output0.GetShape();

		// 假设分类任务的输出维度为 [batch_size, num_classes]
		if (_outputTensorShape.size() < 2) {
			std::cout << "Invalid classification model output shape." << std::endl;
			return false;
		}
		this->class_num = _outputTensorShape.back();  // 获取类别数
		if (class_num <= 1) {
			std::cout << "Classification model requires at least 2 classes." << std::endl;
			return false;
		}
		if (!LoadClassNames(modelPath)) return false;
		return true;
        }
        else {

			_output_name0 = std::move(_OrtSession->GetOutputNameAllocated(0, allocator));
			_outputNodeNames.push_back(_output_name0.get());

		Ort::TypeInfo type_info_output0(nullptr);
		type_info_output0 = _OrtSession->GetOutputTypeInfo(0);  //output0

		auto tensor_info_output0 = type_info_output0.GetTensorTypeAndShapeInfo();
		_outputNodeDataType = tensor_info_output0.GetElementType();
		_outputTensorShape = tensor_info_output0.GetShape();
        }

		// 类别数：det/obb/pose 由输出形状推导（与 OnnxBatchDetect 运行期推导一致）
		if (task != "cls" && task != "seg") {
			bool v8 = _outputTensorShape[1] < _outputTensorShape[2];   // v8: [bs, 4+nc, anchors]
			int dimensions = v8 ? (int)_outputTensorShape[1] : (int)_outputTensorShape[2];
			if (task == "det")          this->class_num = dimensions - (v8 ? 4 : 5);
			else if (task == "obb")     this->class_num = dimensions - (v8 ? 5 : 5 + 180);
			else if (task == "pose")    this->class_num = dimensions - 4 - 3 * key_point_num;
		}

		// 类别名：txt 配置或回退
		if (!LoadClassNames(modelPath)) return false;

		//warm up
		if (isCuda && warmUp) {
			//draw run
			cout << "Start warming up" << endl;
			size_t input_tensor_length = VectorProduct(_inputTensorShape);
			float* temp = new float[input_tensor_length];
			std::vector<Ort::Value> input_tensors;
			std::vector<Ort::Value> output_tensors;
			input_tensors.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo, temp, input_tensor_length, 
			_inputTensorShape.data(), _inputTensorShape.size()));
			for (int i = 0; i < 3; ++i) {
				output_tensors = _OrtSession->Run(Ort::RunOptions{nullptr},
					_inputNodeNames.data(),
					input_tensors.data(),
					_inputNodeNames.size(),
					_outputNodeNames.data(),
					_outputNodeNames.size());
			}
			delete[]temp;
		}
	}
	catch (const std::exception&) {
		return false;
	}
	return true;
}

int YolovOnnx::Preprocessing(const std::vector<cv::Mat>& srcImgs, std::vector<cv::Mat>& outSrcImgs, std::vector<cv::Vec4d>& params) {

	outSrcImgs.clear();
	cv::Size input_size = cv::Size(_netWidth, _netHeight);
	for (int i = 0; i < srcImgs.size(); ++i) {
		cv::Mat temp_img = srcImgs[i];
		cv::Vec4d temp_param = {1,1,0,0};
		if (temp_img.size() != input_size) {
			cv::Mat borderImg;
			LetterBox(temp_img, borderImg, temp_param, input_size);
			outSrcImgs.push_back(borderImg);
			params.push_back(temp_param);
		}
		else {
			outSrcImgs.push_back(temp_img);
			params.push_back(temp_param);
		}
	}

	int lack_num =  _batchSize - srcImgs.size();
	if (lack_num > 0) {
		for (int i = 0; i < lack_num; ++i) {
			cv::Mat temp_img = Mat::zeros(input_size, CV_8UC3);
			cv::Vec4d temp_param = { 1,1,0,0 };
			outSrcImgs.push_back(temp_img);
			params.push_back(temp_param);
		}
	}
	return 0;
}


bool YolovOnnx::OnnxDetect(cv::Mat& srcImg, std::vector<YolovOutput>& output) {

	std::vector<cv::Mat> input_data = {srcImg};
	std::vector<std::vector<YolovOutput>> temp_output;
	if (OnnxBatchDetect(input_data, temp_output)) {
		output = temp_output[0];
		return true;
	}
	else 
        return false;
}


bool YolovOnnx::OnnxBatchDetect(std::vector<cv::Mat>& srcImgs, std::vector<std::vector<YolovOutput>>& output) {

	std::vector<cv::Vec4d> params;
	std::vector<cv::Mat> input_images;
	cv::Size input_size(_netWidth, _netHeight);

	// pre-process
	Preprocessing(srcImgs, input_images, params);
	cv::Mat blob = cv::dnn::blobFromImages(input_images, 1 / 255.0, input_size, cv::Scalar(0, 0, 0), true, false);

	int64_t input_tensor_length = VectorProduct(_inputTensorShape);
	std::vector<Ort::Value> input_tensors;
	std::vector<Ort::Value> output_tensors;
	input_tensors.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo, (float*)blob.data, input_tensor_length, 
			            _inputTensorShape.data(), _inputTensorShape.size()));

    // infer
	output_tensors = _OrtSession->Run(Ort::RunOptions{nullptr}, _inputNodeNames.data(), input_tensors.data(),
		                                _inputNodeNames.size(), _outputNodeNames.data(), _outputNodeNames.size());

	// post-process
	// 推理期状态全部使用局部变量（不写共享成员），保证 OnnxDetect 可被多线程并发调用
	float* data = output_tensors[0].GetTensorMutableData<float>();
	std::vector<int64_t> outputTensorShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
	std::vector<int64_t> outputMaskTensorShape;
	std::vector<int> maskProtosShape;
	int64_t maskProtosLength = 0;
	int class_num = this->class_num;   // 局部副本
	int64_t one_output_length = VectorProduct(outputTensorShape) / outputTensorShape[0];    // outputTensorShape[0] = _batchSize

    // cls
    if (task == "cls") {
        if (outputTensorShape.size() < 2) {
		std::cerr << "Classification model must have at least 2D output shape (batch_size, num_classes)" << std::endl;
		return false;
        }

        int batchSize = outputTensorShape[0];
        // int classCount = _outputTensorShape[1]; // class_num
        if (batchSize != srcImgs.size()) {
		std::cerr << "Batch size mismatch between input images (" << srcImgs.size()
			      << ") and model output (" << batchSize << ")" << std::endl;
		return false;
        }

        for (int img_idx = 0; img_idx < batchSize; ++img_idx) {
		std::vector<YolovOutput> temp_output;
		const float* img_scores = data + img_idx * class_num;
		// Softmax归一化
		std::vector<float> scores(class_num);
		float sum_exp = 0.0f;
		for (int c = 0; c < class_num; ++c) {
			scores[c] = std::exp(img_scores[c]);
			sum_exp += scores[c];
		}
		for (int c = 0; c < class_num; ++c) {
			scores[c] /= sum_exp;
		}
		cv::Mat normalized_scores_mat(1, class_num, CV_32FC1, scores.data());
		// 获取最高置信度类别
		double maxVal;
		cv::Point maxLoc;
		cv::minMaxLoc(normalized_scores_mat, nullptr, &maxVal, nullptr, &maxLoc);
		std::cout << _className[maxLoc.x] << " " << static_cast<float>(maxVal) << std::endl;

		YolovOutput result;
		result.id = maxLoc.x;
		result.confidence = static_cast<float>(maxVal);
		temp_output.push_back(result);
		output.push_back(temp_output);
        }
        if (output.size())
		return true;
        else
		return false;
    }
    
    // v5
	int rows = outputTensorShape[1];
    int dimensions = outputTensorShape[2];
	cv::Mat output0 = cv::Mat(cv::Size(rows, dimensions), CV_32F, data);
    
    if (this->task == "det")
        class_num = dimensions - 5;
    else if (this->task == "obb") {
        class_num = dimensions - 5 - 180;
		output0.release();
		output0 = cv::Mat(cv::Size(dimensions, rows), CV_32F, data);
	}
    else if (this->task == "seg") 
    {
        outputMaskTensorShape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
        maskProtosShape = {1, (int)outputMaskTensorShape[1], (int)outputMaskTensorShape[2], (int)outputMaskTensorShape[3]};
        maskProtosLength = VectorProduct(maskProtosShape); 
        class_num = dimensions - 5 - (int)outputMaskTensorShape[1];
    }

	data = (float *)output0.data;

    // v8
    bool yolov8 = false;
    if (rows < dimensions)
    {
        yolov8 = true;
        rows = outputTensorShape[2];
        dimensions = outputTensorShape[1];
		output0.release();
        output0 = cv::Mat(cv::Size(rows, dimensions), CV_32F, data).t();  //[bs,84,8400]=>[bs,8400,84]
        // std::cout << output0.rows << " " << output0.cols << " " << output0.size() << std::endl;

        data = (float*)output0.data;
		if (this->task == "det")
			class_num = dimensions - 4;
		else if (this->task == "pose")
			class_num = dimensions - 4 - 3 * key_point_num;
        else if (this->task == "obb")
			class_num = dimensions - 5;  
        else if (this->task == "seg") 
        {
		outputMaskTensorShape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
		maskProtosShape = {1, (int)outputMaskTensorShape[1], (int)outputMaskTensorShape[2], (int)outputMaskTensorShape[3]};
		maskProtosLength = VectorProduct(maskProtosShape); 
		class_num = dimensions - 4 - (int)outputMaskTensorShape[1];
        }
    }
	
    // std::cout << class_num << std::endl;
	for (int img_idx = 0; img_idx < srcImgs.size(); ++img_idx) 
    {
		std::vector<int> class_ids;
		std::vector<float> confidences;
		std::vector<cv::Rect> boxes; 
        std::vector<cv::RotatedRect> obboxes; // isobb
        std::vector<std::vector<PoseKeyPoint>> pose_key_points; // pose
        std::vector<std::vector<float>> picked_proposals;   // output0[:,:, 4 + _className.size():net_width]===> for mask

        if (yolov8) 
        {
		for (int i = 0; i < rows; ++i) 
		{
			float *classes_scores = data + 4;
			cv::Mat scores(1, class_num, CV_32FC1, classes_scores);
			cv::Point class_id;
			double max_class_score;
			cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

			if (max_class_score < _classThreshold) {
			    data += dimensions;
			    continue;
			}

			class_ids.push_back(class_id.x);
			confidences.push_back(max_class_score);

			float x = (data[0] - params[img_idx][2]) / params[img_idx][0];  //x
			float y = (data[1] - params[img_idx][3]) / params[img_idx][1];  //y
			float w = data[2] / params[img_idx][0];  //w
			float h = data[3] / params[img_idx][1];  //h

			if (this->task == "obb") {
			    int angle_index = dimensions - 1;
			    float angle = data[angle_index] / CV_PI * 180.0;
			    // std::cout << data[angle_index] << "*****" << angle << std::endl;
			    obboxes.push_back(cv::RotatedRect(cv::Point2f(x, y), cv::Size(w, h), angle));
			}
			else if (this->task == "det") {
			    int left = MAX(int(x - 0.5 * w + 0.5), 0);
			    int top = MAX(int(y - 0.5 * h + 0.5), 0);
			    boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
			}
			else if (this->task == "pose") {
			    int left = MAX(int(x - 0.5 * w + 0.5), 0);
			    int top = MAX(int(y - 0.5 * h + 0.5), 0);
			    boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
			    
			    int key_point_length = dimensions - 5;
			    if (key_point_num * 3 != key_point_length) {
			        std::cout << "Pose should be shape [x, y, confidence] with 17-points" << std::endl;
			        return false;
			    }

			    std::vector<PoseKeyPoint> temp_kpts;
			    for (int kpt = 0; kpt < key_point_length; kpt += 3) {
			        PoseKeyPoint temp_kp;
			        temp_kp.x = (data[5 + kpt] - params[img_idx][2]) / params[img_idx][0];
			        temp_kp.y = (data[6 + kpt] - params[img_idx][3]) / params[img_idx][1];
			        temp_kp.confidence = data[7 + kpt];
			        temp_kpts.push_back(temp_kp);
			    }
			    pose_key_points.push_back(temp_kpts);
			}
			else if (this->task == "seg") {
			    std::vector<float> temp_proto(data + 4 + class_num, data + dimensions);
			    picked_proposals.push_back(temp_proto);
			    int left = MAX(int(x - 0.5 * w + 0.5), 0);
			    int top = MAX(int(y - 0.5 * h + 0.5), 0);
			    boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
			}
			data += dimensions;
		}
        }
        else // yolov5
        {
		for (int i = 0; i < rows; ++i) 
		{
			float confidence = data[4];
			if (confidence < _classThreshold) {
			    data += dimensions;
			    continue;
			}

			float *classes_scores = data + 5;
			cv::Mat scores(1, class_num, CV_32F, classes_scores);
			cv::Point class_id;
			double max_class_score;
			cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

			if (max_class_score < _classThreshold) {
			    data += dimensions;
			    continue;
			}

			class_ids.push_back(class_id.x);
			confidences.push_back(confidence * max_class_score);

			float x = (data[0] - params[img_idx][2]) / params[img_idx][0];  //x
			float y = (data[1] - params[img_idx][3]) / params[img_idx][1];  //y
			float w = data[2] / params[img_idx][0];  //w
			float h = data[3] / params[img_idx][1];  //h

			if (this->task == "obb") {
			    // cv::Mat output0(rows, dimensions, CV_32F, data);  // v5obb   1, 25200, 85->25200, 85
			    int nc = dimensions - 5 - 180; // number of classes
			    cv::Mat angle_scores =  output0.row(i).colRange(5 + nc, dimensions);
			    cv::Point angle_id;
			    double max_angle_score;
			    cv::minMaxLoc(angle_scores, 0, &max_angle_score, 0, &angle_id);
			    float angle = 90 - angle_id.x;
			    // std::cout << angle_id.x << "*****" << angle << std::endl;
			    obboxes.push_back(cv::RotatedRect(cv::Point2f(x, y), cv::Size2f(w, h), angle));
			}
			else if (this->task == "det") {
			    int left = MAX(int(x - 0.5 * w + 0.5), 0);
			    int top = MAX(int(y - 0.5 * h + 0.5), 0);
			    boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
			}
			else if (this->task == "seg") {
			    std::vector<float> temp_proto(data + 5 + class_num, data + dimensions);
			    picked_proposals.push_back(temp_proto);
			    int left = MAX(int(x - 0.5 * w + 0.5), 0);
			    int top = MAX(int(y - 0.5 * h + 0.5), 0);
			    boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
			}
			data += dimensions;
		}
		}

		std::vector<int> nms_result;
        if (this->task == "obb") 
		    cv::dnn::NMSBoxes(obboxes, confidences, _classThreshold, _nmsThreshold, nms_result);
        else
		cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);

        std::vector<std::vector<float>> temp_mask_proposals; // seg
		cv::Rect holeImgRect(0, 0, srcImgs[img_idx].cols, srcImgs[img_idx].rows); // 定义一个图片大小的box

		std::vector<YolovOutput> temp_output;
		for (int i = 0; i < nms_result.size(); ++i) 
        {
			int idx = nms_result[i];
			YolovOutput result;
			result.id = class_ids[idx];
			result.confidence = confidences[idx];

		if (this->task == "obb")
			result.rotatedBox = obboxes[idx];
		else if (this->task == "det")
			result.box = boxes[idx] & holeImgRect;
		else if (this->task == "pose") {
			result.box = boxes[idx] & holeImgRect;
			result.keyPoints = pose_key_points[idx];
		}
		else if (this->task == "seg") {
			result.box = boxes[idx] & holeImgRect;
			temp_mask_proposals.push_back(picked_proposals[idx]);
		}

			temp_output.push_back(result);
		}

        if (this->task == "seg") {
		MaskParams mask_params;
		mask_params.params = params[img_idx];
		mask_params.srcImgShape = srcImgs[img_idx].size();
		mask_params.netHeight = _netHeight;
		mask_params.netWidth = _netWidth;
		mask_params.maskThreshold = _maskThreshold;
		cv::Mat mask_protos = cv::Mat(maskProtosShape, CV_32F, output_tensors[1].GetTensorMutableData<float>() + img_idx * maskProtosLength);
		for (int i = 0; i < temp_mask_proposals.size(); ++i) {
			GetMask(cv::Mat(temp_mask_proposals[i]).t(), mask_protos, temp_output[i], mask_params);
		}
        }

		output.push_back(temp_output);
	}

	if (output.size())
		return true;
	else
		return false;

}


void YolovOnnx::DrawPred(cv::Mat& img, std::vector<YolovOutput> result) {

    double scale;
    int thickness;

    if (img.cols > 3000)
        scale = 1.5, thickness = 2;
    else
        scale = 0.5, thickness = 1;

    cv::Mat mask = img.clone();

    for (int i = 0; i < result.size(); i++) 
    {
        if (result[i].confidence > _classThreshold)
        {   
		int idx = result[i].id + 3;
		cv::Scalar color = cv::Scalar(37 * idx * idx % 255, 17 * idx * idx % 255, 29 * idx * idx % 255);

		int left, top;
		if (result[i].box.area() > 0) {
			cv::rectangle(img, result[i].box, color, thickness, cv::LINE_AA);
			left = result[i].box.x;
			top = result[i].box.y;
		}
		else if (result[i].rotatedBox.size.area() > 0) { // OBB
			cv::Point2f p[4];
			result[i].rotatedBox.points(p);
			for (int l = 0; l < 4; l++)  {
			    cv::line(img, p[l], p[(l + 1) % 4], cv::Scalar(100, 100, 255), thickness, cv::LINE_AA);
			}
			left = result[i].rotatedBox.center.x;
			top = result[i].rotatedBox.center.y;
		}

		std::string conf = cv::format("%.2f", result[i].confidence);
		// 标签按开关拼装：类别 / 置信度 / 跟踪 id（默认只显示类别名）
		std::string label;
		if (drawClass) label += _className[result[i].id];
		if (drawConf)  { if (!label.empty()) label += " "; label += conf; }
		if (drawId && result[i].track_id > 0) { if (!label.empty()) label += " "; label += "id:" + std::to_string(result[i].track_id); }

		if (!label.empty()) {
			int baseLine;
			cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_COMPLEX, scale, thickness, &baseLine);
			cv::rectangle(img, cv::Point(left, top - labelSize.height - 2), cv::Point(left + labelSize.width, top - 2), color, -1, cv::LINE_AA);
			cv::putText(img, label, cv::Point(left, top - 2), cv::FONT_HERSHEY_COMPLEX, scale, cv::Scalar(225, 255, 255), thickness, cv::LINE_AA);
		}

		std::cout << _className[result[i].id] << " @ " << conf << std::endl;

		if (this->task == "pose") {
			if (result[i].keyPoints.size()!= poseparams.kptBodyNames.size())
			continue;

			for (int j = 0; j < result[i].keyPoints.size(); ++j) {
			    PoseKeyPoint kpt = result[i].keyPoints[j];
			    if (kpt.confidence < poseparams.kptThreshold)
			        continue;
			    cv::Scalar kptColor = poseparams.posePalette[poseparams.kptColor[j]];   // three color
			    cv::circle(img, cv::Point(kpt.x, kpt.y), poseparams.kptRadius, kptColor, -1, 8);
			}

			if (poseparams.isDrawKptLine) {
			    for (int j = 0; j < poseparams.skeleton.size(); ++j) {
			        PoseKeyPoint kpt0 = result[i].keyPoints[poseparams.skeleton[j][0] - 1];
			        PoseKeyPoint kpt1 = result[i].keyPoints[poseparams.skeleton[j][1] - 1];
			        if (kpt0.confidence < poseparams.kptThreshold || kpt1.confidence < poseparams.kptThreshold)
			            continue;
			        cv::Scalar kptColor= poseparams.posePalette[poseparams.limbColor[j]];
			        cv::line(img, cv::Point(kpt0.x, kpt0.y),cv::Point(kpt1.x, kpt1.y),  kptColor, 2, 8);
			    }
			}
		}
		else if (this->task == "seg")
			mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);
        }
	}
    if (this->task == "seg")
        cv::addWeighted(img, 0.5, mask, 0.8, 0, img);
}

void YolovOnnx::GetMask(const cv::Mat& maskProposals, const cv::Mat& maskProtos, YolovOutput& output, const MaskParams& maskParams) {
	int net_width = maskParams.netWidth;
	int net_height = maskParams.netHeight;
	int seg_channels = maskProtos.size[1];
	int seg_height = maskProtos.size[2];
	int seg_width = maskProtos.size[3];
	float mask_threshold = maskParams.maskThreshold;
	cv::Vec4f params = maskParams.params;
	cv::Size src_img_shape = maskParams.srcImgShape;
	cv::Rect temp_rect = output.box;

	//crop from mask_protos
	int rang_x = floor((temp_rect.x * params[0] + params[2]) / net_width * seg_width);
	int rang_y = floor((temp_rect.y * params[1] + params[3]) / net_height * seg_height);
	int rang_w = ceil(((temp_rect.x + temp_rect.width) * params[0] + params[2]) / net_width * seg_width) - rang_x;
	int rang_h = ceil(((temp_rect.y + temp_rect.height) * params[1] + params[3]) / net_height * seg_height) - rang_y;

	rang_w = MAX(rang_w, 1);
	rang_h = MAX(rang_h, 1);
	if (rang_x + rang_w > seg_width) {
		if (seg_width - rang_x > 0)
			rang_w = seg_width - rang_x;
		else
			rang_x -= 1;
	}
	if (rang_y + rang_h > seg_height) {
		if (seg_height - rang_y > 0)
			rang_h = seg_height - rang_y;
		else
			rang_y -= 1;
	}

	std::vector<cv::Range> roi_rangs;
	roi_rangs.push_back(cv::Range(0, 1));
	roi_rangs.push_back(cv::Range::all());
	roi_rangs.push_back(cv::Range(rang_y, rang_h + rang_y));
	roi_rangs.push_back(cv::Range(rang_x, rang_w + rang_x));

	//crop
	cv::Mat temp_mask_protos = maskProtos(roi_rangs).clone();
	cv::Mat protos = temp_mask_protos.reshape(0, { seg_channels,rang_w * rang_h });
	cv::Mat matmul_res = (maskProposals * protos).t();
	cv::Mat masks_feature = matmul_res.reshape(1, { rang_h,rang_w });
	cv::Mat dest, mask;

	//sigmoid
	cv::exp(-masks_feature, dest);
	dest = 1.0 / (1.0 + dest);

	int left = floor((net_width / seg_width * rang_x - params[2]) / params[0]);
	int top = floor((net_height / seg_height * rang_y - params[3]) / params[1]);
	int width = ceil(net_width / seg_width * rang_w / params[0]);
	int height = ceil(net_height / seg_height * rang_h / params[1]);

	resize(dest, mask, cv::Size(width, height), cv::INTER_NEAREST);
	cv::Rect mask_rect = temp_rect - cv::Point(left, top);
	mask_rect &= cv::Rect(0, 0, width, height);
	mask = mask(mask_rect) > mask_threshold;
	if (mask.rows != temp_rect.height || mask.cols != temp_rect.width) {
		resize(mask, mask, temp_rect.size(), cv::INTER_NEAREST);
	}
	output.boxMask = mask;

}