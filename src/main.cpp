#include "obb.h"
#include "yolo_onnx.h"

int main(int argc, char** argv) {

    std::string model_path = argv[1];
	std::string img_path = argv[2];
    std::string task = argv[3];    // {"det", "obb", "pose", "seg", "cls"}; 
 

    bool iscrop = false;
    std::string savepath = "../output/";	

    YolovOnnx yolo;
    if(!yolo.ReadModel(model_path, task, false)) {
        std::cout << "cannot load model" << std::endl;
        return -1; 
    }

    // std::string label_file = argv[4];   
    // if (!yolo.ReadDict(label_file)) {
    //     std::cerr << "load dict error !" << std::endl;
    //     return;
    // }

    std::vector<std::string> fn;
    cv::glob(img_path, fn, false);
    for(int j = 0; j < fn.size(); j++) 
    {
		cv::Mat img = cv::imread(fn[j]);
		std::string::size_type iPos = fn[j].find_last_of('\\') + 1;
        std::string filename = fn[j].substr(iPos, fn[j].length() - iPos);
        std::string save_path = savepath + filename;
		std::cout << filename << std::endl;

        cv::Mat resImg = img.clone();
		// cv::Mat resImg;
		// cv::resize(img, resImg, cv::Size(1920, 1080));

        struct timeval t1, t2;
        gettimeofday(&t1,NULL);

        if (iscrop) 
		{		
            int cropSize = 1280;   // same cropSize onnx640 200ms total faster than onnx1024
            float conf = 0.5;

			std::vector<std::vector<int>> roiXY;
			roiXY = imgSplit(img, cropSize);

            std::vector<YolovOutput> postResult;
            for (int i = 0; i < roiXY.size(); i++) 
			{             
                cv::Mat subImg = resImg(cv::Rect(roiXY[i][0], roiXY[i][1], roiXY[i][2], roiXY[i][3]));
				std::vector<YolovOutput> result;
                yolo.OnnxDetect(subImg, result);
                if (task == "obb")
                    obboxFilter(result, postResult, roiXY[i][0], roiXY[i][1], conf);
                else if (task == "det")
                    hbboxFilter(result, postResult, roiXY[i][0], roiXY[i][1], conf);
			}
			yolo.DrawPred(resImg, postResult); 
            cv::imwrite(save_path, resImg);
        }
		else
		{
            std::vector<YolovOutput> result;
			yolo.OnnxDetect(resImg, result);

            if (task != "cls") {
                yolo.DrawPred(resImg, result);    // det+draw 8~10ms
                cv::imwrite(save_path, resImg);  // save 12~14ms
            }
        }
        gettimeofday(&t2,NULL);
        int loadtime = (t2.tv_sec - t1.tv_sec) * 1000 + (int)(t2.tv_usec - t1.tv_usec) / 1000;	
        std::cout << "sum time: "<< loadtime << " ms" << std::endl;
        std::cout << "-----------------------" << std::endl;
    }
	return 0;
}


