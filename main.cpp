#include <iostream>	
#include <fstream>
#include<vector>
#include<opencv2/opencv.hpp>
#include "YOLO.h"
#include "NvInferPlugin.h"
using namespace std;

void Test02()
{
	string fileName = "./model/best.engine";
	YOLOv8 yolo(fileName);
	yolo.WarmUp(10);
	string imagePath = "D:\\Desktop\\Data\\Fangdaqi\\test";//测试图片路径文件夹
	vector<string> imageFiles;
	cv::glob(imagePath, imageFiles);
	for (int i = 0; i < imageFiles.size(); i++)
	{
		cv::Mat srcImg = cv::imread(imageFiles[i]);
		if (srcImg.empty())
		{
			cerr<<"------------------"<<i<<endl;
			continue;
		}
		auto start = std::chrono::system_clock::now();
		vector<Box> detRes;
        yolo.Detect(srcImg,detRes);
		auto end = std::chrono::system_clock::now();
		auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
		printf("Number %d inference takes time: %.3f ms\n", i, tc);
	}
}

int main()
{
	Test02();
	return 0;
}
