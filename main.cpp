#include <iostream>	
#include <fstream>
#include<vector>
#include<opencv2/opencv.hpp>
#include "YOLO.h"
#include "NvInferPlugin.h"
using namespace std;

int Test01()
{
	//第一步读取文件-----------------------------------------------------------
	string fileName = "./model/best.engine";
	ifstream file(fileName, ios::binary); //二进制方式打开文件
	char* trtModelStream = NULL; //创建字符指针
	auto size = 0;
	if (file.good())
	{
		file.seekg(0, ios::end); //将文件指针移动到文件末尾
		size= file.tellg(); //获取文件大小
		file.seekg(0, ios::beg); //将文件指针移动到文件开头
        trtModelStream = new char[size]; //根据文件大小创建字符数组
		file.read(trtModelStream, size); //将文件内容读取到字符数组中
		file.close(); //关闭文件流
	}
	//-------------------------------------------------------------------------

	//第二步创建Logger进行反序列化-----------------------------------------------
	Logger gLogger; //创建Logger
	initLibNvInferPlugins(&gLogger, "");
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger); //创建反序列化引擎runtime
	if (runtime == nullptr) cout << "runtime is null" << endl;
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size); //反序列化模型
	if (engine == nullptr) cout << "runtime is null" << endl;

	delete[] trtModelStream; //删除字符指针

	//创建上下文
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
	if (context == nullptr) cout << "runtime is null" << endl;
	//-------------------------------------------------------------------------

	//第三步获取输入输出---------------------------------------------------------
	int num_bindings = 0;
	num_bindings= engine->getNbIOTensors(); //TRT_10接口 TRT.8接口为 engine->getNbBindings();


	vector<string> input_names;	//输入节点名称
	vector<int> input_size; //输入节点大小
	vector<string> output_names; //存放输出节点名称
	vector<int> output_size; //输出节点大小

	for (int i = 0; i < num_bindings; i++)
	{
		nvinfer1::Dims dims;
		string name = engine->getIOTensorName(i); //TRT_10接口 TRT.8接口为 engine->getBindingName(i);
		nvinfer1::DataType dtype = engine->getTensorDataType(name.c_str()); //TRT_10接口 TRT.8接口为 engine->getBindingDataType();
		int dsize= type_to_size(dtype); //获取数据类型大小

		//获取当前节点数据类型后判断是否为输入节点
		bool isInput = engine->getTensorIOMode(name.c_str())==nvinfer1::TensorIOMode::kINPUT; //TRT_10接口 TRT.8接口为 engine->bindingIsInput();
		if (isInput)
		{
			dims= engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
			context->setInputShape(name.c_str(), dims);//TRT_10接口 TRT.8接口为 engine->setBindingDimensions();
			cout << "input name: " << name << " dims: " << dims.nbDims
				<< " input shape:("<< dims.d[0] << "," << dims.d[1] << "," 
				<< dims.d[2] << "," << dims.d[3] <<")" << endl;

			input_names.push_back(name);

			//计算所需内存
			int inputSize = 1;
			for (int j = 0; j < dims.nbDims; j++)
			{
				inputSize *= dims.d[j];
			}
			input_size.push_back(inputSize*dsize);
		}
		else
		{
			//输出节点处理
			dims=context->getTensorShape(name.c_str());
			cout << "ouput name: " << name << " dims: " << dims.nbDims
				<< " ouput shape:(" << dims.d[0] << "," << dims.d[1] << ","
				<< dims.d[2] << "," << dims.d[3] << ")" << endl;

			output_names.push_back(name);

			int size= 1;
			for (int j = 0; j < dims.nbDims; j++)
			{
				size *= dims.d[j];
			}
			output_size.push_back(size * dsize);
		}
	}
	//----------------------------------------------------------------------------


	//第四步分配GPU内存并进行cuda与主机内存传输---------------------------------------
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	vector<void*> d_ptrs; //device 指针
	for (int i = 0; i < input_names.size(); i++)
	{
		void* d_ptr;
		cudaMallocAsync(&d_ptr, input_size[i], stream); //计算输入节点内存
		d_ptrs.push_back(d_ptr); 
		
		auto name = input_names[i].c_str();
		context->setInputShape(name, engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX));
		context->setTensorAddress(name, d_ptr); //TRT10接口 需要设置输入的张量地址
	}

	vector<void*> h_ptrs; //主机指针>
	for (int i = 0; i < output_names.size(); i++)
	{
		void* d_ptr,*h_ptr; //device 和主机数据拷贝指针用于将GPU数据拷贝到主机
		cudaMallocAsync(&d_ptr, output_size[i], stream); //计算输入节点内存
		cudaHostAlloc(&h_ptr, output_size[i], 0);//锁定主机内存供GPU直接调用
		d_ptrs.push_back(d_ptr);
		h_ptrs.push_back(h_ptr);

		auto name = output_names[i].c_str();
		context->setTensorAddress(name, d_ptr);
	}
	//----------------------------------------------------------------------------

	//可选步骤(预热)---------------------------------------------------------------
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < input_names.size(); j++)
		{
			int size = input_size[j];
			void* h_ptr=(void*)malloc(size); //开辟主机内存
			memset(h_ptr, 0, size); //初始化内存
            cudaMemcpyAsync(d_ptrs[j], h_ptr, size, cudaMemcpyHostToDevice, stream); //将主机数据拷贝到GPU
			free (h_ptr);
		}
		cudaStreamSynchronize(stream); //同步机制

		//执行推理
		context->enqueueV3(stream); //TRT10接口 TRT.8接口为 context->enqueueV2();
		for (int j = 0; j < output_names.size(); j++) 
		{
			int size = output_size[j];
            cudaMemcpyAsync(h_ptrs[j], d_ptrs[j+input_names.size()], size, cudaMemcpyDeviceToHost, stream); //将GPU数据拷贝到主机
		}
		cudaStreamSynchronize(stream);
		cout<<"Warmup "<<i+1<<" times......"<<endl;
	}
	//----------------------------------------------------------------------------

    //第五步执行推理---------------------------------------------------------------
	string imagePath="D:\\Desktop\\Data\\shizhi\\AI\\background";//测试图片路径文件夹
	vector<string> imageFiles;
	cv::glob(imagePath, imageFiles);
	vector<cv::Mat> images;
	for (int i = 0; i < imageFiles.size(); i++)
	{
		images.push_back(cv::imread(imageFiles[i]));
	}

	cv::Size inputSize= cv::Size(640, 640); //输入模型图片尺寸
	for (int i = 0; i < images.size(); i++)
	{
		//输入图片前处理
		cv::Mat inferImg = images[i].clone();
		int width= inferImg.cols;
		int height = inferImg.rows;
		int inp_h= inputSize.height;
		int inp_w= inputSize.width;

		float ratio= std::min((float)inp_h/height,(float)inp_w/width); //计算图像缩放比例因子
		int pad_h= (int)round(height*ratio); //计算缩放后高宽
		int pad_w= (int)round(width*ratio);

		cv::Mat tmp;
		if ((int)width != pad_w || (int)height != pad_h) {
			cv::resize(inferImg, tmp, cv::Size(pad_w, pad_h)); //resize只会按固定长宽比进行放缩
		}
		else {
			tmp = inferImg.clone();
		}

		float dw = inp_w - pad_w;
		float dh = inp_h - pad_h; //计算填充高宽

		dw /= 2.0f; //包围式填充
		dh /= 2.0f;
		int top = int(std::round(dh - 0.1f));
		int bottom = int(std::round(dh + 0.1f));
		int left = int(std::round(dw - 0.1f));
		int right = int(std::round(dw + 0.1f));

		cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, { 114, 114, 114 }); //yolo系列填充114

		cv::Mat out;
		out.create({ 1, 3, (int)inp_h, (int)inp_w }, CV_32F);

		std::vector<cv::Mat> channels;
		cv::split(tmp, channels);

		cv::Mat c0((int)inp_h, (int)inp_w, CV_32F, (float*)out.data);
		cv::Mat c1((int)inp_h, (int)inp_w, CV_32F, (float*)out.data + (int)inp_h * (int)inp_w);
		cv::Mat c2((int)inp_h, (int)inp_w, CV_32F, (float*)out.data + (int)inp_h * (int)inp_w * 2);


		//BGR通道转为BGR
		channels[0].convertTo(c2, CV_32F, 1 / 255.f);
		channels[1].convertTo(c1, CV_32F, 1 / 255.f); 
		channels[2].convertTo(c0, CV_32F, 1 / 255.f);

		//前处理完成
		//分配内存
		for (int i = 0; i < input_names.size(); i++)
		{
			cudaMemcpyAsync(d_ptrs[i], out.ptr<float>(), out.total() * out.elemSize(), cudaMemcpyHostToDevice, stream);
			auto name= input_names[i].c_str();
			context->setInputShape(name, { 1, 3, (int)inp_h, (int)inp_w });
			context->setTensorAddress(name, d_ptrs[i]);
		}
		cudaStreamSynchronize(stream);

		//开始推理
		context->enqueueV3(stream); //TRT10接口 TRT.8接口为 context->enqueueV2();
		for (int j = 0; j < output_names.size(); j++)
		{
			int size = output_size[j];
			cudaMemcpyAsync(h_ptrs[j], d_ptrs[j + input_names.size()], size, cudaMemcpyDeviceToHost, stream); //将GPU数据拷贝到主机
		}
		cudaStreamSynchronize(stream);
		//----------------------------------------------------------------------------

		//进行后处理（根据自己模型的输出）-----------------------------------------------
		//cerr << "Model output nums:" << h_ptrs.size() << endl;
		//for (int i = 0; i < h_ptrs.size(); i++)
		//{
		//	//printf("Number:{%d} output name is %s\n", i, output_names[i].c_str());
		//	//cerr<<h_ptrs[i]<<endl;

		//	auto *data = (float*)h_ptrs[i];
		//}
		//----------------------------------------------------------------------------

		int* num_dets = static_cast<int*>(h_ptrs[0]); //检测框数量
		auto* boxes = static_cast<float*>(h_ptrs[1]); //检测框坐标--float 型指针
		auto* scores = static_cast<float*>(h_ptrs[2]);//检测框得分
		int* labels = static_cast<int*>(h_ptrs[3]);//检测框类别

		cerr<<"Number of detected objects: " << *num_dets << endl;
		printf("Boexs:(%.f,%.f,%.f,%.f)=(x0,y0,x1,y1)\n", *boxes++, *boxes++, *boxes++, *boxes);
		cerr<<"Scores: " << *scores << endl;
		cerr<<"Labels: " << *labels << endl;
	}
	

	return 0;
}

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
	Test01();
	return 0;
}
