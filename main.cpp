#include <iostream>	
#include <fstream>
#include<vector>
#include<opencv2/opencv.hpp>
#include "YOLO.h"
#include "NvInferPlugin.h"
using namespace std;

int Test01()
{
	//��һ����ȡ�ļ�-----------------------------------------------------------
	string fileName = "./model/best.engine";
	ifstream file(fileName, ios::binary); //�����Ʒ�ʽ���ļ�
	char* trtModelStream = NULL; //�����ַ�ָ��
	auto size = 0;
	if (file.good())
	{
		file.seekg(0, ios::end); //���ļ�ָ���ƶ����ļ�ĩβ
		size= file.tellg(); //��ȡ�ļ���С
		file.seekg(0, ios::beg); //���ļ�ָ���ƶ����ļ���ͷ
        trtModelStream = new char[size]; //�����ļ���С�����ַ�����
		file.read(trtModelStream, size); //���ļ����ݶ�ȡ���ַ�������
		file.close(); //�ر��ļ���
	}
	//-------------------------------------------------------------------------

	//�ڶ�������Logger���з����л�-----------------------------------------------
	Logger gLogger; //����Logger
	initLibNvInferPlugins(&gLogger, "");
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger); //���������л�����runtime
	if (runtime == nullptr) cout << "runtime is null" << endl;
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size); //�����л�ģ��
	if (engine == nullptr) cout << "runtime is null" << endl;

	delete[] trtModelStream; //ɾ���ַ�ָ��

	//����������
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
	if (context == nullptr) cout << "runtime is null" << endl;
	//-------------------------------------------------------------------------

	//��������ȡ�������---------------------------------------------------------
	int num_bindings = 0;
	num_bindings= engine->getNbIOTensors(); //TRT_10�ӿ� TRT.8�ӿ�Ϊ engine->getNbBindings();


	vector<string> input_names;	//����ڵ�����
	vector<int> input_size; //����ڵ��С
	vector<string> output_names; //�������ڵ�����
	vector<int> output_size; //����ڵ��С

	for (int i = 0; i < num_bindings; i++)
	{
		nvinfer1::Dims dims;
		string name = engine->getIOTensorName(i); //TRT_10�ӿ� TRT.8�ӿ�Ϊ engine->getBindingName(i);
		nvinfer1::DataType dtype = engine->getTensorDataType(name.c_str()); //TRT_10�ӿ� TRT.8�ӿ�Ϊ engine->getBindingDataType();
		int dsize= type_to_size(dtype); //��ȡ�������ʹ�С

		//��ȡ��ǰ�ڵ��������ͺ��ж��Ƿ�Ϊ����ڵ�
		bool isInput = engine->getTensorIOMode(name.c_str())==nvinfer1::TensorIOMode::kINPUT; //TRT_10�ӿ� TRT.8�ӿ�Ϊ engine->bindingIsInput();
		if (isInput)
		{
			dims= engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
			context->setInputShape(name.c_str(), dims);//TRT_10�ӿ� TRT.8�ӿ�Ϊ engine->setBindingDimensions();
			cout << "input name: " << name << " dims: " << dims.nbDims
				<< " input shape:("<< dims.d[0] << "," << dims.d[1] << "," 
				<< dims.d[2] << "," << dims.d[3] <<")" << endl;

			input_names.push_back(name);

			//���������ڴ�
			int inputSize = 1;
			for (int j = 0; j < dims.nbDims; j++)
			{
				inputSize *= dims.d[j];
			}
			input_size.push_back(inputSize*dsize);
		}
		else
		{
			//����ڵ㴦��
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


	//���Ĳ�����GPU�ڴ沢����cuda�������ڴ洫��---------------------------------------
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	vector<void*> d_ptrs; //device ָ��
	for (int i = 0; i < input_names.size(); i++)
	{
		void* d_ptr;
		cudaMallocAsync(&d_ptr, input_size[i], stream); //��������ڵ��ڴ�
		d_ptrs.push_back(d_ptr); 
		
		auto name = input_names[i].c_str();
		context->setInputShape(name, engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX));
		context->setTensorAddress(name, d_ptr); //TRT10�ӿ� ��Ҫ���������������ַ
	}

	vector<void*> h_ptrs; //����ָ��>
	for (int i = 0; i < output_names.size(); i++)
	{
		void* d_ptr,*h_ptr; //device ���������ݿ���ָ�����ڽ�GPU���ݿ���������
		cudaMallocAsync(&d_ptr, output_size[i], stream); //��������ڵ��ڴ�
		cudaHostAlloc(&h_ptr, output_size[i], 0);//���������ڴ湩GPUֱ�ӵ���
		d_ptrs.push_back(d_ptr);
		h_ptrs.push_back(h_ptr);

		auto name = output_names[i].c_str();
		context->setTensorAddress(name, d_ptr);
	}
	//----------------------------------------------------------------------------

	//��ѡ����(Ԥ��)---------------------------------------------------------------
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < input_names.size(); j++)
		{
			int size = input_size[j];
			void* h_ptr=(void*)malloc(size); //���������ڴ�
			memset(h_ptr, 0, size); //��ʼ���ڴ�
            cudaMemcpyAsync(d_ptrs[j], h_ptr, size, cudaMemcpyHostToDevice, stream); //���������ݿ�����GPU
			free (h_ptr);
		}
		cudaStreamSynchronize(stream); //ͬ������

		//ִ������
		context->enqueueV3(stream); //TRT10�ӿ� TRT.8�ӿ�Ϊ context->enqueueV2();
		for (int j = 0; j < output_names.size(); j++) 
		{
			int size = output_size[j];
            cudaMemcpyAsync(h_ptrs[j], d_ptrs[j+input_names.size()], size, cudaMemcpyDeviceToHost, stream); //��GPU���ݿ���������
		}
		cudaStreamSynchronize(stream);
		cout<<"Warmup "<<i+1<<" times......"<<endl;
	}
	//----------------------------------------------------------------------------

    //���岽ִ������---------------------------------------------------------------
	string imagePath="D:\\Desktop\\Data\\shizhi\\AI\\background";//����ͼƬ·���ļ���
	vector<string> imageFiles;
	cv::glob(imagePath, imageFiles);
	vector<cv::Mat> images;
	for (int i = 0; i < imageFiles.size(); i++)
	{
		images.push_back(cv::imread(imageFiles[i]));
	}

	cv::Size inputSize= cv::Size(640, 640); //����ģ��ͼƬ�ߴ�
	for (int i = 0; i < images.size(); i++)
	{
		//����ͼƬǰ����
		cv::Mat inferImg = images[i].clone();
		int width= inferImg.cols;
		int height = inferImg.rows;
		int inp_h= inputSize.height;
		int inp_w= inputSize.width;

		float ratio= std::min((float)inp_h/height,(float)inp_w/width); //����ͼ�����ű�������
		int pad_h= (int)round(height*ratio); //�������ź�߿�
		int pad_w= (int)round(width*ratio);

		cv::Mat tmp;
		if ((int)width != pad_w || (int)height != pad_h) {
			cv::resize(inferImg, tmp, cv::Size(pad_w, pad_h)); //resizeֻ�ᰴ�̶�����Ƚ��з���
		}
		else {
			tmp = inferImg.clone();
		}

		float dw = inp_w - pad_w;
		float dh = inp_h - pad_h; //�������߿�

		dw /= 2.0f; //��Χʽ���
		dh /= 2.0f;
		int top = int(std::round(dh - 0.1f));
		int bottom = int(std::round(dh + 0.1f));
		int left = int(std::round(dw - 0.1f));
		int right = int(std::round(dw + 0.1f));

		cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, { 114, 114, 114 }); //yoloϵ�����114

		cv::Mat out;
		out.create({ 1, 3, (int)inp_h, (int)inp_w }, CV_32F);

		std::vector<cv::Mat> channels;
		cv::split(tmp, channels);

		cv::Mat c0((int)inp_h, (int)inp_w, CV_32F, (float*)out.data);
		cv::Mat c1((int)inp_h, (int)inp_w, CV_32F, (float*)out.data + (int)inp_h * (int)inp_w);
		cv::Mat c2((int)inp_h, (int)inp_w, CV_32F, (float*)out.data + (int)inp_h * (int)inp_w * 2);


		//BGRͨ��תΪBGR
		channels[0].convertTo(c2, CV_32F, 1 / 255.f);
		channels[1].convertTo(c1, CV_32F, 1 / 255.f); 
		channels[2].convertTo(c0, CV_32F, 1 / 255.f);

		//ǰ�������
		//�����ڴ�
		for (int i = 0; i < input_names.size(); i++)
		{
			cudaMemcpyAsync(d_ptrs[i], out.ptr<float>(), out.total() * out.elemSize(), cudaMemcpyHostToDevice, stream);
			auto name= input_names[i].c_str();
			context->setInputShape(name, { 1, 3, (int)inp_h, (int)inp_w });
			context->setTensorAddress(name, d_ptrs[i]);
		}
		cudaStreamSynchronize(stream);

		//��ʼ����
		context->enqueueV3(stream); //TRT10�ӿ� TRT.8�ӿ�Ϊ context->enqueueV2();
		for (int j = 0; j < output_names.size(); j++)
		{
			int size = output_size[j];
			cudaMemcpyAsync(h_ptrs[j], d_ptrs[j + input_names.size()], size, cudaMemcpyDeviceToHost, stream); //��GPU���ݿ���������
		}
		cudaStreamSynchronize(stream);
		//----------------------------------------------------------------------------

		//���к��������Լ�ģ�͵������-----------------------------------------------
		//cerr << "Model output nums:" << h_ptrs.size() << endl;
		//for (int i = 0; i < h_ptrs.size(); i++)
		//{
		//	//printf("Number:{%d} output name is %s\n", i, output_names[i].c_str());
		//	//cerr<<h_ptrs[i]<<endl;

		//	auto *data = (float*)h_ptrs[i];
		//}
		//----------------------------------------------------------------------------

		int* num_dets = static_cast<int*>(h_ptrs[0]); //��������
		auto* boxes = static_cast<float*>(h_ptrs[1]); //��������--float ��ָ��
		auto* scores = static_cast<float*>(h_ptrs[2]);//����÷�
		int* labels = static_cast<int*>(h_ptrs[3]);//�������

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
	string imagePath = "D:\\Desktop\\Data\\Fangdaqi\\test";//����ͼƬ·���ļ���
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
