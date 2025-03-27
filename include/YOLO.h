#pragma once
#include"common.h"
#include"NvInferPlugin.h"
#include <fstream>
#include<iostream>
#include<chrono>
#include <NvInfer.h>
#include<string>
#include<vector>

using namespace std;
using namespace det;
//默认TRT_10版本
//# define TRT_8 //适配不同版本TRT
#ifdef  YOLOEXPORT_H
#define YOLOEXPORT_H _declspec(dllexport)
#else
#define YOLOEXPORT_H _declspec(dllimport)
#endif

class YOLOEXPORT_H YOLOv8
{
    public:
        YOLOv8() {};
        YOLOv8(const string &engineFilePath);
        ~YOLOv8();

    public:
        void doInference();                                         //执行推理
        void preProcess(const cv::Mat& srcImg,
                cv::Mat &outImg,cv::Size size);                     //预处理
        void bindingData(const cv::Mat& inferImg);                  //绑定输入数据到GPU
        void postProcess(vector<Box>& boxes);                       //后处理
        void drawBoxs(cv::Mat& srcImg,vector<Box>& boxes,
            const vector<string> classNames);                       //绘制推理结果

        //调用接口
        void WarmUp(int times = 10);                                //预热
        /*
        * @function:检测
        * @param defectImg :待检测图片
        * @param res:检测结果
        */
        void Detect(const cv::Mat &defectImg, vector<Box> &detRes); //检测
public:
    int m_BindingsNum;                                              //输入输出节点数量   
    vector<string> m_InputName;                                     //输入节点名称
    vector<string> m_OutputName;                                    //输出节点名称
    vector<void*> m_hostPtrs;                                       //主机数据指针
    vector<void*> m_devicePtrs;                                     //设备数据指针
    vector<Box> m_inferRes;                                         //推理结果
private:
    preParam param;                                                 //预处理参数用于后处理还原

private:
    nvinfer1::ICudaEngine* m_Engine=nullptr;
    nvinfer1::IRuntime* m_Runtime= nullptr;
    nvinfer1::IExecutionContext* m_Context= nullptr;
    cudaStream_t m_Stream= nullptr;

    Logger m_Logger{nvinfer1::ILogger::Severity::kERROR};

    vector<Binding> m_InputBindings;                                //输入节点信息
    vector<Binding> m_OutputBindings;                               //输出节点信息
    vector<cv::Size> m_InputSize;                                   //模型输入图片尺寸
};

