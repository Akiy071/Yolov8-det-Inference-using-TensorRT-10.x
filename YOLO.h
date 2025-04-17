#pragma once

#include<windows.h>
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


#ifndef  YOLO_ORT_EXPORT_H
#define YOLO_ORT_EXPORT_H _declspec(dllexport)
#else
#define YOLO_ORT_EXPORT_H _declspec(dllimport)
#endif
#include<onnxruntime_cxx_api.h>
#include<onnxruntime_cxx_api.h>
class YOLO_ORT_EXPORT_H YOLOv8_ORT
{
public:
    explicit
    YOLOv8_ORT(const string modelPath,bool cudaEnable=true);
    ~YOLOv8_ORT() {
        delete m_session;
    }

public:
    void doInference(const cv::Mat &inferImg);                            //执行推理
    void postProcess(cv::Mat &postImg);                                   //后处理
    void warmUp(int times = 10);                                          //预热——————使用GPU进行ORT时必须进行预热！
    void preProcess(const cv::Mat& srcImg,cv::Mat &outImg,cv::Size size); //预处理
    bool isWarmUp = false;                                                //自动预热选项
private:
    void drawBoxs(cv::Mat& srcImg,vector<Box>& boxes);                    //绘制推理结果
    bool matToBlob(const cv::Mat &inferImg, float* &pdata
        ,Ort::Value &outTensor);                                          //将图片数据转换为模型输入格式
    bool matToBlob(const cv::Mat &inferImg, vector<float>& pData,Ort::Value &outTensor);        //将图片数据转换为模型输入格式
public:
    vector<Box> m_res;                                                    //推理结果
    vector<string> m_classMaps = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };                                                                    //类别名称
private:
    Ort::Env m_env;                                                       //Ort环境 
    Ort::Session* m_session=nullptr;                                      //Ort会话
    vector<Ort::Value> m_outputTensor;                                    //推理中间结果
    vector<vector<int64_t>> m_inputDims;                                  //模型输入维度
    vector<vector<int64_t>> m_outputDims;                                 //模型输入维度
    vector<const char*> m_inputNames;                                     //输入节点名称
    vector<const char*> m_outputNames;                                    //输出节点名称
    int m_inputHeight;                                                    //输入图片高度
    int m_inputWidth;                                                     //输入图片宽度
};

