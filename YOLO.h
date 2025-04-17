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
//Ĭ��TRT_10�汾
//# define TRT_8 //���䲻ͬ�汾TRT
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
        void doInference();                                         //ִ������
        void preProcess(const cv::Mat& srcImg,
                cv::Mat &outImg,cv::Size size);                     //Ԥ����
        void bindingData(const cv::Mat& inferImg);                  //���������ݵ�GPU
        void postProcess(vector<Box>& boxes);                       //����
        void drawBoxs(cv::Mat& srcImg,vector<Box>& boxes,
            const vector<string> classNames);                       //����������

        //���ýӿ�
        void WarmUp(int times = 10);                                //Ԥ��
        /*
        * @function:���
        * @param defectImg :�����ͼƬ
        * @param res:�����
        */
        void Detect(const cv::Mat &defectImg, vector<Box> &detRes); //���
public:
    int m_BindingsNum;                                              //��������ڵ�����   
    vector<string> m_InputName;                                     //����ڵ�����
    vector<string> m_OutputName;                                    //����ڵ�����
    vector<void*> m_hostPtrs;                                       //��������ָ��
    vector<void*> m_devicePtrs;                                     //�豸����ָ��
    vector<Box> m_inferRes;                                         //������
private:
    preParam param;                                                 //Ԥ����������ں���ԭ

private:
    nvinfer1::ICudaEngine* m_Engine=nullptr;
    nvinfer1::IRuntime* m_Runtime= nullptr;
    nvinfer1::IExecutionContext* m_Context= nullptr;
    cudaStream_t m_Stream= nullptr;

    Logger m_Logger{nvinfer1::ILogger::Severity::kERROR};

    vector<Binding> m_InputBindings;                                //����ڵ���Ϣ
    vector<Binding> m_OutputBindings;                               //����ڵ���Ϣ
    vector<cv::Size> m_InputSize;                                   //ģ������ͼƬ�ߴ�
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
    void doInference(const cv::Mat &inferImg);                            //ִ������
    void postProcess(cv::Mat &postImg);                                   //����
    void warmUp(int times = 10);                                          //Ԥ�ȡ�����������ʹ��GPU����ORTʱ�������Ԥ�ȣ�
    void preProcess(const cv::Mat& srcImg,cv::Mat &outImg,cv::Size size); //Ԥ����
    bool isWarmUp = false;                                                //�Զ�Ԥ��ѡ��
private:
    void drawBoxs(cv::Mat& srcImg,vector<Box>& boxes);                    //����������
    bool matToBlob(const cv::Mat &inferImg, float* &pdata
        ,Ort::Value &outTensor);                                          //��ͼƬ����ת��Ϊģ�������ʽ
    bool matToBlob(const cv::Mat &inferImg, vector<float>& pData,Ort::Value &outTensor);        //��ͼƬ����ת��Ϊģ�������ʽ
public:
    vector<Box> m_res;                                                    //������
    vector<string> m_classMaps = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };                                                                    //�������
private:
    Ort::Env m_env;                                                       //Ort���� 
    Ort::Session* m_session=nullptr;                                      //Ort�Ự
    vector<Ort::Value> m_outputTensor;                                    //�����м���
    vector<vector<int64_t>> m_inputDims;                                  //ģ������ά��
    vector<vector<int64_t>> m_outputDims;                                 //ģ������ά��
    vector<const char*> m_inputNames;                                     //����ڵ�����
    vector<const char*> m_outputNames;                                    //����ڵ�����
    int m_inputHeight;                                                    //����ͼƬ�߶�
    int m_inputWidth;                                                     //����ͼƬ���
};

