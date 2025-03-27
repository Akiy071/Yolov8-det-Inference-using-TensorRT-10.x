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

