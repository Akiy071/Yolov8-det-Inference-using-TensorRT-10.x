#pragma once
#include "NvInfer.h"
#include<map>
#include<string>
#include<iostream>
#include "opencv2/opencv.hpp"

using namespace std;
//CUDA 检查函数
#define CHECK(call)                                                     \
do {                                                                    \
    const cudaError_t error_code = call;                                \
    if (error_code != cudaSuccess)                                      \
    {                                                                   \
        printf("CUDA ERROR: \n");                                       \
        printf("    FILE: %s\n", __FILE__);                             \
        printf("    LINE: %d\n", __LINE__);                             \
        printf("    ERROR CODE: %d\n", error_code);                     \
        printf("    ERROR TEXT: %s\n", cudaGetErrorString(error_code)); \
        exit(1);                                                        \
    }                                                                   \
}while(0); 

//输出当前时间用于调试
#define GET_CURRENT_TIME(str) do { \
	struct tm t; \
    std::time_t now = std::time(nullptr); \
    localtime_s(&t, &now); \
	cout << "【" << t.tm_year+1900<< "-" << t.tm_mon+1 << "-" << t.tm_mday << " " << t.tm_hour << ":" << t.tm_min << ":" << t.tm_sec << "】"; \
	cout << str << endl; \
} while(0)

// TRT Logger继承类 具体可看 https://blog.csdn.net/weixin_43605641/article/details/136183188

class Logger : public nvinfer1::ILogger {
public:
    nvinfer1::ILogger::Severity reportableSeverity;

    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO) :
        reportableSeverity(severity)
    {
    }

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        if (severity > reportableSeverity) {
            return;
        }
        switch (severity) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case nvinfer1::ILogger::Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "VERBOSE: ";
            break;
        }
        std::cerr << msg << std::endl;
    }
};

//约束范围防止越界
inline static float clamp(float val, float min, float max)
{
    return val > min ? (val < max ? val : max) : min;
}


namespace det
{
    struct Binding
    {
        size_t size = 1;        //size of the data number.
        size_t dSize = 1;       //size of the data memory= size*sizeof(type(data))
        nvinfer1::Dims dims;    // dims of the binding data.
        std::string name;       //name of the binding.
    };

    struct Box
    {
        cv::Rect rect;          //bounding box
        int label = 0;          //class label
        float prob = 0.0;       //confidence score
    };

    struct preParam
    {
        float ratio = 1.0;       //resize ratio
        float pad_w = 0.0;       //padding width
        float pad_h = 0.0;       //padding height
        float height = 0.0;       //src height
        float width = 0.0;        //src width
    };

} //namespace det

//根据类型判断对应字节大小
inline int type_to_size(const nvinfer1::DataType& dataType)
{
    switch (dataType) {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kINT8:
        return 1;
    case nvinfer1::DataType::kBOOL:
        return 1;
    default:
        return 4;
    }
}

//计算所有维度大小 3*640*640=921600 float type dsize=921600*4=3686400 byte
inline int get_size_by_dims(const nvinfer1::Dims& dims)
{
    int size = 1;
    for (int i = 0; i < dims.nbDims; i++) {
        size *= dims.d[i];
    }
    return size;
}

struct COLORMAP
{
    std::unordered_map<std::string, cv::Scalar> colormap;

    //根据标签创建对应颜色
    void addColor(string name, cv::Scalar color)
    {
        colormap[name] = color;
    }

    void autoRandomColor(vector<string> labels)
    {
        for (int i = 0; i < labels.size(); i++)
        {
            cv::Scalar color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
            addColor(to_string(i), color);
        }
    }

    cv::Scalar getColor(string label) const {
        auto it = colormap.find(label);
        if (it != colormap.end()) {
            return it->second;
        }
        else {
            throw std::runtime_error("Label not found");
        }
    }
};
