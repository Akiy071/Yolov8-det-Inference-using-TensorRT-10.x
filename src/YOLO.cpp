#include "YOLO.h"

//���캯��
YOLOv8::YOLOv8(const string& engineFilePath)
{
    /*
    * ��һ������engine�ļ����Դ�����
    *   1.Logger��������־��¼
    *   2.Context��������ִ��������
    *   3.Runtime��������������
    *   4.Stream��������cuda��
    *   5.Engine�����������л�������������
    *   6.Input&OutPut����������������ڵ���Ϣ
    */

    //1.����engine�ļ�
    ifstream file(engineFilePath, std::ios::binary);
    if (!file.good())
        GET_CURRENT_TIME("Can't find engine file in " + engineFilePath);
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    if (trtModelStream == nullptr)
        GET_CURRENT_TIME(" ");
    file.read(trtModelStream, size);
    file.close();

    //2.����engine�ļ����г�ʼ��
    //Logger->Runtime->Engine->Context->Stream
    initLibNvInferPlugins(&this->m_Logger, "");
    this->m_Runtime = nvinfer1::createInferRuntime(this->m_Logger);
    if (m_Runtime == nullptr)
        GET_CURRENT_TIME(" ");

    this->m_Engine = this->m_Runtime->deserializeCudaEngine(trtModelStream, size);
    if (m_Engine == nullptr)
        GET_CURRENT_TIME(" ");
    delete[] trtModelStream; //�ͷ��ڴ�

    this->m_Context = this->m_Engine->createExecutionContext();
    if (m_Context == nullptr)
        GET_CURRENT_TIME(" ");

    cudaStreamCreate(&this->m_Stream);

    //3.��ȡ��������ڵ���Ϣ
//һЩ�ӿ���TRT 10�汾�Ѿ������� ����ɿ���
// https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration-guide.html#removed-safety-c-apis
#ifdef TRT_8
    this->m_BindingsNum = this->m_Engine->getNbBindings(); //getNbBindings() API which was abandoned in TRT_8.5
    for (int i = 0; i < this->m_BindingsNum; ++i) {
        Binding        binding;
        nvinfer1::Dims dims;

        nvinfer1::DataType dtype = this->m_Engine->getBindingDataType(i);
        std::string  name = this->m_Engine->getBindingName(i);

        binding.name = name;
        binding.dSize = type_to_size(dtype);

        bool IsInput = m_Engine->bindingIsInput(i);
        if (IsInput) {
            dims = this->m_Engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            // set max opt shape
            this->m_Context->setBindingDimensions(i, dims);

            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            m_InputName.push_back(name);
            m_InputBindings.push_back(binding);
        }
        else
        {
            dims = this->m_Context->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            m_InputName.push_back(name);
            m_InputBindings.push_back(binding);
        }
    }
#else
    this->m_BindingsNum = this->m_Engine->getNbIOTensors();
    for (int i = 0; i < m_BindingsNum; i++) {
        Binding binding;

        string name = this->m_Engine->getIOTensorName(i);
        nvinfer1::DataType dtype = this->m_Engine->getTensorDataType(name.c_str());

        binding.dSize = type_to_size(dtype);
        binding.name = name;

        //�жϽڵ�����
        bool isInput = this->m_Engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
        nvinfer1::Dims dims = this->m_Engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
        if (isInput)
        {
            cv::Size modelInputSize = cv::Size(dims.d[2], dims.d[3]);
            m_InputSize.push_back(modelInputSize);

            this->m_Context->setInputShape(name.c_str(), dims);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;

            cout << "input name: " << name << " dims: " << dims.nbDims
                << " input shape:(" << dims.d[0] << "," << dims.d[1] << ","
                << dims.d[2] << "," << dims.d[3] << ")" << endl;

            m_InputName.push_back(name);
            m_InputBindings.push_back(binding);
        }
        //����ڵ�
        else
        {
            dims = this->m_Context->getTensorShape(name.c_str());

            cout << "ouput name: " << name << " dims: " << dims.nbDims
                << " ouput shape:(" << dims.d[0] << "," << dims.d[1] << ","
                << dims.d[2] << "," << dims.d[3] << ")" << endl;

            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            m_OutputName.push_back(name);
            m_OutputBindings.push_back(binding);
        }
    }

    //���������
    for (int i = 0; i < m_InputName.size(); i++)
    {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, m_InputBindings[i].size * m_InputBindings[i].dSize, this->m_Stream));
        m_devicePtrs.push_back(d_ptr);

#ifndef TRT_8
        string name = m_InputName[i];
        this->m_Context->setInputShape(name.c_str(), m_InputBindings[i].dims);
        this->m_Context->setTensorAddress(name.c_str(), d_ptr);
#endif // !TRT_8
    }

    for (int i = 0; i < m_OutputName.size(); i++)
    {
        void* d_ptr, * h_ptr;
        size_t size = m_OutputBindings[i].size * m_OutputBindings[i].dSize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->m_Stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        m_devicePtrs.push_back(d_ptr);
        m_hostPtrs.push_back(h_ptr);

#ifndef TRT_8
        string name = m_OutputName[i];
        this->m_Context->setTensorAddress(name.c_str(), d_ptr);
#endif // !TRT_8
    }

#endif //TRT_8
    GET_CURRENT_TIME("AI model init success!");
}

 void YOLOv8::WarmUp(int times)
{
    for (int i = 0; i < times; i++)
    {
        auto start = std::chrono::system_clock::now();
        for (auto binding : this->m_InputBindings)
        {
            size_t size = binding.size * binding.dSize;
            void* h_ptr = malloc(size);
            memset(h_ptr, 0, size);
            //���������ڴ浽�豸
            CHECK(cudaMemcpyAsync(this->m_devicePtrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->m_Stream));
            free(h_ptr);
        }

        //ִ������
        this->doInference();

        auto end = std::chrono::system_clock::now();
        auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;

        printf("Number %d warmUp time: %.3f ms\n", i, tc);
    }
}

 void YOLOv8::doInference()
{
    int inputNums = m_InputName.size();
    this->m_Context->enqueueV3(this->m_Stream); //TRT_8 using API enqueueV2
    for (int i = 0; i < m_OutputBindings.size(); i++)
    {
        size_t size = m_OutputBindings[i].size * m_OutputBindings[i].dSize;
        CHECK(cudaMemcpyAsync(m_hostPtrs[i], m_devicePtrs[i + inputNums], size,
            cudaMemcpyDeviceToHost, m_Stream)); //cuda���ݿ���������
    }
    cudaStreamSynchronize(this->m_Stream);
}

 void YOLOv8::preProcess(const cv::Mat& srcImg, cv::Mat& outImg, cv::Size size)
{
    const float height = srcImg.rows;
    const float width = srcImg.cols;
    const float input_h = size.height;  //ָ��������С
    const float input_w = size.width;   //ָ��������С

    float ratio = min(input_h / height, input_w / width);  //Ŀ�����ű���
    int pad_w = round(width * ratio);
    int pad_h = round(height * ratio);

    //���Ŀ�����
    cv::Mat tmpImg;
    if ((int)width != (int)pad_w || (int)height != (int)pad_h)
    {
        cv::resize(srcImg, tmpImg, cv::Size(pad_w, pad_h));
    }
    else
    {
        tmpImg = srcImg.clone();
    }

    //�����������Ҫ���ĳߴ�
    pad_h = input_h - pad_h;
    pad_w = input_w - pad_w;

    pad_w /= 2.0f;
    pad_h /= 2.0f;
    int top = int(std::round(pad_h - 0.1f));
    int bottom = int(std::round(pad_h + 0.1f));
    int left = int(std::round(pad_w - 0.1f));
    int right = int(std::round(pad_w + 0.1f));

    cv::copyMakeBorder(tmpImg, tmpImg, top, bottom, left, right, cv::BORDER_CONSTANT, { 114, 114, 114 });

    outImg.create({ 1, 3, (int)input_h, (int)input_w }, CV_32F); //cv shape���ģ�������shape

    std::vector<cv::Mat> channels(3, cv::Mat::zeros(tmpImg.size(), tmpImg.type()));
    cv::split(tmpImg, channels);

    cv::Mat c0((int)input_h, (int)input_w, CV_32F, (float*)outImg.data);
    cv::Mat c1((int)input_h, (int)input_w, CV_32F, (float*)outImg.data + (int)input_h * (int)input_w);
    cv::Mat c2((int)input_h, (int)input_w, CV_32F, (float*)outImg.data + (int)input_h * (int)input_w * 2);

    //��һ��
    channels[0].convertTo(c2, CV_32F, 1 / 255.f);
    channels[1].convertTo(c1, CV_32F, 1 / 255.f);
    channels[2].convertTo(c0, CV_32F, 1 / 255.f);

    //Ԥ�����������
    param.ratio = 1 / ratio;
    param.pad_h = pad_h;
    param.pad_w = pad_w;
    param.height = height;
    param.width = width;
}

 void YOLOv8::bindingData(const cv::Mat& inferImg)
{
    /*
    * ��Ԥ���������ݿ������豸�ڴ��У��˺�����Ҫ������ǰ���е��ã����򽫻ᱨ��
    */
    for (int i = 0; i < m_InputSize.size(); i++)
    {
        cv::Mat predImg;
        cv::Size size = m_InputSize[i];
        preProcess(inferImg, predImg, size); //����ָ��size����Ԥ����

        auto mallocSize = predImg.total() * predImg.elemSize();
        CHECK(cudaMemcpyAsync(m_devicePtrs[i], predImg.ptr<float>(),
            mallocSize, cudaMemcpyHostToDevice, m_Stream));
#ifndef TRT_8
        auto name = m_InputName[i].c_str();
        m_Context->setInputShape(name, nvinfer1::Dims{ 4,{1,3,size.height,size.width } });
        m_Context->setTensorAddress(name, m_devicePtrs[i]);
#else
        m_Context->setBindingDimensions(0, nvinfer1::Dims{ 4, {1, 3, size.height, size.width} });
#endif //TRT_8
    }
}

void YOLOv8::postProcess(vector<Box>& objs)
{
    /*
    * �������ڴ�ָ������ȡ�����������������к���
    */
    objs.clear(); //ȷ��ÿ�����
    int* num_dets = static_cast<int*>(m_hostPtrs[0]);
    auto* boxes = static_cast<float*>(m_hostPtrs[1]);
    auto* scores = static_cast<float*>(m_hostPtrs[2]);
    int* labels = static_cast<int*>(m_hostPtrs[3]);

    //��ԭ����
    auto& dw = this->param.pad_w;
    auto& dh = this->param.pad_h;
    auto& width = this->param.width;
    auto& height = this->param.height;
    auto& ratio = this->param.ratio;

    cerr<<"*num_dets: "<<*num_dets<<endl;
    for (int i = 0; i < num_dets[0]; i++) {
        float* ptr = boxes + i * 4;

        //�����Χ��ȡֵ
        float x0 = *ptr++ - dw;
        float y0 = *ptr++ - dh;
        float x1 = *ptr++ - dw;
        float y1 = *ptr - dh;

        //��ֹԤ���Խ��
        x0 = clamp(x0 * ratio, 0.f, width);
        y0 = clamp(y0 * ratio, 0.f, height);
        x1 = clamp(x1 * ratio, 0.f, width);
        y1 = clamp(y1 * ratio, 0.f, height);

        //��x,y,x1,y1��->(x,y,w,h)
        Box obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.prob = *(scores + i);
        obj.label = *(labels + i);

        objs.push_back(obj);
    }
}

void YOLOv8::drawBoxs(cv::Mat& srcImg, vector<Box>& boxes, const vector<string> classNames)
{
    for (int i = 0; i < boxes.size(); i++)
    {
        cv::Scalar color = cv::Scalar(0, 0, 255);
        cv::rectangle(srcImg, boxes[i].rect, color, 2);

        char text[256];
        sprintf_s(text, "%s %.1f%%", classNames[boxes[i].label].c_str(), boxes[i].prob * 100);

        int      baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        //��ȡ�ı��ĳߴ�ͻ���λ�ã��Ա���ȷ�����ı���
        int x = (int)boxes[i].rect.x;
        int y = (int)boxes[i].rect.y + 1;

        if (y > srcImg.rows) {
            y = srcImg.rows;
        }

        cv::rectangle(srcImg, cv::Rect(x, y, label_size.width, label_size.height + baseLine), { 0, 0, 255 }, -1);
        cv::putText(srcImg, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, { 255, 255, 255 }, 1);
    }
}

void YOLOv8::Detect(const cv::Mat& defectImg,vector<Box> &detRes)
{
    /*
    * �����й������ӳ�һ����������
    *       1.Ԥ������Ԥ���������ݿ������豸�ڴ���
    *       2.����
    *       3.�����������豸�ڴ�ȡ��
    *       4.����
    *       5.����ѡ������������
    */
    this->bindingData(defectImg);
    this->doInference();
    this->postProcess(detRes);
}

//�������������ͷ���Դ
YOLOv8::~YOLOv8()
{
    /*
    * �ͷ���Դ��һ������Ҫ�ͷŵ�ָ���У�
    *   1.Context
    *   2.Engine
    *   3.Runtime
    *   4.Stream
    *   5.CUDA Device&Host
    */

#ifdef TRT_8
    this->m_Context->destroy();
    this->m_Engine->destroy();
    this->m_Runtime->destroy();
#else
    delete this->m_Context;
    delete this->m_Engine;
    delete this->m_Runtime;
#endif //TRT_8

    cudaStreamDestroy(this->m_Stream);
    for (auto ptr : m_devicePtrs) //�ͷ��豸ָ��
        CHECK(cudaFree(ptr));
    for (auto ptr : m_hostPtrs) //�ͷ�����ָ��
        CHECK(cudaFreeHost(ptr));
}
