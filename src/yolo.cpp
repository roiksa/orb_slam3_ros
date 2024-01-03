#include "yolo.h"

const int& DEFAULT_LETTERBOX_PAD_VALUE = 144;

void letterbox(const cv::Mat& image,
    cv::Mat& outImage,
    const cv::Size& newShape,
    cv::Scalar_<double> color,
    bool auto_,
    bool scaleFill,
    bool scaleUp, int stride
) {
    cv::Size shape = image.size();
    float r = std::min(static_cast<float>(newShape.height) / static_cast<float>(shape.height),
        static_cast<float>(newShape.width) / static_cast<float>(shape.width));
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2]{ r, r };
    int newUnpad[2]{ static_cast<int>(std::round(static_cast<float>(shape.width) * r)),
                     static_cast<int>(std::round(static_cast<float>(shape.height) * r)) };

    auto dw = static_cast<float>(newShape.width - newUnpad[0]);
    auto dh = static_cast<float>(newShape.height - newUnpad[1]);

    if (auto_)
    {
        dw = static_cast<float>((static_cast<int>(dw) % stride));
        dh = static_cast<float>((static_cast<int>(dh) % stride));
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        newUnpad[0] = newShape.width;
        newUnpad[1] = newShape.height;
        ratio[0] = static_cast<float>(newShape.width) / static_cast<float>(shape.width);
        ratio[1] = static_cast<float>(newShape.height) / static_cast<float>(shape.height);
    }

    dw /= 2.0f;
    dh /= 2.0f;

    //cv::Mat outImage;
    if (shape.width != newUnpad[0] || shape.height != newUnpad[1])
    {
        cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
    }
    else
    {
        outImage = image.clone();
    }

    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));


    if (color == cv::Scalar()) {
        color = cv::Scalar(DEFAULT_LETTERBOX_PAD_VALUE, DEFAULT_LETTERBOX_PAD_VALUE, DEFAULT_LETTERBOX_PAD_VALUE);
    }

    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);

}

Yolo::Yolo(const std::string& modelPath){

    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    std::vector<std::string> available_providers = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(available_providers.begin(),available_providers.end(), "CUDAExecutionProvider");
    if(cudaAvailable == available_providers.end()){
        std::cout << "CUDA not available on device, proceeding with CPU" << std::endl;
    }
    else{
        std::cout << "Cuda available on device, proceeding with GPU" << std::endl;
        OrtCUDAProviderOptions cudaOption;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);    
    }
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    _session = new Ort::Session(_env, modelPath.c_str(), sessionOptions);
    Ort::AllocatorWithDefaultOptions allocator;

    _inputNodesNum = _session->GetInputCount();
    _inputName = std::move(_session->GetInputNameAllocated(0,allocator));
    _inputNodeNames.push_back(_inputName.get());
    Ort::TypeInfo inputTypeInfo = _session->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    _inputNodeDataType = inputTensorInfo.GetElementType();
    _inputTensorShape = inputTensorInfo.GetShape();
    if (_inputTensorShape[0] == -1){
		_isDynamicShape = true;
		_inputTensorShape[0] = _batchSize;
    }
	if (_inputTensorShape[2] == -1 || _inputTensorShape[3] == -1) {
		_isDynamicShape = true;
		_inputTensorShape[2] = _netHeight;
		_inputTensorShape[3] = _netWidth;
	}
    _outputNodesNum = _session->GetOutputCount();
    if (_outputNodesNum != 2){
        std::cout<< "Model output is not 2, not a segmentation model" << std::endl;
        return;
    }
    _outputName0 = std::move(_session->GetOutputNameAllocated(0,allocator));
    _outputName1 = std::move(_session->GetOutputNameAllocated(1,allocator));
    Ort::TypeInfo typeInfoOutput0(nullptr);
	Ort::TypeInfo typeInfoOutput1(nullptr);
	bool flag = false;

	flag = strcmp(_outputName0.get(), _outputName1.get()) < 0;
	if (flag)  //make sure "output0" is in front of  "output1"
	{
		typeInfoOutput0 = _session->GetOutputTypeInfo(0);  //output0
		typeInfoOutput1 = _session->GetOutputTypeInfo(1);  //output1
		_outputNodeNames.push_back(_outputName0.get());
		_outputNodeNames.push_back(_outputName1.get());
	}
	else {
		typeInfoOutput0 = _session->GetOutputTypeInfo(1);  //output0
		typeInfoOutput1 = _session->GetOutputTypeInfo(0);  //output1
		_outputNodeNames.push_back(_outputName1.get());
		_outputNodeNames.push_back(_outputName0.get());
	}
    auto tensorInfoOutput0 = typeInfoOutput0.GetTensorTypeAndShapeInfo();
    _outputNodeDataType = tensorInfoOutput0.GetElementType();
    _outputTensorShape = tensorInfoOutput0.GetShape();
    auto tensorInfoOutput1 = typeInfoOutput1.GetTensorTypeAndShapeInfo();

    std::cout << "Model Loaded, warming up..."<< std::endl;
    size_t inputTensorLength = VectorProduct(_inputTensorShape);
    float* temp = new float[inputTensorLength];
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;
    Ort::MemoryInfo _OrtMemoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPUOutput));
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        _OrtMemoryInfo, temp, inputTensorLength, _inputTensorShape.data(),
        _inputTensorShape.size()));
    for (int i = 0; i < 3; ++i) {
        outputTensors = _session->Run(Ort::RunOptions{ nullptr },
            _inputNodeNames.data(),
            inputTensors.data(),
            _inputNodeNames.size(),
            _outputNodeNames.data(),
            _outputNodeNames.size());
    }
    delete[]temp;

    std::srand(time(0));
    for (int i = 0; i < 80; i++) {
        int b = std::rand() % 256;
        int g = std::rand() % 256;
        int r = std::rand() % 256;
        _color.push_back(cv::Scalar(b, g, r));
    }
    std::cout<<"Model Ready!"<<std::endl;
}

std::vector<YoloResult> Yolo::ProcessImage(const cv::Mat img){
    // std::cout<<"Img received"<<std::endl;
    cv::Mat newImage;
    const bool& auto_=false;
    const bool& scaleFill=false;
    letterbox(img, newImage,_netShape, cv::Scalar(), auto_, scaleFill, true, 32);
    cv::Mat blob = cv::dnn::blobFromImage(newImage, 1/255.0, _netShape, cv::Scalar(0,0,0), true, false);
    // std::cout<<"Infering"<<std::endl;
    int64_t inputTensorLength = VectorProduct(_inputTensorShape);
	std::vector<Ort::Value> inputTensors;
	std::vector<Ort::Value> outputTensors;
    Ort::MemoryInfo _OrtMemoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPUOutput));
    inputTensors.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo, (float*)blob.data, inputTensorLength, _inputTensorShape.data(), _inputTensorShape.size()));
    outputTensors = _session->Run(Ort::RunOptions{ nullptr },
		_inputNodeNames.data(),
		inputTensors.data(),
		_inputNodeNames.size(),
		_outputNodeNames.data(),
		_outputNodeNames.size()
	);
    // std::cout<<"Processing result"<<std::endl;
    int classNamesNum = _className.size();
    std::vector<int64_t> output0TensorShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int64_t> output1TensorShape = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();
    float* all_data = outputTensors[0].GetTensorMutableData<float>();
    cv::Mat output0 = cv::Mat(cv::Size((int)output0TensorShape[2], (int)output0TensorShape[1]), CV_32F, all_data).t();
    std::vector<int> maskSize = { 1,(int)output1TensorShape[1],(int)output1TensorShape[2],(int)output1TensorShape[3] };
    cv::Mat output1 = cv::Mat(maskSize, CV_32F, outputTensors[1].GetTensorMutableData<float>());
    std::cout<<output1.cols<<" "<<output1.rows<<std::endl;
    std::cout<<output0.size()<<std::endl;
    std::cout<<output1.size()<<std::endl;
    std::vector<YoloResult> output;
    // if(output1.rows<=0||output1.cols<=0){
    //     return output;
    // }
    int maskFeaturesNum = output1TensorShape[1];
    int maskW = output1TensorShape[2];
    int maskH = output1TensorShape[3];
    
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> masks;

    int dataWidth = classNamesNum + 4 + maskFeaturesNum;
    int rows = output0.rows;    
    float* pdata = (float*)output0.data;
   
    // std::cout<<"PostProcess "<<rows<<std::endl;
    for(int r = 0; r<rows; r++){
        cv::Mat scores(1, classNamesNum, CV_32FC1, pdata+4);
        cv::Point classId;
        double maxConf;
        minMaxLoc(scores, 0, &maxConf, 0, &classId);
        if(maxConf >= _classTreshold){
            if(classId.x == 0){
                masks.push_back(std::vector<float>(pdata+4+classNamesNum, pdata+dataWidth));
                classIds.push_back(classId.x);
                confidences.push_back(maxConf);
                float w = pdata[2];
                float h = pdata[3];
                float left = MAX(int(pdata[0]-0.5*w+0.5),0);
                float top = MAX(int(pdata[1]-0.5*h+0.5),0);
                cv::Rect_<float> bbox = cv::Rect(left, top, (w+0.5), (h+0.5));
                cv::Rect_<float> scaledBbox = scale_boxes(cv::Size(_netWidth, _netHeight), bbox, cv::Size(img.cols, img.rows));
                boxes.push_back(scaledBbox);
            }
        }
        pdata += dataWidth;
    }

    // std::cout<<"NMS"<<std::endl;
    std::vector<int> nmsResult;
    cv::dnn::NMSBoxes(boxes, confidences, _classTreshold, _nmsTreshold, nmsResult);
    cv::Size downsampledSize = cv::Size(maskW, maskH);
    // std::cout<<maskW<<" "<<maskH<<std::endl;
    // std::cout<<output1.cols<<" "<<output1.rows<<std::endl;
    std::vector<cv::Range> roiRangs = { cv::Range(0, 1), cv::Range::all(),
                                         cv::Range(0, downsampledSize.height), cv::Range(0, downsampledSize.width) };
    cv::Mat tempMask = output1(roiRangs).clone();
    // std::cout<<"2"<<std::endl;
    cv::Mat proto = tempMask.reshape(0, { maskFeaturesNum, downsampledSize.width * downsampledSize.height });
    // std::cout<<"3"<<std::endl;
    for (int i = 0; i < nmsResult.size(); ++i)
    {
        // std::cout<<i<<std::endl;
        int idx = nmsResult[i];
        boxes[idx] = boxes[idx] & cv::Rect(0, 0, img.cols, img.rows);
        YoloResult result = { classIds[idx] ,confidences[idx] ,boxes[idx] };
        GetMask(cv::Mat(masks[idx]).t(), proto, img.cols, img.rows, boxes[idx], result.mask,
            _netWidth, _netHeight, maskW, maskH, maskFeaturesNum);
        // std::cout<<i<<"1"<<std::endl;
        output.push_back(result);
    }
    // std::cout<<"done"<<std::endl;
    return output;
}

void Yolo::GetMask(const cv::Mat &masks_features, const cv::Mat& proto, int imWidth, const int imHeight, const cv::Rect bound, cv::Mat& mask_out, int iw, int ih, int mw, int mh, int& masks_features_num)
{
    cv::Size img0_shape = cv::Size(imWidth, imHeight);
    cv::Size img1_shape = cv::Size(iw, ih);
    cv::Size downsampled_size = cv::Size(mw, mh);

    cv::Rect_<float> bound_float(
        static_cast<float>(bound.x),
        static_cast<float>(bound.y),
        static_cast<float>(bound.width),
        static_cast<float>(bound.height)
    );

    cv::Rect_<float> downsampled_bbox = scale_boxes(img0_shape, bound_float, downsampled_size);
    cv::Size bound_size = cv::Size(mw, mh);
    clip_boxes(downsampled_bbox, bound_size);

    cv::Mat matmul_res = (masks_features * proto).t();
    matmul_res = matmul_res.reshape(1, { downsampled_size.height, downsampled_size.width });
    // apply sigmoid to the mask:
    cv::Mat sigmoid_mask;
    cv::exp(-matmul_res, sigmoid_mask);
    sigmoid_mask = 1.0 / (1.0 + sigmoid_mask);
    cv::Mat resized_mask;
    cv::Rect_<float> input_bbox = scale_boxes(img0_shape, bound_float, img1_shape);
    cv::resize(sigmoid_mask, resized_mask, img1_shape, 0, 0, cv::INTER_LANCZOS4);
    cv::Mat pre_out_mask = resized_mask(input_bbox);
    cv::Mat scaled_mask;
    scale_image(scaled_mask, resized_mask, img0_shape);
    cv::resize(scaled_mask, mask_out, img0_shape);
    mask_out = mask_out(bound) > _maskTreshold;
}

cv::Mat Yolo::DrawImage(std::vector<YoloResult> results, const cv::Mat img){
    cv::Mat mask = img.clone()*0.0;

    // int radius = 5;
    // bool drawLines = true;

    // auto raw_image_shape = img.size();
    for (const auto& res : results) {
        // float left = res.bbox.x;
        // float top = res.bbox.y;
        // int color_num = res.classIds;

        // Draw bounding box
        // cv::rectangle(img, res.bbox, _color[res.classIds], 2);

        // Draw mask if available
        if (res.mask.rows && res.mask.cols > 0) {
            mask(res.bbox).setTo(cv::Scalar(255.0,255.0,255.0), res.mask);
        }

        // Create label
        // std::stringstream labelStream;
        // labelStream << _className[res.classIds] << " " << std::fixed << std::setprecision(2) << res.conf;
        // std::string label = labelStream.str();

        // cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
        // cv::Rect rect_to_fill(left - 1, top - text_size.height - 5, text_size.width + 2, text_size.height + 5);
        // cv::Scalar text_color = cv::Scalar(255.0, 255.0, 255.0);
        // cv::rectangle(img, rect_to_fill, _color[res.classIds], -1);
        // cv::putText(img, label, cv::Point(left - 1.5, top - 2.5), cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);

        // Check if keypoints are available
    }

    // Combine the image and mask
    // cv::addWeighted(img, 0.6, mask, 0.4, 0, img);

    cvtColor(mask,mask,cv::COLOR_BGR2GRAY);
    return mask;
}

cv::Rect_<float> Yolo::scale_boxes(const cv::Size& img1_shape, cv::Rect_<float>& box, const cv::Size& img0_shape) {
    std::pair<float, cv::Point2f> ratio_pad = std::make_pair(-1.0f, cv::Point2f(-1.0f, -1.0f));
    bool padding = true;

    float gain, pad_x, pad_y;

    if (ratio_pad.first < 0.0f) {
        gain = std::min(static_cast<float>(img1_shape.height) / static_cast<float>(img0_shape.height),
            static_cast<float>(img1_shape.width) / static_cast<float>(img0_shape.width));
        pad_x = roundf((img1_shape.width - img0_shape.width * gain) / 2.0f - 0.1f);
        pad_y = roundf((img1_shape.height - img0_shape.height * gain) / 2.0f - 0.1f);
    }
    else {
        gain = ratio_pad.first;
        pad_x = ratio_pad.second.x;
        pad_y = ratio_pad.second.y;
    }

    //cv::Rect scaledCoords(box);
    cv::Rect_<float> scaledCoords(box);

    if (padding) {
        scaledCoords.x -= pad_x;
        scaledCoords.y -= pad_y;
    }

    scaledCoords.x /= gain;
    scaledCoords.y /= gain;
    scaledCoords.width /= gain;
    scaledCoords.height /= gain;

    // Clip the box to the bounds of the image
    clip_boxes(scaledCoords, img0_shape);

    return scaledCoords;
}

void Yolo::clip_boxes(cv::Rect_<float>& box, const cv::Size& shape) {
    box.x = std::max(0.0f, std::min(box.x, static_cast<float>(shape.width)));
    box.y = std::max(0.0f, std::min(box.y, static_cast<float>(shape.height)));
    box.width = std::max(0.0f, std::min(box.width, static_cast<float>(shape.width - box.x)));
    box.height = std::max(0.0f, std::min(box.height, static_cast<float>(shape.height - box.y)));
}

void Yolo::scale_image(cv::Mat& scaled_mask, const cv::Mat& resized_mask, const cv::Size& im0_shape) {
    std::pair<float, cv::Point2f> ratio_pad = std::make_pair(-1.0f, cv::Point2f(-1.0f, -1.0f));
    cv::Size im1_shape = resized_mask.size();

    // Check if resizing is needed
    if (im1_shape == im0_shape) {
        scaled_mask = resized_mask.clone();
        return;
    }

    float gain, pad_x, pad_y;

    if (ratio_pad.first < 0.0f) {
        gain = std::min(static_cast<float>(im1_shape.height) / static_cast<float>(im0_shape.height),
                        static_cast<float>(im1_shape.width) / static_cast<float>(im0_shape.width));
        pad_x = (im1_shape.width - im0_shape.width * gain) / 2.0f;
        pad_y = (im1_shape.height - im0_shape.height * gain) / 2.0f;
    }
    else {
        gain = ratio_pad.first;
        pad_x = ratio_pad.second.x;
        pad_y = ratio_pad.second.y;
    }

    int top = static_cast<int>(pad_y);
    int left = static_cast<int>(pad_x);
    int bottom = static_cast<int>(im1_shape.height - pad_y);
    int right = static_cast<int>(im1_shape.width - pad_x);

    // Clip and resize the mask
    cv::Rect clipped_rect(left, top, right - left, bottom - top);
    cv::Mat clipped_mask = resized_mask(clipped_rect);
    cv::resize(clipped_mask, scaled_mask, im0_shape);
}