#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <numeric>
#include <time.h>
#include <onnxruntime_cxx_api.h>

struct YoloResult{
    int classIds{};
    float conf{};
    cv::Rect_<float> bbox;
    cv::Mat mask; 
};

void letterbox(const cv::Mat& image,
    cv::Mat& outImage,
    const cv::Size& newShape,
    cv::Scalar_<double> color,
    bool auto_,
    bool scaleFill,
    bool scaleUp, int stride
);

class Yolo{
    public: 
        Yolo(const std::string& modelPath);
        std::vector<YoloResult> ProcessImage(const cv::Mat img);
        cv::Mat DrawImage(std::vector<YoloResult> result, const cv::Mat img);

        std::vector<std::string> _className = {
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush"
	    };

    private:

        template <typename T>
        T VectorProduct(const std::vector<T>& v)
        {
            return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
        };

        void GetMask(const cv::Mat &masks_features, const cv::Mat& proto, int imWidth, const int imHeight, const cv::Rect bound, cv::Mat& mask_out, int iw, int ih, int mw, int mh, int& masks_features_num);
        cv::Rect_<float> scale_boxes(const cv::Size& img1_shape, cv::Rect_<float>& box, const cv::Size& img0_shape); 
        void clip_boxes(cv::Rect_<float>& box, const cv::Size& shape);
        void scale_image(cv::Mat& scaled_mask, const cv::Mat& resized_mask, const cv::Size& im0_shape);
        
        Ort::Session* _session = nullptr;
        Ort::Env _env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Yolov8-Seg");   

        size_t _inputNodesNum = 0;
        size_t _outputNodesNum = 0;
        std::shared_ptr<char> _inputName, _outputName0,_outputName1;
        std::vector<char*> _inputNodeNames;
        std::vector<char*> _outputNodeNames;
        ONNXTensorElementDataType _inputNodeDataType;
        ONNXTensorElementDataType _outputNodeDataType;
        std::vector<int64_t> _inputTensorShape;
        std::vector<int64_t> _outputTensorShape;
        std::vector<int64_t> _outputMaskTensorShape;
        int _batchSize = 1;
        const int _netWidth = 640;
        const int _netHeight = 640;
        const cv::Size _netShape = cv::Size(_netWidth, _netHeight);
        bool _isDynamicShape = false;
        float _classTreshold = 0.25;
        float _nmsTreshold = 0.45;
        float _maskTreshold = 0.5;

        std::vector<cv::Scalar> _color;
	    

};
