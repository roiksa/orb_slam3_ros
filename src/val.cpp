#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <boost/filesystem.hpp>
#include "yolo.h"


int main(int argc, char **argv){
    Yolo yoloHandler = Yolo("/home/benyamin/YoloCpp/yolov8-ros/src/yolov8seg-ros/yolov8n-seg.onnx");
    std::ofstream jsonFile;
    jsonFile.open("test.json");
    // std::string path = "/home/benyamin/COCO/coco/val2017/*.jpg";
    std::vector<cv::String> fn;
    cv::glob("/home/benyamin/COCO/coco/val2017/*.jpg", fn, true);

    size_t count = fn.size(); //number of png files in images folder
    for (size_t i=0; i<5; i++){
        cv::Mat img = cv::imread(fn[i]);
        boost::filesystem::path fName(fn[i]);
        std::vector<YoloResult> result = yoloHandler.ProcessImage(img);
        cv::imwrite((fName.stem().string()+".jpg"),yoloHandler.DrawImage(result,img));
        yoloHandler.Save(result, img, jsonFile,fName.stem().string());
    }
    yoloHandler.ShowTime();
    jsonFile.close();
}

