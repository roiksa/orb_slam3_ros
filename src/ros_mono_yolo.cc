/**
* 
* Adapted from ORB-SLAM3: Examples/ROS/src/ros_mono.cc
*
*/

#include "common.h"
#include "yolo.h"

using namespace std;

Yolo* yoloHandler;

class YoloNode{
    public:
        YoloNode(ros::NodeHandle *nodeHandler, image_transport::ImageTransport *itHandler){
            std::string node_name = ros::this_node::getName();
            std::string voc_file, settings_file;
            node_handler.param<std::string>(node_name + "/voc_file", voc_file, "file_not_set");
            node_handler.param<std::string>(node_name + "/settings_file", settings_file, "file_not_set");

            if (voc_file == "file_not_set" || settings_file == "file_not_set")
            {
                ROS_ERROR("Please provide voc_file and settings_file in the launch file");       
                ros::shutdown();
                return 1;
            }

            node_handler.param<std::string>(node_name + "/world_frame_id", world_frame_id, "map");
            node_handler.param<std::string>(node_name + "/cam_frame_id", cam_frame_id, "camera");

            bool enable_pangolin;
            node_handler.param<bool>(node_name + "/enable_pangolin", enable_pangolin, true);

            yoloHandler = new Yolo("/home/benyamin/YoloCpp/yolov8-ros/src/yolov8seg-ros/yolov8m-seg.onnx");

            // Create SLAM system. It initializes all system threads and gets ready to process frames.
            sensor_type = ORB_SLAM3::System::MONOCULAR;
            pSLAM = new ORB_SLAM3::System(voc_file, settings_file, sensor_type, enable_pangolin);

            imgSub = nodeHandler.subscribe("/camera/image_raw", 1, &YoloNode::ImgCallback, this);

            setup_publishers(node_handler, image_transport, node_name);
            setup_services(node_handler, node_name);
            resPub = itHandler->advertise("/seg_image",1);
        }
    
    private:
        ros::Subscriber imgSub;
        image_transport::Publisher resPub;

        void ImgCallback(const sensor_msgs::ImageConstPtr& msg){
            try{
                cv::Mat img = cv_bridge::toCvShare(msg,"bgr8")->image;
            }catch(cv_bridge::Exception e){
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }

            std::vector<YoloResult> result = yoloHandler->ProcessImage(img);
            cv::Mat resImg = yoloHandler->DrawImage(result, img);
            sensor_msgs::ImagePtr resImgMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", resImg).toImageMsg();
            resPub.publish(resImgMsg);

            Sophus::SE3f Tcw = pSLAM->TrackMonocular(cv_ptr->image, cv_ptr->header.stamp.toSec());

            ros::Time msg_time = msg->header.stamp;

            publish_topics(msg_time);
        }
};



int main(int argc, char **argv){
    ros::init(argc,argv, "monoYolo");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    if (argc > 1)
    {
        ROS_WARN ("Arguments supplied via command line are ignored.");
    }
    
    ros::NodeHandle nodeHandler;
    image_transport::ImageTransport itHandler(nodeHandler);    
    YoloNode node = YoloNode(&nodeHandler, &itHandler);

    ros::spin();
    pSLAM->Shutdown();
    ros::shutdown();

    return 0;
}
