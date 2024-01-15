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
            ROS_INFO("Initiating Node");

            std::string node_name = ros::this_node::getName();
            std::string voc_file, settings_file;
            nodeHandler->param<std::string>(node_name + "/voc_file", voc_file, "file_not_set");
            nodeHandler->param<std::string>(node_name + "/settings_file", settings_file, "file_not_set");

            if (voc_file == "file_not_set" || settings_file == "file_not_set")
            {
                ROS_ERROR("Please provide voc_file and settings_file in the launch file");       
                ros::shutdown();    
            }

            nodeHandler->param<std::string>(node_name + "/world_frame_id", world_frame_id, "map");
            nodeHandler->param<std::string>(node_name + "/cam_frame_id", cam_frame_id, "camera");

            bool enable_pangolin;
            nodeHandler->param<bool>(node_name + "/enable_pangolin", enable_pangolin, true);

            yoloHandler = new Yolo("/home/benyamin/catkin_ws/src/orb_slam3_ros/yolov8n-seg.onnx");

            // Create SLAM system. It initializes all system threads and gets ready to process frames.
            sensor_type = ORB_SLAM3::System::MONOCULAR;
            pSLAM = new ORB_SLAM3::System(voc_file, settings_file, sensor_type, enable_pangolin);

            imgSub = nodeHandler->subscribe("/camera/image_raw", 1, &YoloNode::ImgCallback, this);

            setup_publishers_seg(*nodeHandler, *itHandler, node_name);
            setup_services(*nodeHandler, node_name);
            resPub = itHandler->advertise("/seg_image",1);
        }

        float avgProc = 0;
        int count = 0;
    
    private:
        ros::Subscriber imgSub;
        image_transport::Publisher resPub;

        void ImgCallback(const sensor_msgs::ImageConstPtr& msg){
            count++;
            auto start = std::chrono::high_resolution_clock::now();
            cv_bridge::CvImageConstPtr cv_ptr;
            try{
                cv_ptr = cv_bridge::toCvShare(msg);
            }
            catch(cv_bridge::Exception e){
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }
            // std::cout<<"processing"<<std::endl;
            std::vector<YoloResult> result = yoloHandler->ProcessImage(cv_ptr->image);
            std::cout<<result.size()<<std::endl;
            cv::Mat mask = yoloHandler->DrawImage(result, cv_ptr->image);
            // std::cout<<"mask created"<<std::endl;
            sensor_msgs::ImagePtr resImgMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", mask).toImageMsg();
            resPub.publish(resImgMsg);

            Sophus::SE3f Tcw = pSLAM->TrackMonocular(cv_ptr->image, cv_ptr->header.stamp.toSec());
            pSLAM->Cobain2(mask);
            // std::cout<<"tracking"<<std::endl;
            ros::Time msg_time = msg->header.stamp;
            publish_topics_seg(msg_time, mask);
            // std::cout<<"publishing"<<std::endl;
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            avgProc = avgProc+((duration.count()-avgProc)/count);
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
    std::cout<<"Average processing time : "<<node.avgProc/1000<<"ms"<<std::endl;
    pSLAM->Shutdown();
    pSLAM->SaveTrajectoryEuRoC("CameraTrajectory.txt");
    pSLAM->SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    ros::shutdown();

    return 0;
}
