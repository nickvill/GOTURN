// Adapted from tracker_manager.cpp

#include "tracker_handler.h"

#include <string>

#include "helper/helper.h"
#include "train/tracker_trainer.h"

namespace tracker_handler {

TrackerHandler::TrackerHandler(const std::string& deploy_proto, 
                               const std::string& caffe_model, 
                               const int gpu_id) {
  InitNetwork(deploy_proto, caffe_model, gpu_id);
  SetupTracker();
}

void TrackerHandler::InitNetwork(const std::string& deploy_proto, 
                                 const std::string& caffe_model, 
                                 const int gpu_id) {
  // Initializes the network
  // ::google::InitGoogleLogging("1");
  Regressor regressor(deploy_proto, caffe_model, gpu_id, false);
  regressor_ = &regressor;
  regressor_->Init();
}

void TrackerHandler::SetupTracker() {
  Tracker tracker(false);
  tracker_ = &tracker;
}

void TrackerHandler::RecoverDetection(cv::Mat& image_prev, 
                                      std::vector<float>& prev_detection,
                                      cv::Mat& image_curr,
                                      std::vector<float>& new_detection) {
  // Turn detection into a bounding box
  BoundingBox bbox_prev;
  DetectionToBoundingBox(prev_detection, bbox_prev);
  std::cout << "Converted to BBox!" << std::endl;

  // Set the previous image and bbox
  tracker_->Init(image_prev, bbox_prev, regressor_);
  std::cout << "Initialized the tracker!" << std::endl;

  //Predict bbox location
  BoundingBox bbox_estimate_uncentered;
  tracker_->Track(image_curr, regressor_, &bbox_estimate_uncentered);
  std::cout << "Got track estimate!" << std::endl;
  BoundingBoxToDetection(new_detection, bbox_estimate_uncentered);
  std::cout << "Converted BBox to Detection!" << std::endl;
}

void TrackerHandler::DetectionToBoundingBox(std::vector<float>& prev_detection,
                                            BoundingBox& bbox_prev) {
  bbox_prev.x1_ = prev_detection[3];
  bbox_prev.y1_ = prev_detection[4];
  bbox_prev.x2_ = prev_detection[5];
  bbox_prev.y2_ = prev_detection[6];
}

void TrackerHandler::BoundingBoxToDetection(std::vector<float>& new_detection,
                                            BoundingBox& bbox_estimate_uncentered) {
  if (new_detection.size() != 7) {
    new_detection = {0,0,0,0,0,0,0};
  }

  new_detection[3] = bbox_estimate_uncentered.x1_;
  new_detection[4] = bbox_estimate_uncentered.y1_;
  new_detection[5] = bbox_estimate_uncentered.x2_;
  new_detection[6] = bbox_estimate_uncentered.y2_;
}

} // namespace tracker_handler
