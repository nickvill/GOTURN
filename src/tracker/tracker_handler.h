// Adapted from tracker_manager.h

#ifndef TRACKER_HANDLER_H
#define TRACKER_HANDLER_H

#include "network/regressor.h"
#include "tracker.h"
#include "helper/high_res_timer.h"
#include "helper/bounding_box.h"

namespace tracker_handler {

class TrackerHandler {
public:
  TrackerHandler(const std::string& deploy_proto, 
                 const std::string& caffe_model, 
                 const int gpu_id);
  void InitNetwork(model_file, trained_file, gpu_id);
  void SetupTracker();
  void RecoverDetection(cv::Mat& image_prev, 
                        std::vector<float>& prev_detection,
                        cv::Mat& image_curr,
                        std::vector<float>& new_detection);
  void DetectionToBoundingBox(std::vector<float>& prev_detection,
                              BoundingBox& bbox_prev);
  void BoundingBoxToDetection(std::vector<float>& new_detection,
                              BoundingBox& bbox_estimate_uncentered);

protected:
  // Neural network tracker.
  RegressorBase* regressor_;

  // Tracker.
  Tracker* tracker_;
};

}

#endif // TRACKER_HANDLER_H