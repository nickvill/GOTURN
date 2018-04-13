#ifndef TRACKER_HANDLER_H
#define TRACKER_HANDLER_H

#include "network/regressor.h"
#include "tracker/tracker.h"
#include "helper/high_res_timer.h"
#include "helper/bounding_box.h"

namespace tracker_handler {

class TrackerHandler {
public:
  TrackerHandler(RegressorBase* regressor, Tracker* tracker);
  void InitNetwork();
  void RecoverDetection(cv::Mat& image_prev, 
                        std::vector<float>& prev_detection,
                        cv::Mat& image_curr,
                        std::vector<float>& new_detection);
  void DetectionToBoundingBox(std::vector<float>& prev_detection,
                              BoundingBox& bbox_prev);
  void BoundingBoxToDetection(std::vector<float>& new_detection,
                              BoundingBox& bbox_estimate_uncentered);
};

}