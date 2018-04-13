#ifndef TRACKER_HANDLER_H
#define TRACKER_HANDLER_H

#include "network/regressor.h"
#include "tracker/tracker.h"
#include "helper/high_res_timer.h"

class TrackerHandler {
public:
  TrackerHandler(RegressorBase* regressor, Tracker* tracker);

  void RecoverDetection();
}