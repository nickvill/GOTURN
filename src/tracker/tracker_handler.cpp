#include "tracker_handler.h"

#include <string>

#include "helper/helper.h"
#include "train/tracker_trainer.h"


TrackerHandler::TrackerHandler(RegressorBase* regressor, Tracker* tracker) :
  regressor_(regressor),
  tracker_(tracker)
{
}

void TrackerHandler::InitTracker() {
	tracker_->Init()
}

void TrackerHandler::RecoverDetection() {
	//
}
