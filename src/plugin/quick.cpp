/*
  _   _ ____  _____ ___        _      _
 | | | |  _ \| ____/ _ \ _   _(_) ___| | __
 | |_| | |_) |  _|| | | | | | | |/ __| |/ /
 |  _  |  __/| |__| |_| | |_| | | (__|   <
 |_| |_|_|   |_____\__\_\\__,_|_|\___|_|\_\

Quick demo of human pose estimation using OpenVINO
Author: Paolo Bosetti
*/

#include "../source.hpp"
#include <chrono>

#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>

#include <models/hpe_model_openpose.h>
#include <models/input_data.h>
#include <nlohmann/json.hpp>
#include <pipelines/async_pipeline.h>
#include <pipelines/metadata.h>
#include <utils/common.hpp>
#include <pugg/Kernel.h>

#ifndef PLUGIN_NAME
#define PLUGIN_NAME "hpequick"
#endif

using namespace std;
using json = nlohmann::json;

// Map of OpenPOSE keypoint names
// TODO: update with Miroscic names
map<int, string> keypoints_map = {
    {0, "Nose"},   {1, "Neck"},      {2, "RShoulder"}, {3, "RElbow"},
    {4, "RWrist"}, {5, "LShoulder"}, {6, "LElbow"},    {7, "LWrist"},
    {8, "RHip"},   {9, "RKnee"},     {10, "RAnkle"},   {11, "LHip"},
    {12, "LKnee"}, {13, "LAnkle"},   {14, "REye"},     {15, "LEye"},
    {16, "REar"},  {17, "LEar"}};


class HPEQuick : public Source<json> {

  /*
    ____  _        _   _
   / ___|| |_ __ _| |_(_) ___
   \___ \| __/ _` | __| |/ __|
    ___) | || (_| | |_| | (__
   |____/ \__\__,_|\__|_|\___|

  */

  static cv::Mat renderHumanPose(HumanPoseResult &result,
                                 OutputTransform &outputTransform) {
    if (!result.metaData) {
      throw std::invalid_argument("Renderer: metadata is null");
    }

    auto output_img = result.metaData->asRef<ImageMetaData>().img;

    if (output_img.empty()) {
      throw std::invalid_argument(
          "Renderer: image provided in metadata is empty");
    }
    outputTransform.resize(output_img);
    static const cv::Scalar colors[HPEOpenPose::keypointsNumber] = {
        cv::Scalar(255, 0, 0),   cv::Scalar(255, 85, 0),
        cv::Scalar(255, 170, 0), cv::Scalar(255, 255, 0),
        cv::Scalar(170, 255, 0), cv::Scalar(85, 255, 0),
        cv::Scalar(0, 255, 0),   cv::Scalar(0, 255, 85),
        cv::Scalar(0, 255, 170), cv::Scalar(0, 255, 255),
        cv::Scalar(0, 170, 255), cv::Scalar(0, 85, 255),
        cv::Scalar(0, 0, 255),   cv::Scalar(85, 0, 255),
        cv::Scalar(170, 0, 255), cv::Scalar(255, 0, 255),
        cv::Scalar(255, 0, 170), cv::Scalar(255, 0, 85)};
    static const std::pair<int, int> keypointsOP[] = {
        {1, 2}, {1, 5},  {2, 3},   {3, 4},  {5, 6},   {6, 7},
        {1, 8}, {8, 9},  {9, 10},  {1, 11}, {11, 12}, {12, 13},
        {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}};
    static const std::pair<int, int> keypointsAE[] = {
        {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
        {5, 6},   {5, 7},   {6, 8},   {7, 9},   {8, 10},  {1, 2},  {0, 1},
        {0, 2},   {1, 3},   {2, 4},   {3, 5},   {4, 6}};
    const int stick_width = 4;
    const cv::Point2f absent_keypoint(-1.0f, -1.0f);
    for (auto &pose : result.poses) {
      for (size_t keypoint_idx = 0; keypoint_idx < pose.keypoints.size();
           keypoint_idx++) {
        if (pose.keypoints[keypoint_idx] != absent_keypoint) {
          outputTransform.scaleCoord(pose.keypoints[keypoint_idx]);
          cv::circle(output_img, pose.keypoints[keypoint_idx], 4,
                     colors[keypoint_idx], -1);
        }
      }
    }
    std::vector<std::pair<int, int>> limb_keypoints_ids;
    if (!result.poses.empty()) {
      if (result.poses[0].keypoints.size() == HPEOpenPose::keypointsNumber) {
        limb_keypoints_ids.insert(limb_keypoints_ids.begin(),
                                  std::begin(keypointsOP),
                                  std::end(keypointsOP));
      } else {
        limb_keypoints_ids.insert(limb_keypoints_ids.begin(),
                                  std::begin(keypointsAE),
                                  std::end(keypointsAE));
      }
    }
    cv::Mat pane = output_img.clone();
    for (auto pose : result.poses) {
      for (const auto &limb_keypoints_id : limb_keypoints_ids) {
        std::pair<cv::Point2f, cv::Point2f> limb_keypoints(
            pose.keypoints[limb_keypoints_id.first],
            pose.keypoints[limb_keypoints_id.second]);
        if (limb_keypoints.first == absent_keypoint ||
            limb_keypoints.second == absent_keypoint) {
          continue;
        }

        float mean_x = (limb_keypoints.first.x + limb_keypoints.second.x) / 2;
        float mean_y = (limb_keypoints.first.y + limb_keypoints.second.y) / 2;
        cv::Point difference = limb_keypoints.first - limb_keypoints.second;
        double length = std::sqrt(difference.x * difference.x +
                                  difference.y * difference.y);
        int angle = static_cast<int>(std::atan2(difference.y, difference.x) *
                                     180 / CV_PI);
        std::vector<cv::Point> polygon;
        cv::ellipse2Poly(cv::Point2d(mean_x, mean_y),
                         cv::Size2d(length / 2, stick_width), angle, 0, 360, 1,
                         polygon);
        cv::fillConvexPoly(pane, polygon, colors[limb_keypoints_id.second]);
      }
    }
    cv::addWeighted(output_img, 0.4, pane, 0.6, 0, output_img);
    return output_img;
  }

  /*
     ____ _                                          _
    / ___| | __ _ ___ ___   _ __ ___   ___ _ __ ___ | |__   ___ _ __ ___
   | |   | |/ _` / __/ __| | '_ ` _ \ / _ \ '_ ` _ \| '_ \ / _ \ '__/ __|
   | |___| | (_| \__ \__ \ | | | | | |  __/ | | | | | |_) |  __/ |  \__ \
    \____|_|\__,_|___/___/ |_| |_| |_|\___|_| |_| |_|_.__/ \___|_|  |___/

  */

public:
  // Constructor
  HPEQuick() : _agent_id(PLUGIN_NAME) {}

  void set_params(void *params) override {
    json j = *(json *)params;

    if (j.contains("device")) {
      _device = j["device"];
    }

    if (j.contains("model_file")) {
      _model_file = j["model_file"];
    } else {
      throw std::invalid_argument("Missing model_file parameter");
    }

    if (j.contains("out_res")) {
      out_res = j["out_res"];
    }

    if (j.contains("agent_id")) {
      _agent_id = j["agent_id"];
    }

    _start_time = chrono::steady_clock::now();
    // setup video capture
    _cap.open(_device);
    if (!_cap.isOpened()) {
      throw std::invalid_argument("Cannot open the video camera");
    }
    _cap >> _curr_frame;
    cv::Size resolution = _curr_frame.size();
    size_t found = out_res.find("x");
    if (found != string::npos) {
      resolution = cv::Size{stoi(out_res.substr(0, found)),
                            stoi(out_res.substr(found + 1, out_res.length()))};
      _output_transform = OutputTransform(_curr_frame.size(), resolution);
      resolution = _output_transform.computeResolution();
    }
    cout << "Resolution: " << resolution << endl;

    // setup inference model
    double aspect_ratio =
        _curr_frame.cols / static_cast<double>(_curr_frame.rows);
    _model.reset(new HPEOpenPose(_model_file, aspect_ratio, tsize,
                                 static_cast<float>(threshold), layout));
    // setup pipeline
    _pipeline = new AsyncPipeline(
        std::move(_model),
        ConfigFactory::getUserConfig(d, nireq, nstreams, nthreads), _core);
    frame_num = _pipeline->submitData(
        ImageInputData(_curr_frame),
        std::make_shared<ImageMetaData>(_curr_frame, _start_time));
  }

  // Destructor
  ~HPEQuick() {
    _cap.release();
    delete _pipeline;
  }

  string kind() override { return PLUGIN_NAME; }

  // Process a single frame
  return_type get_output(json *out, std::vector<unsigned char> *blob = nullptr) override {

    out->clear();
    (*out)["agent_id"] = _agent_id;

    // Submit data to the pipeline
    if (_pipeline->isReadyToProcess()) {
      // Capturing frame
      _start_time = std::chrono::steady_clock::now();
      _cap >> _curr_frame;
      if (_curr_frame.empty()) {
        // Input stream is over
        return return_type::error;
      }
      frame_num = _pipeline->submitData(
          ImageInputData(_curr_frame),
          std::make_shared<ImageMetaData>(_curr_frame, _start_time));
    }

    // Waiting for free input slot or output data available. Function will
    // return immediately if any of them are available.
    _pipeline->waitForData();

    // Deal with results
    if (!(result = _pipeline->getResult())) {
      return return_type::warning;
    }
    if (!no_show) {
      _out_frame =
          renderHumanPose(result->asRef<HumanPoseResult>(), _output_transform);
      //cv::imshow("Human Pose Estimation Results", _out_frame);
      cv::imshow("Human Pose Estimation Results", result->asRef<HumanPoseResult>().heatMaps[0]);
      int key = cv::waitKey(1000.0 / fps);
      if (27 == key || 'q' == key || 'Q' == key) { // Esc
        return return_type::error;
      }
    }
    frames_processed++;

    // Prepare output
    std::vector<HumanPose> poses = result->asRef<HumanPoseResult>().poses;
    for (int i = 0; i < poses.size(); i++) {
      for (int kp = 0; kp < poses[i].keypoints.size(); kp++) {
        auto &keypoint = poses[i].keypoints[kp];
        if (keypoint.x < 0 || keypoint.y < 0)
          continue;
        (*out)["poses"][i][keypoints_map[kp]] = {keypoint.x, keypoint.y};
      }
    }

    return return_type::success;
  }

  void save_last_frame(string filename) { cv::imwrite(filename, _out_frame); }

  map<string, string> info() override {
    return {
      {"device", to_string(_device)},
      {"model_file", _model_file},
      {"out_res", out_res},
      {"kind", kind()},
      {"agent_id", _agent_id}
    };
  };


  uint32_t tsize = 0;         // target size
  double threshold = 0.1;     // probability threshold
  string layout = "";         // inputs layouts (NCHW, NHWC)
  string d = "CPU";           // computation device
  uint32_t nireq = 0;         // number of infer requests
  string nstreams = "";       // number of streams
  uint32_t nthreads = 0;      // number of CPU threads
  bool no_show = false;       // Don't show
  string out_res = "800x600"; // output resolution
  uint32_t frames_processed = 0;
  int64_t frame_num = 0;
  double fps = 25;
  unique_ptr<ResultBase> result;

private:
  int _device = 0;
  string _model_file;
  string _agent_id;
  cv::VideoCapture _cap;
  chrono::steady_clock::time_point _start_time;
  cv::Mat _curr_frame, _out_frame;
  OutputTransform _output_transform;
  unique_ptr<ModelBase> _model;
  ov::Core _core;
  AsyncPipeline *_pipeline;
};


/*
  ____  _             _             _      _
 |  _ \| |_   _  __ _(_)_ __     __| |_ __(_)_   _____ _ __
 | |_) | | | | |/ _` | | '_ \   / _` | '__| \ \ / / _ \ '__|
 |  __/| | |_| | (_| | | | | | | (_| | |  | |\ V /  __/ |
 |_|   |_|\__,_|\__, |_|_| |_|  \__,_|_|  |_| \_/ \___|_|
                |___/
This is the plugin driver, it should not need to be modified
*/
INSTALL_SOURCE_DRIVER(HPEQuick, json)




/*
  __  __       _
 |  \/  | __ _(_)_ __
 | |\/| |/ _` | | '_ \
 | |  | | (_| | | | | |
 |_|  |_|\__,_|_|_| |_|

*/

int main(int argc, char *argv[]) {
  return_type rt = return_type::success;
  try {
    int cam_id = 0;
    if (argc < 2) {
      throw std::invalid_argument("Usage: " + string(argv[0]) +
                                  " <model.xml> [cam_id]");
    }
    if (argc == 3) {
      cam_id = stoi(argv[2]);
    }
    // HPEQuick hpe(cam_id, argv[1]);
    HPEQuick hpe;
    json params = {
      {"device", cam_id}, 
      {"model_file", argv[1]},
      {"out_res", "800x450"}
    };
    hpe.set_params(&params);
    json output = {};

    while ((rt = hpe.get_output(&output)) != return_type::error) {
      if (rt == return_type::warning) {
        cout <<endl << "*** Warning: no result." << endl;
      } else {
        cout << "Output: " << output.dump() << endl;
      }
    }
    cout << endl;
  } catch (const std::exception &error) {
    cerr << error.what() << endl;
    return 1;
  }

  return 0;
}
