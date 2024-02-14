/*
   ____ _            _            _             _
  / ___| | ___   ___| | __  _ __ | |_   _  __ _(_)_ __
 | |   | |/ _ \ / __| |/ / | '_ \| | | | |/ _` | | '_ \
 | |___| | (_) | (__|   <  | |_) | | |_| | (_| | | | | |
  \____|_|\___/ \___|_|\_\ | .__/|_|\__,_|\__, |_|_| |_|
                           |_|            |___/
Produces current time and date
*/

#include "../source.hpp"
#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <pugg/Kernel.h>
#include <sstream>
#include <thread>

#ifndef PLUGIN_NAME
#define PLUGIN_NAME "clock"
#endif

using namespace std;
using namespace cv;
using namespace std::chrono;
using json = nlohmann::json;

// Plugin class. This shall be the only part that needs to be modified,
// implementing the actual functionality
class Webcam : public Source<json> {
public:
  Webcam() {
    _error = "none";
    _blob_format = "jpg";
  }

  ~Webcam() {
    _cap.release();
    destroyAllWindows();
  }

  string kind() override { return PLUGIN_NAME; }

  static string get_ISO8601(
      const system_clock::time_point &time = chrono::system_clock::now()) {
    time_t tt = system_clock::to_time_t(time);
    tm *tt2 = localtime(&tt);

    // Get milliseconds hack
    auto timeTruncated = system_clock::from_time_t(tt);
    int ms =
        std::chrono::duration_cast<milliseconds>(time - timeTruncated).count();

    return (
               stringstream()
               << put_time(tt2, "%FT%T")               // "2023-03-30T19:49:53"
               << "." << setw(3) << setfill('0') << ms // ".005"
               << put_time(tt2, "%z") // "+0200" (time zone offset, optional)
               )
        .str();
  }

  return_type get_output(json *out,
                         std::vector<unsigned char> *blob = nullptr) override {
    return_type result = return_type::success;
    Mat hist;
    int hist_size = 20;
    if (_params.contains("hist_size")) {
      hist_size = _params["hist_size"];
    }
    while (true) {
      _cap >> _frame;
      hist.release();
      if (_frame.empty())
        return return_type::error;
      resize(_frame, _frame, Size(), 0.3, 0.3);
      cvtColor(_frame, _gray, COLOR_BGR2GRAY);
      calcHist(&_gray,
               1,         // number of images
               0,         // channels
               Mat(),     // mask
               hist,      // histogram
               1,         // dimensionality
               &hist_size, // number of bins
               0);
      normalize(hist, hist, 0, 255, NORM_MINMAX, CV_32F);

      putText(_frame, get_ISO8601(), Point{5, _frame.rows - 5}, 0, 0.5, 1, 1, 8,
              false);
      imshow("frame", _frame);
      // imshow("Histogram", hist_image);
      char k = waitKey(1000.0 / 25);
      if (' ' == k) {
        imwrite(_params["image_name"], _frame);
        if (blob) {
          imencode("." + _blob_format, _frame, *blob);
        }
        break;
      } else if ('q' == k) {
        result = return_type::critical;
        _error = "Quit requested";
        break;
      }
    }
    if (return_type::success == result) {
      (*out)["frame_size"] = {{cvRound(_cap.get(CAP_PROP_FRAME_WIDTH)),
                               cvRound(_cap.get(CAP_PROP_FRAME_HEIGHT))}};
      (*out)["image_size"] = {{_frame.cols, _frame.rows}};
      (*out)["histogram"] = json::array();
      for (int i = 0; i < hist_size; i++) {
        (*out)["histogram"].push_back(cvRound(hist.at<float>(i)));
      }
    }
    return result;
  }

  void set_params(void *params) override {
    _params = *(json *)params;
    int device = _params["device"];
    _cap.open(device, CAP_AVFOUNDATION);
    if (!_cap.isOpened()) {
      throw runtime_error("Error: Unable to open webcam");
    }
  }

private:
  json _data, _params;
  Mat _frame, _gray;
  VideoCapture _cap;
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

class WebcamDriver : public SourceDriver<json> {
public:
  WebcamDriver() : SourceDriver(PLUGIN_NAME, Webcam::version) {}
  Source<json> *create() { return new Webcam(); }
};

extern "C" EXPORTIT void register_pugg_plugin(pugg::Kernel *kernel) {
  kernel->add_driver(new WebcamDriver());
}

/*
                  _
  _ __ ___   __ _(_)_ __
 | '_ ` _ \ / _` | | '_ \
 | | | | | | (_| | | | | |
 |_| |_| |_|\__,_|_|_| |_|

For testing purposes, when directly executing the plugin
*/
int main(int argc, char const *argv[]) {
  Webcam wc;
  json output;
  vector<unsigned char> blob;
  wc.set_params(new json({
    {"device", 0}, 
    {"image_name", "image.jpg"}}));

  auto now = system_clock::now();
  auto today = floor<days>(now);
  auto tod = duration_cast<seconds>(now - today);
  cout << "Local time: " << tod.count() << endl;

  while (wc.get_output(&output, &blob) == return_type::success) {
    cout << "Webcam: " << output.dump() << endl;
    cout << "Image size: " << blob.size() << " bytes" << endl;
  }

  return 0;
}
