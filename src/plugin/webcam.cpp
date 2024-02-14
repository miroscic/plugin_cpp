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
    cap.release();
    destroyAllWindows();
  }

  string kind() override { return PLUGIN_NAME; }

  static string get_ISO8601(const system_clock::time_point &time = chrono::system_clock::now()) {
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

  return_type get_output(json *out, std::vector<unsigned char> *blob = nullptr) override {
    return_type result = return_type::success;
    while (true) {
      cap >> frame;
      if (frame.empty())
        return return_type::error;
      resize(frame, frame, Size(), 0.3, 0.3);
      putText(frame, get_ISO8601(), Point{5, frame.rows - 5}, 0, 0.5, 1, 1, 8, false);
      imshow("frame", frame);
      char k = waitKey(1000.0/25);
      if(' ' == k) {
        imwrite(_params["image_name"], frame);
        if (blob) {
          imencode("." + _blob_format, frame, *blob);
        }
        break;
      } else if ('q' == k) {
        result = return_type::critical;
        _error = "Quit requested";
        break;
      }
    }
    if (return_type::success == result) {
      (*out)["frame_size"] = {{cvRound(cap.get(CAP_PROP_FRAME_WIDTH)), cvRound(cap.get(CAP_PROP_FRAME_HEIGHT))}};
      (*out)["image_size"] = {{frame.cols, frame.rows}};
    }
    return result;
  }

  void set_params(void *params) override { 
    _params = *(json *)params; 
    int device = _params["device"];
    cap.open(device, CAP_AVFOUNDATION);
    if (!cap.isOpened()) {
      throw runtime_error("Error: Unable to open webcam");
    }
    
  }

private:
  json _data, _params;
  Mat frame;
  VideoCapture cap;
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
  wc.set_params(new json({{"device", 0}, {"image_name", "image.jpg"}}));

  auto now = system_clock::now();
  auto today = floor<days>(now);
  auto tod = duration_cast<seconds>(now - today);
  cout << "Local time: " << tod.count() << endl;

  wc.get_output(&output, &blob);

  cout << "Webcam: " << output.dump() << endl;
  cout << "Image size: " << blob.size() << " bytes" << endl;
  

  return 0;
}
