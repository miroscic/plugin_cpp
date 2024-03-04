/*
  ____  _        _      _              _             _____ ____
 / ___|| | _____| | ___| |_ ___  _ __ (_)_______ _ _|___ /|  _ \
 \___ \| |/ / _ \ |/ _ \ __/ _ \| '_ \| |_  / _ \ '__||_ \| | | |
  ___) |   <  __/ |  __/ || (_) | | | | |/ /  __/ |  ___) | |_| |
 |____/|_|\_\___|_|\___|\__\___/|_| |_|_/___\___|_| |____/|____/
 */

#include "../source.hpp"
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <pugg/Kernel.h>


using namespace cv;
using namespace std;
using json = nlohmann::json;

class Skeletonizer3D : public Source<json> {
public:
  int acquire_frame();               // Nicola
  int skeleton_from_depth_compute(); // Nicola
  int skeleton_from_rgb_compute();   // Alessandro
  int point_cloud_filter();          // Nicola
  int hessian_compute();             // ? Alessandro
  int cov3D_from_cov2D_compute();    // Alessandro
  int cov3D_adjust();                // Marco == Matteo

  // Protocollo plugin
  void set_params(void *params); //

  return_type
  get_output(json *out, vector<unsigned_char> *blob) map<string, string> info();

  string kind();

private:
  Mat _rgbd, _rgb;
  map<string, vector> _skeleton2D, _skeleton3D;
  Mat _heatmap;
  Mat _point_cloud;
  Mat _cov2D, _cov3D, _cov3D_adj;
  json _params;
}

INSTALL_SOURCE_DRIVER(Skeletonizer3D, json);

/*
Example of JSON parameters:
{
    "device": id or relative_video_uri,
    "resolution_rgbd": "lxh",
    "resolution_rgb": "lxh",
    "fps": 30,
    "extrinsic": [[diagonal],[off_diagonal],[translation]]
}
*/