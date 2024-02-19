/*
  _____                          _       _    __                      
 |  ___|__  _ __ ___ ___   _ __ | | __ _| |_ / _| ___  _ __ _ __ ___  
 | |_ / _ \| '__/ __/ _ \ | '_ \| |/ _` | __| |_ / _ \| '__| '_ ` _ \ 
 |  _| (_) | | | (_|  __/ | |_) | | (_| | |_|  _| (_) | |  | | | | | |
 |_|  \___/|_|  \___\___| | .__/|_|\__,_|\__|_|  \___/|_|  |_| |_| |_|
                          |_|                                         
*/

#include "../source.hpp"
#include <chrono>
#include <nlohmann/json.hpp>
#include <pugg/Kernel.h>
#include <sstream>
#include <thread>
#include <tuple>
#include <mlpack.hpp>

#ifndef PLUGIN_NAME
#define PLUGIN_NAME "force_platform"
#endif

using namespace std;
using json = nlohmann::json;
using namespace arma;
using namespace mlpack;

// Plugin class. This shall be the only part that needs to be modified,
// implementing the actual functionality
class ForcePlatform : public Source<json> {
public:
  ForcePlatform() : _image(""), _dummy(false) {
  }

  ~ForcePlatform() {
  }

  string kind() override { return PLUGIN_NAME; }

  return_type get_output(json *out,
                         std::vector<unsigned char> *blob = nullptr) override {
    return_type result = return_type::success;
    vec x, y, z, xz, yz, b;
    double mass = 0.0;
    double force = 0.0;

    if (_dummy && _image.empty()) {
      result = get_dummy();
    } else if (!_image.empty()) {
      result = get_image();
    } else {
      throw std::runtime_error("Not implemented");
    }
    if (result == return_type::success) {
      throw(std::runtime_error("Error acquiring data"));
    }
    find_clusters();

    (*out)["n_clusters"] = _n_clusters;
    for (size_t i = 0; i < _n_clusters; i++) {
      x = _data.submat(uvec({0}), find(_assignments == i)).as_col();
      y = _data.submat(uvec({1}), find(_assignments == i)).as_col();
      z = _data.submat(uvec({2}), find(_assignments == i)).as_col();
      force = sum(z);
      if (_include_clusters) {
        (*out)["clusters"][i]["x"] = x;
        (*out)["clusters"][i]["y"] = y;
        (*out)["clusters"][i]["z"] = z;
      }
      xz = x % z;
      yz = y % z;
      mass = sum(z);
      b = {sum(xz) / mass, sum(yz) / mass};
      b = round(b);
      (*out)["barycenters"][i]["x"] = b(0);
      (*out)["barycenters"][i]["y"] = b(1);
      (*out)["barycenters"][i]["f"] = force;
      (*out)["centroids"][i]["x"] = _centroids(0, i);
      (*out)["centroids"][i]["y"] = _centroids(1, i);
    }
    if (_include_raw_data) {
      (*out)["data"]["x"] = _data.row(0);
      (*out)["data"]["y"] = _data.row(1);
      (*out)["data"]["z"] = _data.row(2);
    }
    
    return result;
  }

  void set_params(void *params) override {
    _params = *(json *)params;
    if (_params.contains("dummy") && _params["dummy"].is_boolean()) {
      _dummy = _params["dummy"];
    }
    if (_params.contains("image") && _params["image"].is_string()) {
      _image = _params["image"];
    }
    if (_params.contains("include_clusters") && _params["include_clusters"].is_boolean()) {
      _include_clusters = _params["include_clusters"];
    }
    if (_params.contains("include_raw_data") && _params["include_raw_data"].is_boolean()) {
      _include_raw_data = _params["include_raw_data"];
    }
    if (_params.contains("epsilon") && _params["epsilon"].is_number()) {
      _epsilon = _params["epsilon"];
    }
    if (_params.contains("min_pts") && _params["min_pts"].is_number()) {
      _min_pts = _params["min_pts"];
    }
    if (_params.contains("platform_size") && _params["platform_size"].is_object()) {
      _platform_size = make_tuple(_params["platform_size"]["x"], _params["platform_size"]["y"]);
    }
    if (_params.contains("dummy_pts") && _params["dummy_pts"].is_object()) {
      _rows = _params["dummy_pts"]["x"];
      _cols = _params["dummy_pts"]["y"];
    }
    if (_params["clean_threshold"].is_number()) {
      _clean_threshold = _params["clean_threshold"];
    }
    if (_params["conv_factor"].is_number()) {
      _conv_factor = _params["conv_factor"];
    }
    if (_params.contains("coord_transform") && _params["coord_transform"].is_object()) {
      auto h = _params["coord_transform"];
      _rot = h["rot"];
      _htm(0, 0) = cos(_rot / 180 * M_PI);
      _htm(0, 1) = -sin(_rot / 180 * M_PI);
      _htm(1, 0) = sin(_rot / 180 * M_PI);
      _htm(1, 1) = cos(_rot / 180 * M_PI);
      _htm(0, 2) = h["dx"];
      _htm(1, 2) = h["dy"];
    }
  }

  map<string, string> info() override {
    map<string, string> m;
    m["dummy"] = _dummy ? "yes" : "no";
    m["image"] = _image;
    m["include_clusters"] = _include_clusters ? "yes" : "no";
    m["include_raw_data"] = _include_raw_data ? "yes" : "no";
    m["epsilon"] = to_string(_epsilon);
    m["min_pts"] = to_string(_min_pts);
    m["platform_size"] = "[" + to_string(get<0>(_platform_size)) + ", " + to_string(get<1>(_platform_size)) + "]";
    m["dummy_pts"] = "[" + to_string(_rows) + ", " + to_string(_cols) + "]";
    m["clean_threshold"] = to_string(_clean_threshold);
    m["conv_factor"] = to_string(_conv_factor);
    m["coord_transform"] = "rot: " + to_string(_rot) + ", trans [" + to_string(_htm(0, 2)) + ", " + to_string(_htm(1, 2)) + "]";
    return m;
  }

  return_type get_dummy() {
    return return_type::success;
  }

  return_type get_image() {
    return return_type::success;
  }

  void find_clusters() {

  }


private:
  json _params;
  bool _dummy;
  string _image;
  arma::Row<size_t> _assignments;
  mat _data, _centroids;
  // Parameters
  size_t _n_clusters = 0;
  bool _include_clusters = false;
  bool _include_raw_data = false;
  bool do_transform = false;
  double _clean_threshold = 5.0;
  double _conv_factor = 1.0;
  double _epsilon = 3.0;
  size_t _min_pts = 3;
  tuple<double, double> _platform_size{1230, 560};
  uword _rows = 123, _cols = 56;
  mat _htm = eye(3, 3);
  double _rot = 0;
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

class ForcePlatformDriver : public SourceDriver<json> {
public:
  ForcePlatformDriver() : SourceDriver(PLUGIN_NAME, ForcePlatform::version) {}
  Source<json> *create() { return new ForcePlatform(); }
};

extern "C" EXPORTIT void register_pugg_plugin(pugg::Kernel *kernel) {
  kernel->add_driver(new ForcePlatformDriver());
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
  ForcePlatform fp;
  json output;
  json params = json({
    {"dummy", true},
    {"image", "data/test.jpg"}
  });
  params["platform_size"] = {{"x", 1230}, {"y", 560}};
  fp.set_params(&params);


  for (auto &[k, v] : fp.info()) {
    cout << k << ": " << v << endl;
  }

  return 0;
}
