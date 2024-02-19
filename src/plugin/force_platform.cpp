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
#include <mlpack.hpp>
#include <armadillo>

#ifndef PLUGIN_NAME
#define PLUGIN_NAME "force_platform"
#endif

using namespace std;
using namespace std::chrono;
using json = nlohmann::json;

// Plugin class. This shall be the only part that needs to be modified,
// implementing the actual functionality
class ForcePlatform : public Source<json> {
public:
  ForcePlatform() {
    _error = "none";
    _blob_format = "jpg";
  }

  ~ForcePlatform() {

  }

  string kind() override { return PLUGIN_NAME; }

  return_type get_output(json *out,
                         std::vector<unsigned char> *blob = nullptr) override {
    return_type result = return_type::success;
    
    return result;
  }

  void set_params(void *params) override {
    _params = *(json *)params;
  }

private:
  json _data, _params;
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
  fp.set_params(new json({
    {"device", 0}, 
    {"image_name", "image.jpg"}, 
    {"hist_size", 50},
    {"scale", 1/3.0},
    {"flip", true}
  }));

  cout << "Press space to capture image, q to quit" << endl;

  while (fp.get_output(&output, nullptr) == return_type::success) {
    cout << "ForcePlatform plugin output: " << output.dump() << endl;
  }

  return 0;
}
