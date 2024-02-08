/*
  _____     _               _         _             _
 | ____|___| |__   ___     | |  _ __ | |_   _  __ _(_)_ __
 |  _| / __| '_ \ / _ \ _  | | | '_ \| | | | |/ _` | | '_ \
 | |__| (__| | | | (_) | |_| | | |_) | | |_| | (_| | | | | |
 |_____\___|_| |_|\___/ \___/  | .__/|_|\__,_|\__, |_|_| |_|
                               |_|            |___/
It takes a JSON input and returns it as output, enriched with the parameters
*/

#include "../filter.hpp"
#include <nlohmann/json.hpp>
#include <pugg/Kernel.h>

#ifndef PLUGIN_NAME
#define PLUGIN_NAME "echoj"
#endif

using namespace std;
using json = nlohmann::json;

class Echo : public Filter<json, json> {
public:
  string kind() override { return PLUGIN_NAME; }
  bool load_data(json &d) override {
    _data = d;
    return true;
  }
  bool process(json *out) override {
    (*out)["data"] = _data;
    (*out)["params"] = _params;
    return true;
  }

  void set_params(void *params) override { _params = *(json *)params; }

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
*/

class EchoDriver : public FilterDriver<json, json> {
public:
  EchoDriver() : FilterDriver(PLUGIN_NAME, Echo::version) {}
  Filter<json, json> *create() { return new Echo(); }
};

extern "C" EXPORTIT void register_pugg_plugin(pugg::Kernel *kernel) {
  kernel->add_driver(new EchoDriver());
}

/*
                  _
  _ __ ___   __ _(_)_ __
 | '_ ` _ \ / _` | | '_ \
 | | | | | | (_| | | | | |
 |_| |_| |_|\__,_|_|_| |_|

*/

int main(int argc, char const *argv[]) {
  Echo echo;
  json params = {{"times", 3}};
  json data = {{"array", {1, 2, 3, 4}}};
  echo.set_params(&params);
  echo.load_data(data);
  json result;
  if (!echo.process(&result)) {
    cerr << "Error processing data" << endl;
    return 1;
  }
  cout << "Input: " << data << endl;
  cout << "Output: " << result << endl;

  return 0;
}
