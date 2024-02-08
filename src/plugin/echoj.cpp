#include <nlohmann/json.hpp>
#include "../filter.hpp"

using json = nlohmann::json;

class Echo : public Filter<json, json> {
public:
  std::string kind() override { return "EchoJ"; }
  bool load_data(json &d) override { 
    _data = d;
    return true; 
  }
  bool process(json *out) override { 
    (*out)["data"] = _data;
    (*out)["params"] = _params;
    return true; 
  }

  void set_params(void *params) override {
    _params = *(json *)params;
  }

private:
  json _data, _params;
};

/*
  ____  _             _       
 |  _ \| |_   _  __ _(_)_ __  
 | |_) | | | | |/ _` | | '_ \ 
 |  __/| | |_| | (_| | | | | |
 |_|   |_|\__,_|\__, |_|_| |_|
                |___/         
*/

#ifndef HAVE_MAIN
#include <pugg/Kernel.h>

class EchoDriver : public FilterDriver<json, json> {
public:
  EchoDriver() : FilterDriver("EchoJDriver", Echo::version) {}
  Filter<json, json> *create() { return new Echo(); }
};

extern "C" EXPORTIT void register_pugg_plugin(pugg::Kernel *kernel) {
  kernel->add_driver(new EchoDriver());
}


#else
using namespace std;
int main(int argc, char const *argv[])
{
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

#endif