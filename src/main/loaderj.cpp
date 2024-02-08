#include "../filter.hpp"
#include <pugg/Kernel.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <thread>
#include <vector>

using namespace std;

// Parameters for twice plugin
struct TwiceParams {
  int times = 2;
};

using json = nlohmann::json;
using FilterJ = Filter<json, json>;
using FilterDriverJ = FilterDriver<json, json>;
using Vec = vector<double>;
using Filter1 = Filter<double, double>;
using FilterDriver1 = FilterDriver<double, double>;
using Filter2 = Filter<Vec, Vec>;
using FilterDriver2 = FilterDriver<Vec, Vec>;

int main(int argc, char *argv[]) {
  pugg::Kernel kernel;
  // add a generic server to the kernel to initilize it
  // kernel.add_server(Filter<>::filter_name(),
  //                   Filter<>::version);
  kernel.add_server<Filter<>>();

  // CLI needs unoe or two plugin paths
  // the first on must have doubles as input and output
  if (argc < 2) {
    cout << "Usage: " << argv[0] << " <plugin>" << endl;
    return 1;
  }

  cout << "Loading plugin... ";
  cout.flush();
  // load the first plugin
  kernel.load_plugin(argv[1]);
  FilterDriverJ *echo_driver =
      kernel.get_driver<FilterDriverJ>(FilterJ::server_name(), "echoj");
  FilterJ *echo = echo_driver->create();
  // Now we can use the filter echo_driver as an instance of Filter class
  cout << "Loaded plugin: " << echo->kind() << endl;

  json in = {{"array", {1, 2, 3, 4}}};
  json params = {{"name", "echo test"}};
  json out;
  echo->set_params(&params);
  echo->load_data(in);
  echo->process(&out);
  cout << "Input: " << in << endl;
  cout << "Output: " << out << endl;
  delete echo;

  kernel.clear_drivers();
}