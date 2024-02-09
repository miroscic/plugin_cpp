#include "../source.hpp"
#include <iostream>
#include <nlohmann/json.hpp>
#include <pugg/Kernel.h>
#include <thread>
#include <vector>

using namespace std;

using json = nlohmann::json;
using SourceJ = Source<json>;
using SourceDriverJ = SourceDriver<json>;

int main(int argc, char *argv[]) {
  pugg::Kernel kernel;
  // add a generic server to the kernel to initilize it
  // kernel.add_server(Filter<>::filter_name(),
  //                   Filter<>::version);
  kernel.add_server<Source<>>();

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
  SourceDriverJ *echo_driver =
      kernel.get_driver<SourceDriverJ>(SourceJ::server_name(), "clock");
  SourceJ *echo = echo_driver->create();
  // Now we can use the filter echo_driver as an instance of Filter class
  cout << "Loaded plugin: " << echo->kind() << endl;

  json params = {{"name", "clock test"}};
  json out;
  echo->set_params(&params);
  echo->get_output(&out);
  cout << "Output: " << out << endl;
  delete echo;

  kernel.clear_drivers();
}