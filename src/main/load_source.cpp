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
    cout << "Usage: " << argv[0] << " <plugin> [name]" << endl;
    return 1;
  }

  cout << "Loading plugin... ";
  cout.flush();
  // load the plugin
  kernel.load_plugin(argv[1]);

  // find the proper driver in the plugin
  // - if there's only one, load it
  // - if there are more, list them and select the one passed on the CLI
  auto drivers = kernel.get_all_drivers<SourceDriverJ>(SourceJ::server_name());
  SourceDriverJ *driver = nullptr;
  if (drivers.size() == 1 && argc == 2) {
    driver = drivers[0];
    cout << "loaded default driver " << driver->name();
  } else if (drivers.size() > 1) {
    cout << "found multiple drivers:" << endl;
    for (auto &d : drivers) {
      cout << " - " << d->name();
      if (argc >= 3 && d->name() == argv[2]) {
        driver = d;
        cout << " -> selected" << endl;
      } else {
        cout << endl;
      }
    }
  }

  // No driver can be loaded
  if (!driver) {
    cout << "No driver to load, exiting" << endl;
    exit(1);
  }

  SourceJ *source = driver->create();
  // Now we can create an instance of class SourceJ from the driver
  cout << "Loaded plugin: " << source->kind() << endl;

  json params = {
      {"name", "clock test"}, {"device", 0}, {"image_name", "image.jpg"}};
  json out;
  source->set_params(&params);
  source->get_output(&out);
  cout << "Output: " << out << endl;
  delete source;

  kernel.clear_drivers();
}