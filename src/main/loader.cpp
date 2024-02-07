#include "../filter.hpp"
#include <pugg/Kernel.h>

#include <iostream>
#include <thread>
#include <vector>

using namespace std;

// Parameters for twice plugin
struct TwiceParams {
  int times = 2;
};

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

  cout << "Loading plugins..." << endl;
  // load the first plugin
  {
    kernel.load_plugin(argv[1]);
    FilterDriver1 *echo_driver =
        kernel.get_driver<FilterDriver1>(Filter1::server_name(), "EchoDriver");
    Filter1 *echo = echo_driver->create();
    // Now we can use the filter echo_driver as an instance of Filter class
    cout << "\nLoaded filter: " << echo->kind() << endl;
    double data = 3.14, out = 0;
    echo->load_data(data);
    echo->process(&out);
    cout << "Input: " << data << endl;
    cout << "Output: " << out << endl;
    delete echo;
  }

  // Load the second plugin: it must have a double vector as input and output
  if (argc == 3) {
    kernel.load_plugin(argv[2]);
    FilterDriver2 *twice_driver =
        kernel.get_driver<FilterDriver2>(Filter2::server_name(), "TwiceDriver");
    Filter2 *twice = twice_driver->create();
    cout << "\nLoaded filter: " << twice->kind() << endl;
    Vec data = {1, 2, 3, 4};
    Vec result(data.size());
    TwiceParams params{3};
    twice->set_params(&params);
    twice->load_data(data);
    if (twice->process(&result)) {
      cout << "Input: " << endl << "{ ";
      for (auto &d : data) {
        cout << d << " ";
      }
      cout << "}" << endl;
      cout << "Output: " << endl << "{ ";
      for (auto &d : result) {
        cout << d << " ";
      }
      cout << "}" << endl;
    } else {
      cout << "Error processing data" << endl;
    }

    delete twice;
  }

  kernel.clear_drivers();
}