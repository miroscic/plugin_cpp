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

int main(int argc, char *argv[]) {
  pugg::Kernel kernel;
  // add a generic server to the kernel to initilize it
  kernel.add_server(Filter<>::filter_name(),
                    Filter<>::version);
  
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
    FilterDriver<double, double> *echo_driver =
        kernel.get_driver<FilterDriver<double, double>>(
            Filter<double, double>::filter_name(), "EchoDriver");
    Filter<double, double> *echo = echo_driver->create();
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
    FilterDriver<vector<double>, vector<double>> *twice_driver =
        kernel.get_driver<FilterDriver<vector<double>, vector<double>>>(
            Filter<vector<double>, vector<double>>::filter_name(),
            "TwiceDriver");
    Filter<vector<double>, vector<double>> *twice = twice_driver->create();
    cout << "\nLoaded filter: " << twice->kind() << endl;
    vector<double> data = {1, 2, 3, 4};
    vector<double> result(data.size());
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