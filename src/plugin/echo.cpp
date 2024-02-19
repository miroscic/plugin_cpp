/*
  _____     _                   _             _
 | ____|___| |__   ___    _ __ | |_   _  __ _(_)_ __
 |  _| / __| '_ \ / _ \  | '_ \| | | | |/ _` | | '_ \
 | |__| (__| | | | (_) | | |_) | | |_| | (_| | | | | |
 |_____\___|_| |_|\___/  | .__/|_|\__,_|\__, |_|_| |_|
                         |_|            |___/
*/
#include "../filter.hpp"
#include <pugg/Kernel.h>

#ifndef PLUGIN_NAME
#define PLUGIN_NAME "echo"
#endif

class Echo : public Filter<double, double> {
public:
  std::string kind() override { return PLUGIN_NAME; }
  return_type load_data(double &d) override {
    _data = d;
    return return_type::success;
  }
  return_type process(double *out) override {
    *out = _data;
    return return_type::success;
  }

  std::map<std::string, std::string> info() override {
    return {};
  };

private:
  double _data;
};

class EchoDriver : public FilterDriver<double, double> {
public:
  EchoDriver() : FilterDriver(PLUGIN_NAME, Echo::version) {}
  Filter<double, double> *create() { return new Echo(); }
};

extern "C" EXPORTIT void register_pugg_plugin(pugg::Kernel *kernel) {
  kernel->add_driver(new EchoDriver());
}

int main(int argc, char const *argv[]) {
  Echo echo;
  double data = 3.14, out = 0;
  echo.load_data(data);
  echo.process(&out);
  std::cout << "Input: " << data << std::endl;
  std::cout << "Output: " << out << std::endl;
  return 0;
}
