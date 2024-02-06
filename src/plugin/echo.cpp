#include "../filter.hpp"

class Echo : public Filter<double, double> {
public:
  std::string kind() override { return "Echo"; }
  bool load_data(double &d) override { 
    _data = d;
    return true; 
  }
  bool process(double *out) override { 
    *out = _data;
    return true; 
  }

private:
  double _data;
};

#ifndef HAVE_MAIN
#include <pugg/Kernel.h>
class EchoDriver : public FilterDriver<double, double> {
public:
  EchoDriver() : FilterDriver("EchoDriver", Echo::version) {}
  Filter<double, double> *create() { return new Echo(); }
};

extern "C" EXPORTIT void register_pugg_plugin(pugg::Kernel *kernel) {
  kernel->add_driver(new EchoDriver());
}
#endif