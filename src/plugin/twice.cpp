#include "../filter.hpp"

using namespace std;
using Vec = std::vector<double>;

struct TwiceParams {
  int times = 2;
};

// The filter class
// We use the defauilt template parameters std::vector<double> for the input and
// output data
class Twice : public Filter<> {
public:
  string kind() override { return "Twice"; }
  bool load_data(Vec &d) override {
    _data = d;
    return true;
  }
  bool process(Vec *out) override {
    if (out == nullptr) {
      return false;
    }
    out->clear();
    for (auto &d : _data) {
      out->push_back(d * _params.times);
    }
    return true;
  }
  void set_params(void *params) override {
    _params = *(TwiceParams *)params;
  }

private:
  Vec _data;
  TwiceParams _params;
};

#ifndef HAVE_MAIN
#include <pugg/Kernel.h>

class TwiceDriver : public FilterDriver<> {
public:
  TwiceDriver() : FilterDriver("TwiceDriver", Twice::version) {}
  Filter<> *create() { return new Twice(); }
};

extern "C" EXPORTIT void register_pugg_plugin(pugg::Kernel *kernel) {
  kernel->add_driver(new TwiceDriver());
}


#else

int main(int argc, char const *argv[])
{
  Twice twice;
  TwiceParams params{3};
  vector<double> data = {1, 2, 3, 4};
  twice.set_params(&params);
  twice.load_data(data);
  vector<double> result(data.size());
  twice.process(&result);
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

  return 0;
}

#endif