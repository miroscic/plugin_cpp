#ifndef COMMON_HPP
#define COMMON_HPP

#define INSTALL_SOURCE_DRIVER(klass, type)                                     \
  class klass##Driver : public SourceDriver<type> {                            \
  public:                                                                      \
    klass##Driver() : SourceDriver(PLUGIN_NAME, klass::version) {}             \
    Source<type> *create() { return new klass(); }                             \
  };                                                                           \
  extern "C" EXPORTIT void register_pugg_plugin(pugg::Kernel *kernel) {        \
    kernel->add_driver(new klass##Driver());                                   \
  }

#define INSTALL_FILTER_DRIVER(klass, type_in, type_out)                        \
  class klass##Driver : public FilterDriver<type_in, type_out> {               \
  public:                                                                      \
    klass##Driver() : FilterDriver(PLUGIN_NAME, klass::version) {}             \
    Filter<type_in, type_out> *create() { return new klass(); }                \
  };                                                                           \
  extern "C" EXPORTIT void register_pugg_plugin(pugg::Kernel *kernel) {        \
    kernel->add_driver(new klass##Driver());                                   \
  }

enum class return_type {
  success = 0,
  warning,
  error,
  critical
};


#endif // COMMON_HPP