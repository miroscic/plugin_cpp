# Plugins!

This example project explores how to develop a plugin system for a C++ application. It is based on the [pugg plugin system]().

## Building

To build the project, you need to have CMake installed. Then, you can run the following commands:

```bash
mkdir build
ccmake -Bbuild -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This creates two plugins (in the form of two shared libraries) and an executable that uses them. The plugins are named `echo` and `twice`. The first one simply echoes the scalar input, while the second one takes a `std::vector<double>` and twices each element.

Plugins are named **Filters**, for they are expected to act as filters, taking an input and producing an output. The plugins must be implemented as derived classes of the templated class `Filter` (see `src/filter.hpp`).