/*
  _____ _ _ _
 |  ___(_) | |_ ___ _ __
 | |_  | | | __/ _ \ '__|
 |  _| | | | ||  __/ |
 |_|   |_|_|\__\___|_|

Base class for filter plugins
*/
#ifndef FILTER_HPP
#define FILTER_HPP

#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#define EXPORTIT __declspec(dllexport)
#else
#define EXPORTIT
#endif

/*
 * Base class for filters
 *
 * This class is the base class for all filters. It defines the interface for
 * loading data and processing it.
 * Child classes must implement the kind, load_data and process methods.
 * Optionally, they can implement the set_params method to receive parameters
 * as a void pointer.
 *
 * @tparam Tin Input data type
 * @tparam Tout Output data type
 */
template <typename Tin = std::vector<double>,
          typename Tout = std::vector<double>>
class Filter {
public:
  Filter() : _error("No error") {}
  virtual ~Filter() {}

  /*
   * Returns the kind of filter
   *
   * This method returns the kind of filter. It is used to identify the filter
   * when loading it from a plugin.
   *
   * @return The kind of filter
   */
  virtual std::string kind() = 0;

  /*
   * Loads the input data
   *
   * This method loads the input data into the filter. It returns true if the
   * data was loaded successfully, and false otherwise.
   *
   * @param data The input data
   * @return True if the data was loaded successfully, and false otherwise
   */
  virtual bool load_data(Tin &data) = 0;

  /*
   * Processes the input data
   *
   * This method processes the input data and returns the result. It returns
   * true if the data was processed successfully, and false otherwise.
   *
   * @param out The output data
   * @return True if the data was processed successfully, and false otherwise
   */
  virtual bool process(Tout *out) = 0;

  /*
   * Sets the parameters
   *
   * This method sets the parameters for the filter. It receives a void pointer
   * to the parameters. The child class must cast the pointer to the correct
   * type.
   *
   * @param params The parameters (typically a pointer to a struct)
   */
  virtual void set_params(void *params){};

  /*
   * Returns the error message
   *
   * This method returns the error message.
   *
   * @return The error message
   */
  std::string error() { return _error; }

  static const int version = 1;
  static const std::string server_name() { return "FilterServer"; }

private:
  std::string _error;
};

#ifndef HAVE_MAIN
#include <pugg/Driver.h>

template <typename Tin = std::vector<double>,
          typename Tout = std::vector<double>>
class FilterDriver : public pugg::Driver {
public:
  FilterDriver(std::string name, int version)
      : pugg::Driver(Filter<Tin, Tout>::server_name(), name, version) {}
  virtual Filter<Tin, Tout> *create() = 0;
};
#endif

#endif // FILTER_HPP