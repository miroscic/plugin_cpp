/*
  ____                           
 / ___|  ___  _   _ _ __ ___ ___ 
 \___ \ / _ \| | | | '__/ __/ _ \
  ___) | (_) | |_| | | | (_|  __/
 |____/ \___/ \__,_|_|  \___\___|
                                 
Base class for source plugins
*/
#ifndef FILTER_HPP
#define FILTER_HPP

#include <iostream>
#include <string>
#include <vector>
#include "common.hpp"

#ifdef _WIN32
#define EXPORTIT __declspec(dllexport)
#else
#define EXPORTIT
#endif

/*
 * Base class for sources
 *
 * This class is the base class for all sources. It defines the interface for
 * providing output of data internally acquired.
 * Child classes must implement the kind() and get_output() methods.
 * Optionally, they can implement the set_params() method to receive parameters
 * as a void pointer.
 *
 * @tparam Tout Output data type
 */
template <typename Tout = std::vector<double>>
class Source {
public:
  Source() : _error("No error") {}
  virtual ~Source() {}

  /*
   * Returns the kind of source
   *
   * This method returns the kind of source. It is used to identify the source
   * when loading it from a plugin.
   *
   * @return The kind of source
   */
  virtual std::string kind() = 0;

  /*
   * Get the output data
   *
   * This method provides the output data. It returns
   * true if the data was fetched successfully, and false otherwise.
   *
   * @param out The output data
   * @return True if the data was processed successfully, and false otherwise
   */
  virtual return_type get_output(Tout *out) = 0;

  /*
   * Sets the parameters
   *
   * This method sets the parameters for the source. It receives a void pointer
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

  /*
   * Set it to true to enable dummy mode
  */
  bool dummy;

  static const int version = 1;
  static const std::string server_name() { return "SourceServer"; }

private:
  std::string _error;
};

#ifndef HAVE_MAIN
#include <pugg/Driver.h>

template <typename Tout = std::vector<double>>
class SourceDriver : public pugg::Driver {
public:
  SourceDriver(std::string name, int version)
      : pugg::Driver(Source<Tout>::server_name(), name, version) {}
  virtual Source<Tout> *create() = 0;
};
#endif

#endif // FILTER_HPP