#include "arguments.hpp"

#include <cstdlib>
#include <iostream>
#include <type_traits>

using namespace gasan::alloc_sim;
using namespace gasan::alloc_sim::args;
using namespace argparse;

// Check type equivalences
// This is needed for the argparse library
static_assert(std::is_same<size_t, unsigned long>::value);

ArgumentParser gasan::alloc_sim::ARGS("gasan-alloc-sim", "0.0");

void gasan::alloc_sim::args::parse(int argc, char **argv) {

  ARGS.add_argument("TRACE").help("the trace file to simulate");

  ARGS.add_argument("-s", "--scale")
      .metavar("SCALE")
      .default_value(8ul)
      .scan<'u', size_t>()
      .help("how many bytes of normal memory one byte of shadow memory codes "
            "for (log-base-2)");

  ARGS.add_argument("-b", "--redzone-base")
      .metavar("REDZONE-BASE")
      .default_value(256ul)
      .scan<'i', size_t>()
      .help("minimum number of redzone bytes per allocation");
  ARGS.add_argument("-l", "--redzone-linear")
      .metavar("REDZONE-SCALE")
      .default_value(0.5)
      .scan<'f', double>()
      .help("number of redzone bytes as a fraction of allocation size");

  ARGS.add_argument("-d", "--debug")
      .default_value(false)
      .implicit_value(true)
      .help("print debug outputs for the region list");

  // Try to parse
  // See: https://github.com/p-ranav/argparse
  try {
    ARGS.parse_args(argc, argv);

    // Use this opportunity to verify invariants that argparse can't check
    {
      size_t sc = ARGS.get<size_t>("--scale");
      if (sc >= 64)
        throw std::runtime_error("scale must be less than 64");
    }
    {
      double rs = ARGS.get<double>("--redzone-linear");
      if (rs < 0.0 || 1.0 < rs)
        throw std::runtime_error("redzone linear must be between 0.0 and 1.0");
    }

  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << ARGS;
    std::exit(EXIT_FAILURE);
  } catch (const std::invalid_argument &err) {
    std::cerr << ARGS;
    std::exit(EXIT_FAILURE);
  }
}
