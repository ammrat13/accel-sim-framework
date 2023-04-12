#include <cassert>
#include <fstream>
#include <iostream>

#include "Simulator.hpp"
#include "arguments.hpp"

using namespace gasan::alloc_sim;

// Check type equivalences
// This is needed for string parsing
static_assert(std::is_same<uint64_t, unsigned long>::value);
static_assert(std::is_same<size_t, unsigned long>::value);

int main(int argc, char **argv) {

  // Parse command-line arguments
  // This initializes ARGS
  args::parse(argc, argv);

  // Open the file for reading
  std::ifstream trace(ARGS.get("TRACE"));
  if (!trace.is_open()) {
    std::cerr << "error: could not open file '" << ARGS.get("TRACE") << "'"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // Setup exceptions for unlikely failures
  // We don't do failbit because that happens if we have a trailing newline, as
  // we should.
  trace.exceptions(std::ios::badbit);

  // Simulate all the trace entries
  Simulator sim;
  while (!trace.eof()) {
    // Read the line
    // Skip it if it is empty, as will happen if we have a trailing newline
    std::string line;
    std::getline(trace, line);
    if (line.size() == 0)
      continue;

    // Convert to a stream because it's more convenient
    std::stringstream line_stream(line);
    line_stream.exceptions(std::ios::badbit | std::ios::failbit);

    // Figure out the command we need to run
    std::string command;
    std::getline(line_stream, command, ',');

    if (command == "cudaMalloc") {
      // Get parameters
      uint64_t tag;
      size_t sz;
      {
        std::string tag_str;
        std::string sz_str;
        std::getline(line_stream, tag_str, ',');
        std::getline(line_stream, sz_str, ',');
        tag = std::stoul(tag_str, nullptr, 0);
        sz = std::stoul(sz_str, nullptr, 0);
        assert(line_stream.eof() && "Too many arguments");
      }
      // Do the command
      sim.cudaMalloc(tag, sz);
      continue;
    }

    if (command == "cudaFree") {
      // Get parameters
      uint64_t tag;
      {
        std::string tag_str;
        std::getline(line_stream, tag_str, ',');
        tag = std::stoul(tag_str, nullptr, 0);
        assert(line_stream.eof() && "Too many arguments");
      }
      // Do the command
      sim.cudaFree(tag);
      continue;
    }

    std::cerr << "error: unknown command '" << command << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Die if error
  if (trace.fail() && !trace.eof()) {
    std::cerr << "error: could not parse file '" << ARGS.get("TRACE") << "'"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Print statistics and exit
  std::cout << sim.getStats();
}
