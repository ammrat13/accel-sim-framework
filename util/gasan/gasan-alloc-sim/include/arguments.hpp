#pragma once

#include "argparse/argparse.hpp"

namespace gasan::alloc_sim {

/**
 * \brief Global arguments object
 *
 * This object is used to hold all of the arguments for the program. It is
 * initialized on program startup.
 */
extern argparse::ArgumentParser ARGS;

namespace args {

/**
 * \brief Function to parse all arguments
 *
 * This will populate the ARGS object with information on all the arguments,
 * then try to parse the command line options provided. If it fails, this
 * function terminates the program.
 */
void parse(int argc, char **argv);

} // namespace args
} // namespace gasan::alloc_sim
