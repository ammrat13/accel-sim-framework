#define _GNU_SOURCE
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// Environment variable we read to get the output filename
const char *const OUT_NAME_ENVVAR = "GASAN_ALLOC_RECORD_OUT_FNAME";

// File that we write to as output
// By default is NULL, and has to be setup in can_output. This file is never
// closed, but that should be fine
FILE *out = NULL;

// Setup the output file
// Return whether or not to log. So if this returns false, the instrumented
// function should not output anything.
static bool can_output(void) {

  // If we already have an output file, we're done
  if (out != NULL)
    return true;

  // Otherwise, get the filename from an environment variable
  const char *out_name = getenv(OUT_NAME_ENVVAR);
  // Fail if the environment variable doesn't exist
  if (out_name == NULL)
    return false;
  // Fail if we can't open
  out = fopen(out_name, "w");
  if (out == NULL)
    return false;

  // We succeeded, and the output is ready to write
  return true;
}

// Macro to handle common tracing functionality
// It looks up the original function and puts that in a predefined variable
#define TRACE(func, args...) \
  cudaError_t func(args) { \
    static cudaError_t (*real_##func)(args) = NULL; \
    if (real_##func == NULL) \
      real_##func = dlsym(RTLD_NEXT, #func); \
    if (real_##func == NULL) \
      return cudaErrorInitializationError;

#define ENDTRACE }

// -----------------------------------------------------------------------------

TRACE(cudaMalloc, void **devPtr, size_t size)
  cudaError_t ret = real_cudaMalloc(devPtr, size);
  if (can_output()) {
    if (devPtr != NULL)
      fprintf(out, "cudaMalloc,%p,%zu\n", *devPtr, size);
    else
      fprintf(out, "cudaMalloc,(nil),%zu\n", size);
  }
  return ret;
ENDTRACE

TRACE(cudaFree, void *devPtr)
  cudaError_t ret = real_cudaFree(devPtr);
  if (can_output())
    fprintf(out, "cudaFree,%p\n", devPtr);
  return ret;
ENDTRACE
