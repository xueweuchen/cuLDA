#ifndef _CUDA_UTIL_
#define _CUDA_UTIL_

#include <cstdlib>
#include <cstdio>
#include "cuda_runtime.h"
#include "device_atomic_functions.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>


const int CUDA_NUM_THREADS = 512;

inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

static void HandleError(cudaError_t err,
  const char *file,
  int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err),
      file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}


#endif // !_CUDA_UTIL_
