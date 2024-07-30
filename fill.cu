#include "fill.h"

  __global__
void _fill(uint64_t n, float val, float* out)
{
  uint64_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < n) {
    out[i] = val;
  }
}

void fill(
  cudaStream_t stream,
  uint64_t n, float val, float* out)
{
  dim3 blockSize(256);
  dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

  _fill<<<gridSize,blockSize,0,stream>>>(n, val, out);
}
