#include "setup.h"

#include <cuda_runtime.h>
#include <algorithm>

#include "fill.h"

void _check_cuda(cudaError_t err, string filename, int line) {
  if(err != cudaSuccess) {
    throw std::runtime_error(
      "cuda no success at " + filename + ":" + write_with_ss(line) 
      + ". Error: " + cudaGetErrorString(err));
  }
}
#define check_cuda(call) _check_cuda(call, __FILE__, __LINE__)

// stream1 is the origin stream
// cudaStreamBeginCapture(stream1);
//
// kernel_A<<< ..., stream1 >>>(...);
//
// // Fork into stream2
// cudaEventRecord(event1, stream1);
// cudaStreamWaitEvent(stream2, event1);
//
// kernel_B<<< ..., stream1 >>>(...);
// kernel_C<<< ..., stream2 >>>(...);
//
// // Join stream2 back to origin stream (stream1)
// cudaEventRecord(event2, stream2);
// cudaStreamWaitEvent(stream1, event2);
//
// kernel_D<<< ..., stream1 >>>(...);
//
// // End capture in the origin stream
// cudaStreamEndCapture(stream1, &graph);
//
// // stream1 and stream2 no longer in capture mode

int main() {
  uint64_t n = 10000;

  cudaEvent_t event_init_at_device_0, event_final_at_device_1;

  cudaStream_t stream0, stream1;
  void *mem00, *mem01, *mem10, *mem11;

  void* out0 = malloc(n * sizeof(float));
  void* out1 = malloc(n * sizeof(float));

  check_cuda(cudaSetDevice(0));
  check_cuda(cudaEventCreate(&event_init_at_device_0)); 
  check_cuda(cudaStreamCreate(&stream0));
  check_cuda(cudaMalloc(&mem00, n*sizeof(float)));
  check_cuda(cudaMalloc(&mem01, n*sizeof(float)));

  check_cuda(cudaSetDevice(1));
  check_cuda(cudaEventCreate(&event_final_at_device_1));
  check_cuda(cudaStreamCreate(&stream1));
  check_cuda(cudaMalloc(&mem10, n*sizeof(float)));
  check_cuda(cudaMalloc(&mem11, n*sizeof(float)));

  //////////////

  // Begin capture on stream 0
  check_cuda(cudaSetDevice(0));
  check_cuda(cudaStreamBeginCapture(stream0, cudaStreamCaptureModeGlobal));

  // Tell stream1 that stream0 has started
  check_cuda(cudaEventRecord(event_init_at_device_0, stream0));
  check_cuda(cudaStreamWaitEvent(stream1, event_init_at_device_0));

  // At device 0, fill mem00 with a value and copy to mem11 on device 1
  fill(stream0, n, 1.953, reinterpret_cast<float*>(mem00));
  check_cuda(cudaMemcpyAsync(
    mem11, mem00, n*sizeof(float), cudaMemcpyDeviceToDevice, 
    stream0));

  // At device 1, fill mem10 with a value and copy to mem01 on device 0
  check_cuda(cudaSetDevice(1));
  fill(stream1, n, 6.824, reinterpret_cast<float*>(mem10));
  check_cuda(cudaMemcpyAsync(
    mem01, mem10, n*sizeof(float), cudaMemcpyDeviceToDevice, 
    stream1));

  // Tell stream0 on device0 that stream1 on device1 has finished
  check_cuda(cudaEventRecord(event_final_at_device_1, stream1));
  check_cuda(cudaStreamWaitEvent(stream0, event_final_at_device_1));

  // create the graph from stream0
  cudaGraph_t graph;
  check_cuda(cudaStreamEndCapture(stream0, &graph));

  //////////////
  cudaGraphExec_t instance;
  check_cuda(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

  check_cuda(cudaGraphLaunch(instance, stream0));

  check_cuda(cudaStreamSynchronize(stream0));

  //check_cuda(cudaSetDevice(0));
  //check_cuda(cudaDeviceSynchronize());

  //check_cuda(cudaSetDevice(1));
  //check_cuda(cudaDeviceSynchronize());
  
  //////////////
  check_cuda(cudaMemcpy(out0, mem00, n*sizeof(float), cudaMemcpyDeviceToHost));
  check_cuda(cudaMemcpy(out1, mem10, n*sizeof(float), cudaMemcpyDeviceToHost));

  for(int i = 0; i != std::min(uint64_t(10), n); ++i) {
    DOUT(
      (reinterpret_cast<float*>(out0)[i]) << " " <<
      (reinterpret_cast<float*>(out1)[i])          );
  }

  //////////////
  // Not bothering to free memory here.
}
