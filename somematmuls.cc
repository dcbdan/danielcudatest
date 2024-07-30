#include "setup.h"

#include <condition_variable>
#include <mutex>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "fill.h"

void _check_cublas(cublasStatus_t err, string filename, int line) {
  if(err != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
      "cublas no success at " + filename + ":" + write_with_ss(line));
  }
}
#define check_cublas(call) _check_cublas(call, __FILE__, __LINE__)

void _check_cuda(cudaError_t err, string filename, int line) {
  if(err != cudaSuccess) {
    throw std::runtime_error(
      "cuda no success at " + filename + ":" + write_with_ss(line));
  }
}
#define check_cuda(call) _check_cuda(call, __FILE__, __LINE__)

struct env_t {
  cublasHandle_t handle;

  env_t() {
    check_cublas(cublasCreate(&handle));
  }

  ~env_t() {
    check_cublas(cublasDestroy(handle));
  }
};

// col major at ij,jk->ik
// A: mk    ij
// B: kn    jk
// C: mn    ik
void matmul(
  env_t& env,
  cudaStream_t stream,
  uint64_t ni,
  uint64_t nj,
  uint64_t nk,
  float const* lhs,
  float const* rhs,
  float*       out)
{
  static float const alpha = 1.0f;
  static float const beta  = 0.0f;

  check_cublas(cublasSetStream(env.handle, stream));

  check_cublas(cublasSgemm(
    env.handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    ni,nk,nj,
    &alpha,
    lhs,ni,
    rhs,nj,
    &beta,
    out,ni));
}

// THE DESIGN:
// Test 1: Launch N matmuls on a single stream and time how long it takes
// Test 2: Launch N matmuls, where each matmul launches the next matmul
// Report total flops

struct data_env_t {
  data_env_t(uint64_t ni, uint64_t nj, uint64_t nk)
    : ni(ni), nj(nj), nk(nk)
  {
    check_cuda(cudaMalloc(&_lhs, sizeof(float)*ni*nj));
    check_cuda(cudaMalloc(&_rhs, sizeof(float)*nj*nk));
    check_cuda(cudaMalloc(&_out, sizeof(float)*ni*nk));

    fill(NULL, ni*nj, 1.0, lhs());
    fill(NULL, nj*nk, 1.0, rhs());

    check_cuda(cudaDeviceSynchronize());
  }

  ~data_env_t() {
    check_cuda(cudaFree(_lhs));
    check_cuda(cudaFree(_rhs));
    check_cuda(cudaFree(_out));
  }

  float* lhs() { return reinterpret_cast<float*>(_lhs); }
  float* rhs() { return reinterpret_cast<float*>(_rhs); }
  float* out() { return reinterpret_cast<float*>(_out); }

  vector<float> read_lhs() const { return _read(ni*nj, _lhs); }
  vector<float> read_rhs() const { return _read(nj*nk, _rhs); }
  vector<float> read_out() const { return _read(ni*nk, _out); }

  vector<float> _read(uint64_t nelem, void const* data) const {
    vector<float> ret(nelem);
    check_cuda(cudaMemcpy(ret.data(), data, sizeof(float)*nelem, cudaMemcpyDefault));
    return ret;
  }

  uint64_t const ni;
  uint64_t const nj;
  uint64_t const nk;

  void* _lhs;
  void* _rhs;
  void* _out;
};

double flops(uint64_t ni, uint64_t nj, uint64_t nk, int nmm, float msec)
{
  double f = 1.0*(ni*nj*nk*uint64_t(nmm));
  double ret = f / double(msec);
  ret *= 1000.0;
  return ret;
}

void matmuls_on_stream(
  env_t& env,
  uint64_t ni,
  uint64_t nj,
  uint64_t nk,
  int nmm,
  int nrep)
{
  data_env_t denv(ni,nj,nk);

  cudaStream_t stream;
  check_cuda(cudaStreamCreate(&stream));

  for(int rep = 0; rep != nrep; ++rep) {
    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));

    check_cuda(cudaEventRecord(start, stream));
    for(int i = 0; i != nmm; ++i) {
      matmul(env, stream, ni, nj, nk, denv.lhs(), denv.rhs(), denv.out());
    }

    check_cuda(cudaEventRecord(stop, stream));
    check_cuda(cudaEventSynchronize(stop));

    float msec = 0.0f;
    check_cuda(cudaEventElapsedTime(&msec, start, stop));

    DOUT(flops(ni,nj,nk,nmm,msec));
  }

  check_cuda(cudaStreamDestroy(stream));
}

struct event_loop_t {
  event_loop_t(cudaStream_t s, env_t& e, data_env_t& d)
    : stream(s), env(e), data(d)
  {}

  void run(int n) {
    while(n != 0) {
      launch();
      std::unique_lock lk(m_notify);
      cv_notify.wait(lk);
      n -= 1;
    }
  }

  void launch() {
    matmul(
      env, stream,
      data.ni, data.nj, data.nk,
      data.lhs(), data.rhs(), data.out());

    check_cuda(cudaStreamAddCallback(
      stream,
      [](cudaStream_t stream, cudaError_t status, void* user_data) {
        event_loop_t* self = reinterpret_cast<event_loop_t*>(user_data);
        self->callback();
      },
      reinterpret_cast<void*>(this),
      0));
  };

  void callback() {
    {
      std::unique_lock lk(m_notify);
      // modify the shared state here (there isn't any)
    }

    cv_notify.notify_one();
  }

  cudaStream_t stream;
  env_t& env;
  data_env_t& data;

  std::mutex m_notify;
  std::condition_variable cv_notify;
};

void matmuls_on_callback(
  env_t& env,
  uint64_t ni,
  uint64_t nj,
  uint64_t nk,
  int nmm,
  int nrep)
{
  data_env_t denv(ni,nj,nk);

  cudaStream_t stream;
  check_cuda(cudaStreamCreate(&stream));

  for(int rep = 0; rep != nrep; ++rep) {
    event_loop_t looper(stream, env, denv);

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));


    check_cuda(cudaEventRecord(start, stream));
    looper.run(nmm);

    check_cuda(cudaEventRecord(stop, stream));
    check_cuda(cudaEventSynchronize(stop));

    float msec = 0.0f;
    check_cuda(cudaEventElapsedTime(&msec, start, stop));

    DOUT(flops(ni,nj,nk,nmm,msec));
  }

  check_cuda(cudaStreamDestroy(stream));
}

void matmuls_on_cudagraph(
  env_t& env,
  uint64_t ni,
  uint64_t nj,
  uint64_t nk,
  int nmm,
  int nrep)
{
  data_env_t denv(ni,nj,nk);

  cudaStream_t stream;
  check_cuda(cudaStreamCreate(&stream));

  cudaGraph_t graph;
  cudaGraphExec_t instance;

  check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for(int i = 0; i != nmm; ++i) {
    matmul(env, stream, ni, nj, nk, denv.lhs(), denv.rhs(), denv.out());
  }
  check_cuda(cudaStreamEndCapture(stream, &graph));
  check_cuda(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

  for(int i = 0; i != nrep; ++i) {
    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));

    check_cuda(cudaEventRecord(start, stream));
    check_cuda(cudaGraphLaunch(instance, stream));
    check_cuda(cudaEventRecord(stop, stream));

    check_cuda(cudaEventSynchronize(stop));

    float msec = 0.0f;
    check_cuda(cudaEventElapsedTime(&msec, start, stop));

    DOUT(flops(ni,nj,nk,nmm,msec));
  }

  // TODO: cleanup graph or instance? who knows..
  check_cuda(cudaStreamDestroy(stream));
}

// Args: ni,nj,nk,nmatmul
int main(int argc, char** argv) {
  if(argc != 6) {
    throw std::runtime_error("invalid number of args.");
  }

  uint64_t ni = parse_with_ss<uint64_t>(argv[1]);
  uint64_t nj = parse_with_ss<uint64_t>(argv[2]);
  uint64_t nk = parse_with_ss<uint64_t>(argv[3]);
  int nmm     = parse_with_ss<int     >(argv[4]);
  int nrep    = parse_with_ss<int     >(argv[5]);

  env_t env;

  DOUT("matmuls on stream");
  matmuls_on_stream(env, ni, nj, nk, nmm, nrep);

  DOUT("matmuls on callback");
  matmuls_on_callback(env, ni, nj, nk, nmm, nrep);

  DOUT("matmuls on cudagraph");
  matmuls_on_cudagraph(env, ni, nj, nk, nmm, nrep);
}
