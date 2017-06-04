#include "gpulda.h"
#include "cuda_util.h"
#include <iostream>


void MMLDA::Init(Docs &docs, int K, double alpha, double beta) {
  LDA::Init(docs, K, alpha, beta);
  HANDLE_ERROR(cudaSetDevice(0));
  HANDLE_ERROR(cudaMalloc((void**)&nmk_d, M * K * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&nm_d, M * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&nkt_d, V * K * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&nk_d, K * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&phi_d, V * K * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&theta_d, M * K * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(nm_d, &nm[0], M * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(nk_d, &nk[0], K * sizeof(int), cudaMemcpyHostToDevice));

  for (int m = 0; m < M; ++m) {
    HANDLE_ERROR(cudaMemcpy(&nmk_d[m*K], &nmk[m][0], K * sizeof(int), cudaMemcpyHostToDevice));
  }

  for (int k = 0; k < K; ++k) {
    HANDLE_ERROR(cudaMemcpy(&nkt_d[k*V], &nkt[k][0], V * sizeof(int), cudaMemcpyHostToDevice));
  }
  HANDLE_ERROR(cudaMemset(phi_d, 0, V * K * sizeof(int)));
  HANDLE_ERROR(cudaMemset(theta_d, 0, M * K * sizeof(int)));

  auto doc_list = this->docs->GetDoclist();
  std::vector<int> dw_h;
  std::vector<int> wnums_h(M, 0);
  std::vector<int> start_h(M, 0);
  this->N = 0;
  for (int m = 0; m < M; ++m) {
    auto wlist = doc_list[m].GetWords();
    int wnum = wlist.size();
    this->N += wnum;
    wnums_h[m] = wnum;
    start_h[m] = dw_h.size();
    for (int w = 0; w < wnum; ++w) {
      dw_h.push_back(word2id[wlist[w]]);
    }
  }
  HANDLE_ERROR(cudaMalloc((void**)&wnums_d, M * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(wnums_d, &wnums_h[0], M * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc((void**)&start_d, M * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(start_d, &start_h[0], M * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc((void**)&dw_d, dw_h.size() * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(dw_d, &dw_h[0], dw_h.size() * sizeof(int), cudaMemcpyHostToDevice));

  this->UpdateParamGpu();
  std::cout << "Finish initialize Mean Mode LDA!" << std::endl;
}

__global__ void ComputePhi(float *phi, int *nkt, int *nk, int K, int V, float beta) {
  CUDA_KERNEL_LOOP(i, K * V) {
    int k = i / V;
    int t = i % V;
    phi[k * V + t] = (nkt[k * V + t] + beta) / (nk[k] + V * beta);
  }
}

__global__ void ComputeTheta(float *theta, int *nmk, int *nm, int K, int M, float alpha) {
  CUDA_KERNEL_LOOP(i, K * M) {
    int m = i / K;
    int k = i % K;
    theta[m * K + k] = (nmk[m * K + k] + alpha) / (nm[m] + K * alpha);
  }
}

void MMLDA::UpdateParamGpu() {
  ComputePhi<<<GET_BLOCKS((K * V)), CUDA_NUM_THREADS>>>(phi_d, nkt_d, nk_d, K, V, beta);
  ComputeTheta<<<GET_BLOCKS((K * M)), CUDA_NUM_THREADS>>>(theta_d, nmk_d, nm_d, K, M, alpha);
  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());
}


__global__ void SampleDoc(int *dw, int *wnums, int *start, int *nkt, int *nk, int *nmk, 
    int *nm, int K, int M, int V, float *phi, float *theta) {
  CUDA_KERNEL_LOOP(m, M) {
    unsigned int seed = m;
    curandState s;
    curand_init(seed, 0, 0, &s);
     
    int wnum = wnums[m];
    float *p = new float[K];
    for (int windex = 0; windex < wnum; ++windex) {
      float sum = 0;
      int wid = dw[start[m] + windex];
      for (int j = 0; j < K; ++j) {
        sum += phi[j * V + wid] * theta[m * K + j];
        p[j] = sum;
      }

      float stop = curand_uniform(&s) * sum;
      int j = 0;
      for (; j < K; ++j) {
        if (stop < p[j])
          break;
      }

      atomicAdd(&nkt[j * V + wid], 1);
      atomicAdd(&nk[j], 1);
      nmk[m * K + j]++;
      nm[m]++;
    }
    delete p;
  }
}

void MMLDA::Estimate(int max_iter) {
  for (int iter = 0; iter < max_iter; ++iter) {
    HANDLE_ERROR(cudaMemset(nmk_d, 0, M * K * sizeof(int)));
    HANDLE_ERROR(cudaMemset(nm_d, 0, M * sizeof(int)));
    HANDLE_ERROR(cudaMemset(nkt_d, 0, V * K * sizeof(int)));
    HANDLE_ERROR(cudaMemset(nk_d, 0, K * sizeof(int)));
    SampleDoc<<<GET_BLOCKS(M), CUDA_NUM_THREADS >>> (dw_d, wnums_d, start_d, nkt_d, 
      nk_d, nmk_d, nm_d, K, M, V, phi_d, theta_d);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    this->UpdateParamGpu();
    float llh = LikelihoodGpu();
    std::cout << "The " << iter << "-th iteration finished, llh = " << llh << "."
      << std::endl;
  }
}

__global__ void LlhKernel(float *result, int *dw, int *wnums, int *start, int K, 
    int M, int V, float *phi, float *theta) {
  CUDA_KERNEL_LOOP(m, M) {
    int wnum = wnums[m];
    result[m] = 0.0;
    for (int windex = 0; windex < wnum; ++windex) {
      int wid = dw[start[m] + windex];
      float sum = 0.0;
      for (int j = 0; j < K; ++j) {
        sum += phi[j * V + wid] * theta[m * K + j];
      }
      result[m] += log(sum);
    }
  }
}

float MMLDA::LikelihoodGpu() {
  float *result_d;
  HANDLE_ERROR(cudaMalloc((void**)&result_d, M * sizeof(float)));
  LlhKernel <<<GET_BLOCKS(M), CUDA_NUM_THREADS>>> (result_d, dw_d, wnums_d, start_d, K, M, V, phi_d, theta_d);
  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());
  thrust::device_ptr<float> dptr = thrust::device_pointer_cast(result_d);
  float result = thrust::reduce(dptr, dptr+M, (float)0, thrust::plus<float>());
  HANDLE_ERROR(cudaFree(result_d));
  return result / N;
}

void MMLDA::Release() {
  HANDLE_ERROR(cudaFree(nmk_d));
  HANDLE_ERROR(cudaFree(nm_d));
  HANDLE_ERROR(cudaFree(nkt_d));
  HANDLE_ERROR(cudaFree(nk_d));
  HANDLE_ERROR(cudaFree(dw_d));
  HANDLE_ERROR(cudaFree(wnums_d));
  HANDLE_ERROR(cudaFree(start_d));
  HANDLE_ERROR(cudaFree(theta_d));
  HANDLE_ERROR(cudaFree(phi_d));
}

