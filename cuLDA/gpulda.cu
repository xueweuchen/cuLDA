#include "gpulda.h"
#include "cuda_util.h"
#include <iostream>


void MMLDA::Init(Docs &docs, int K, double alpha, double beta) {
  LDA::Init(docs, K, alpha, beta);
  HANDLE_ERROR(cudaSetDevice(0));
  HANDLE_ERROR(cudaMalloc((void**)&nmk_d_, M * K * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&nm_d_, M * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&nkt_d_, V * K * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&nk_d_, K * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&phi_d_, V * K * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&theta_d_, M * K * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(nm_d_, &nm_[0], M * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(nk_d_, &nk_[0], K * sizeof(int), cudaMemcpyHostToDevice));

  for (int m = 0; m < M; ++m) {
    HANDLE_ERROR(cudaMemcpy(&nmk_d_[m*K], &nmk_[m][0], K * sizeof(int), cudaMemcpyHostToDevice));
  }

  for (int k = 0; k < K; ++k) {
    HANDLE_ERROR(cudaMemcpy(&nkt_d_[k*V], &nkt_[k][0], V * sizeof(int), cudaMemcpyHostToDevice));
  }
  HANDLE_ERROR(cudaMemset(phi_d_, 0, V * K * sizeof(int)));
  HANDLE_ERROR(cudaMemset(theta_d_, 0, M * K * sizeof(int)));

  auto doc_list_ = this->docs_->GetDoclist();
  std::vector<int> dw_h;
  std::vector<int> wnums_h(M, 0);
  std::vector<int> start_h(M, 0);
  this->N = 0;
  for (int m = 0; m < M; ++m) {
    auto wlist = doc_list_[m].GetWords();
    int wnum = wlist.size();
    this->N += wnum;
    wnums_h[m] = wnum;
    start_h[m] = dw_h.size();
    for (int w = 0; w < wnum; ++w) {
      dw_h.push_back(word2id_[wlist[w]]);
    }
  }
  HANDLE_ERROR(cudaMalloc((void**)&wnums_d_, M * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(wnums_d_, &wnums_h[0], M * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc((void**)&start_d_, M * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(start_d_, &start_h[0], M * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc((void**)&dw_d_, dw_h.size() * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(dw_d_, &dw_h[0], dw_h.size() * sizeof(int), cudaMemcpyHostToDevice));

  this->UpdateParamGpu();
  std::cout << "Finish initialize Mean Mode LDA!" << std::endl;
}

__global__ void ComputePhi(float *phi_, int *nkt_, int *nk_, int K, int V, float beta) {
  CUDA_KERNEL_LOOP(i, K * V) {
    int k = i / V;
    int t = i % V;
    phi_[k * V + t] = (nkt_[k * V + t] + beta) / (nk_[k] + V * beta);
  }
}

__global__ void ComputeTheta(float *theta_, int *nmk_, int *nm_, int K, int M, float alpha) {
  CUDA_KERNEL_LOOP(i, K * M) {
    int m = i / K;
    int k = i % K;
    theta_[m * K + k] = (nmk_[m * K + k] + alpha) / (nm_[m] + K * alpha);
  }
}

void MMLDA::UpdateParamGpu() {
  ComputePhi<<<GET_BLOCKS((K * V)), CUDA_NUM_THREADS>>>(phi_d_, nkt_d_, nk_d_, K, V, beta);
  ComputeTheta<<<GET_BLOCKS((K * M)), CUDA_NUM_THREADS>>>(theta_d_, nmk_d_, nm_d_, K, M, alpha);
  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());
}


__global__ void SampleDoc(int *dw, int *wnums, int *start, int *nkt_, int *nk_, int *nmk_, 
    int *nm_, int K, int M, int V, float *phi_, float *theta_) {
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
        sum += phi_[j * V + wid] * theta_[m * K + j];
        p[j] = sum;
      }

      float stop = curand_uniform(&s) * sum;
      int j = 0;
      for (; j < K; ++j) {
        if (stop < p[j])
          break;
      }

      atomicAdd(&nkt_[j * V + wid], 1);
      atomicAdd(&nk_[j], 1);
      nmk_[m * K + j]++;
      nm_[m]++;
    }
    delete p;
  }
}

void MMLDA::Estimate(int max_iter) {
  for (int iter = 0; iter < max_iter; ++iter) {
    HANDLE_ERROR(cudaMemset(nmk_d_, 0, M * K * sizeof(int)));
    HANDLE_ERROR(cudaMemset(nm_d_, 0, M * sizeof(int)));
    HANDLE_ERROR(cudaMemset(nkt_d_, 0, V * K * sizeof(int)));
    HANDLE_ERROR(cudaMemset(nk_d_, 0, K * sizeof(int)));
    SampleDoc<<<GET_BLOCKS(M), CUDA_NUM_THREADS >>> (dw_d_, wnums_d_, start_d_, nkt_d_, 
      nk_d_, nmk_d_, nm_d_, K, M, V, phi_d_, theta_d_);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    this->UpdateParamGpu();
    float llh = LikelihoodGpu();
    std::cout << "The " << iter << "-th iteration finished, llh = " << llh << "."
      << std::endl;
  }
}

__global__ void LlhKernel(float *result, int *dw, int *wnums, int *start, int K, 
    int M, int V, float *phi_, float *theta_) {
  CUDA_KERNEL_LOOP(m, M) {
    int wnum = wnums[m];
    result[m] = 0.0;
    for (int windex = 0; windex < wnum; ++windex) {
      int wid = dw[start[m] + windex];
      float sum = 0.0;
      for (int j = 0; j < K; ++j) {
        sum += phi_[j * V + wid] * theta_[m * K + j];
      }
      result[m] += log(sum);
    }
  }
}

float MMLDA::LikelihoodGpu() {
  float *result_d;
  HANDLE_ERROR(cudaMalloc((void**)&result_d, M * sizeof(float)));
  LlhKernel <<<GET_BLOCKS(M), CUDA_NUM_THREADS>>> (result_d, dw_d_, wnums_d_, start_d_, K, M, V, phi_d_, theta_d_);
  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());
  thrust::device_ptr<float> dptr = thrust::device_pointer_cast(result_d);
  float result = thrust::reduce(dptr, dptr+M, (float)0, thrust::plus<float>());
  HANDLE_ERROR(cudaFree(result_d));
  return result / N;
}

void MMLDA::Release() {
  HANDLE_ERROR(cudaFree(nmk_d_));
  HANDLE_ERROR(cudaFree(nm_d_));
  HANDLE_ERROR(cudaFree(nkt_d_));
  HANDLE_ERROR(cudaFree(nk_d_));
  HANDLE_ERROR(cudaFree(dw_d_));
  HANDLE_ERROR(cudaFree(wnums_d_));
  HANDLE_ERROR(cudaFree(start_d_));
  HANDLE_ERROR(cudaFree(theta_d_));
  HANDLE_ERROR(cudaFree(phi_d_));
}

