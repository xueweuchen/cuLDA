#ifndef _GPULDA_
#define _GPULDA_

#include <vector>
#include <tuple>
#include "documents.h"
#include "cpulda.h"


class MMLDA : public LDA {
private:
  int *nmk_d_;
  int *nm_d_;
  int *nkt_d_;
  int *nk_d_;
  int *dw_d_;
  int *wnums_d_;
  int *start_d_;
  float *theta_d_;
  float *phi_d_;
  int N;
public:
  void Init(Docs& docs, int K, double alpha = 0.1, double beta = 0.1);
  virtual void Estimate(int max_iter);
  virtual void UpdateParamGpu();
  virtual float LikelihoodGpu();
  void Release();
};



#endif // _GPULDA_
