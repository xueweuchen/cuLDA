#ifndef _GPULDA_
#define _GPULDA_

#include <vector>
#include <tuple>
#include "documents.h"
#include "cpulda.h"


class MMLDA : public LDA {
private:
  int *nmk_d;
  int *nm_d;
  int *nkt_d;
  int *nk_d;
  int *dw_d;
  int *wnums_d;
  int *start_d;
  float *theta_d;
  float *phi_d;
  int N;
public:
  void Init(Docs& docs, int K, double alpha = 0.1, double beta = 0.1);
  virtual void Estimate(int max_iter);
  virtual void UpdateParamGpu();
  virtual float LikelihoodGpu();
  void Release();
};



#endif // _GPULDA_
