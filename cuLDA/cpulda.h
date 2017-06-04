#ifndef _CPULDA_
#define _CPULDA_
#include <vector>
#include <tuple>
#include "documents.h"

class LDA {
public:
  LDA() {}
  ~LDA() {}

  virtual void Init(Docs& docs, int K, double alpha = 0.1, double beta = 0.1);
  virtual void Estimate(int max_iter) = 0;
  void InferInit(Doc & doc);
  void Inference(int max_iter, std::vector<double>& topics);
  double InferLikelihood();
  double Likelihood();
  virtual void Release() {}
  void SaveModel(const std::string &path);
  void LoadModel(const std::string &path);

protected:
  void UpdateParam();

protected:
  int K, M, V;
  std::vector<std::vector<int>> nmk_;
  std::vector<int> nm_;
  std::vector<std::vector<int>> nkt_;
  std::vector<int> nk_;
  std::vector<std::vector<int>> z_;
  std::vector<std::vector<double>> theta_;
  std::vector<std::vector<double>> phi_;
  double alpha;
  double beta;
  Docs* docs_;
  std::map<int, std::string> id2word_;
  std::map<std::string, int> word2id_;

  std::vector<int> test_nk_;
  std::vector<int> test_z_;
  Doc* test_doc_;
};

class GibbsLDA : public LDA {
public:
  void Init(Docs& docs, int K, double alpha = 0.1, double beta = 0.1);
  virtual void Estimate(int max_iter);
};

class AliasLDA : public LDA {
public:
  void Init(Docs& docs, int K, double alpha = 0.1, double beta = 0.1);
  virtual void Estimate(int max_iter);
protected:
  void GenerateAlias(int w);
  int SampleAlias(int w);
  std::vector<std::vector<std::tuple<int, int, double>>> qw_alias_;
  std::vector<int> qnum_;
  std::vector<std::vector<double>> qw_;
  std::vector<double> Qw_;
  std::vector<std::set<int>> doc_topic_;
};

class LightLDA : public LDA {
public:
  void Init(Docs& docs, int K, double alpha = 0.1, double beta = 0.1);
  virtual void Estimate(int max_iter);
protected:
  void GenerateAlias(int w);
  int SampleAlias(int w);
  std::vector<std::vector<std::tuple<int, int, double>>> qw_alias_;
  std::vector<int> qnum_;
  std::vector<std::vector<double>> qw_;
};




#endif // !_LDA_