#ifndef _CPULDA_
#define _CPULDA_
#include <vector>
#include <tuple>
#include "documents.h"

class LDA {
public:
  LDA() {}
  ~LDA() {}

  virtual void init(Docs& docs, int K, double alpha = 0.1, double beta = 0.1);
  virtual void estimate(int max_iter) = 0;
  void infer_init(Doc & doc);
  void inference(int max_iter, std::vector<double>& topics);
  double infer_likelihood();
  double likelihood();
  virtual void release() {}
  void save_model(const std::string &path);
  void load_model(const std::string &path);

protected:
  void update_param();

protected:
  int K, M, V;
  std::vector<std::vector<int>> nmk;
  std::vector<int> nm;
  std::vector<std::vector<int>> nkt;
  std::vector<int> nk;
  std::vector<std::vector<int>> z;
  std::vector<std::vector<double>> theta;
  std::vector<std::vector<double>> phi;
  double alpha;
  double beta;
  Docs* docs;
  std::map<int, std::string> id2word;
  std::map<std::string, int> word2id;

  std::vector<int> test_nk;
  std::vector<int> test_z;
  Doc* test_doc;
};

class GibbsLDA : public LDA {
public:
  void init(Docs& docs, int K, double alpha = 0.1, double beta = 0.1);
  virtual void estimate(int max_iter);
};

class AliasLDA : public LDA {
public:
  void init(Docs& docs, int K, double alpha = 0.1, double beta = 0.1);
  virtual void estimate(int max_iter);
protected:
  void generate_alias(int w);
  int sample_alias(int w);
  std::vector<std::vector<std::tuple<int, int, double>>> qw_alias;
  std::vector<int> qnum;
  std::vector<std::vector<double>> qw;
  std::vector<double> Qw;
  std::vector<std::set<int>> doc_topic;
};

class LightLDA : public LDA {
public:
  void init(Docs& docs, int K, double alpha = 0.1, double beta = 0.1);
  virtual void estimate(int max_iter);
protected:
  void generate_alias(int w);
  int sample_alias(int w);
  std::vector<std::vector<std::tuple<int, int, double>>> qw_alias;
  std::vector<int> qnum;
  std::vector<std::vector<double>> qw;
};




#endif // !_LDA_