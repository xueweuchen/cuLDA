#include "cpulda.h"
#include "util.h"


void LDA::Init(Docs& docs, int K, double alpha, double beta) {
  std::srand(std::time(NULL));
  this->K = K;
  this->alpha = alpha;
  this->beta = beta;
  this->docs_ = &docs;
  auto doc_list_ = docs.GetDoclist();
  word2id_ = docs.GetWordToId();
  id2word_ = docs.GetIdToWord();
  this->M = doc_list_.size();
  this->V = word2id_.size();

  for (int m = 0; m < M; ++m) {
    nmk_.push_back(std::vector<int>(K, 0));
    nm_.push_back(0);
    theta_.push_back(std::vector<double>(K, 0.0));
  }
  for (int k = 0; k < K; ++k) {
    nkt_.push_back(std::vector<int>(V, 0));
    nk_.push_back(0);
    phi_.push_back(std::vector<double>(V, 0.0));
  }
  for (int m = 0; m < M; ++m) {
    auto wlist = doc_list_[m].GetWords();
    int wnum = wlist.size();
    z_.push_back(std::vector<int>(wnum, 0));
    for (int w = 0; w < wnum; w++) {
      int k = std::rand() % K;
      z_[m][w] = k;
      nmk_[m][k] += 1;
      nm_[m] += 1;
      nkt_[k][word2id_[wlist[w]]] += 1;
      nk_[k] += 1;
    }
  }
}


void LDA::InferInit(Doc &doc) {
  std::srand(std::time(NULL));
  test_nk_.clear();
  test_z_.clear();
  this->test_doc_ = &doc;
  auto wlist = doc.GetWords();
  int wnum = wlist.size();
  for (int k = 0; k < K; ++k) {
    test_nk_.push_back(0);
  }
  for (int w = 0; w < wnum; ++w) {
    test_z_.push_back(0);
  }
  for (int w = 0; w < wnum; ++w) {
    int k = std::rand() % K;
    test_nk_[k] += 1;
    test_z_[w] = k;
  }
}

void LDA::Inference(int max_iter, std::vector<double>& topics) {
  std::srand(std::time(NULL));
  std::vector<double> p(K, 0);
  auto wlist = test_doc_->GetWords();
  int wnum = wlist.size();

  for (int iter = 0; iter < max_iter; ++iter) {
    for (int w = 0; w < wnum; ++w) {
      int wid = word2id_[wlist[w]];
      int topic = test_z_[w];
      test_nk_[topic] -= 1;

      double prob = 0.0;
      for (int k = 0; k < K; ++k) {
        prob += (test_nk_[k] + alpha) * phi_[k][wid];
        p[k] = prob;
      }
      double r = std::rand() * 1.0 / (RAND_MAX + 1) * prob;
      int new_topic = std::lower_bound(p.begin(), p.end(), r) - p.begin();

      test_z_[w] = new_topic;
      test_nk_[new_topic] += 1;
    }
    double llh = InferLikelihood();
    std::cout << "The " << iter << "-th iteration finished, llh = " << llh << "."
      << std::endl;
  }

  topics.clear();
  int norm = std::accumulate(test_nk_.begin(), test_nk_.end(), 0);
  for (int k = 0; k < K; ++k) {
    topics.push_back((test_nk_[k] + alpha) / (norm + K * alpha));
  }
}

double LDA::InferLikelihood() {
  double logllh = 0.0;
  int word_num = 0;
  auto wlist = test_doc_->GetWords();

  int test_nm = std::accumulate(test_nk_.begin(), test_nk_.end(), 0);
  for (int w = 0; w < wlist.size(); ++w) {
    double prob = 0.0;
    int wid = word2id_[wlist[w]];
    for (int k = 0; k < K; ++k) {
      prob += (test_nk_[k] + alpha) / (test_nm + K * alpha) * phi_[k][wid];
    }
    logllh += std::log(prob);
    word_num++;
  }
  return logllh / word_num;
}

double LDA::Likelihood() {
  double logllh = 0.0;
  int word_num = 0;
  auto doc_list_ = docs_->GetDoclist();
  for (int m = 0; m < M; ++m) {
    auto wlist = doc_list_[m].GetWords();
    for (int w = 0; w < wlist.size(); ++w) {
      double prob = 0.0;
      int wid = word2id_[wlist[w]];
      for (int k = 0; k < K; ++k) {
        prob += (nmk_[m][k] + alpha) / (nm_[m] + K * alpha) *
          (nkt_[k][wid] + beta) / (nk_[k] + V * beta);
      }
      logllh += std::log(prob);
      word_num++;
    }
  }
  return logllh / word_num;
}

void LDA::SaveModel(const std::string & path) {
  std::ofstream fout(path);
  fout << alpha << ' ' << beta << std::endl;
  fout << M << ' ' << K << ' ' << V << std::endl;
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      fout << theta_[m][k] << ' ';
  for (int k = 0; k < K; ++k)
    for (int t = 0; t < V; ++t)
      fout << phi_[k][t] << ' ';
  for (auto wi : word2id_)
    fout << wi.first << ' ' << wi.second << ' ';
}

void LDA::LoadModel(const std::string & path) {
  std::ifstream fin(path);
  fin >> alpha >> beta;
  fin >> M >> K >> V;
  theta_ = std::vector<std::vector<double>>(M, std::vector<double>(K, 0));
  phi_ = std::vector<std::vector<double>>(K, std::vector<double>(V, 0));
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      fin >> theta_[m][k];
  for (int k = 0; k < K; ++k)
    for (int t = 0; t < V; ++t)
      fin >> phi_[k][t];
  std::string str;
  int index;
  for (int t = 0; t < V; ++t) {
    fin >> str >> index;
    word2id_[str] = index;
    id2word_[index] = str;
  }
}

void LDA::UpdateParam() {
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      theta_[m][k] = (nmk_[m][k] + alpha) / (nm_[m] + K * alpha);
  for (int k = 0; k < K; ++k)
    for (int t = 0; t < V; ++t)
      phi_[k][t] = (nkt_[k][t] + beta) / (nk_[k] + V * beta);
}

void GibbsLDA::Init(Docs & docs, int K, double alpha, double beta) {
  LDA::Init(docs, K, alpha, beta);
}

void GibbsLDA::Estimate(int max_iter) {
  std::srand(std::time(NULL));
  std::vector<double> p(K, 0);
  auto doc_list_ = docs_->GetDoclist();

  for (int iter = 0; iter < max_iter; ++iter) {
    for (int m = 0; m < M; ++m) {
      auto wlist = doc_list_[m].GetWords();
      for (int w = 0; w < wlist.size(); ++w) {
        int wid = word2id_[wlist[w]];
        int topic = z_[m][w];
        nmk_[m][topic] -= 1;
        nm_[m] -= 1;
        nkt_[topic][wid] -= 1;
        nk_[topic] -= 1;

        double prob = 0.0;
        for (int k = 0; k < K; ++k) {
          prob += (nmk_[m][k] + alpha) / (nm_[m] + K * alpha) *
            (nkt_[k][wid] + beta) / (nk_[k] + V * beta);
          p[k] = prob;
        }
        double r = std::rand() * 1.0 / (RAND_MAX + 1) * prob;
        int new_topic = std::lower_bound(p.begin(), p.end(), r) - p.begin();

        z_[m][w] = new_topic;
        nmk_[m][new_topic] += 1;
        nm_[m] += 1;
        nkt_[new_topic][wid] += 1;
        nk_[new_topic] += 1;
      }
    }
    double llh = Likelihood();
    std::cout << "The " << iter << "-th iteration finished, llh = " << llh << "."
      << std::endl;
  }
  this->UpdateParam();
}

void AliasLDA::Init(Docs & docs, int K, double alpha, double beta) {
  LDA::Init(docs, K, alpha, beta);
  for (int t = 0; t < V; ++t) {
    Qw_.push_back(0.0);
    qnum_.push_back(0);
    qw_.push_back(std::vector<double>(K, 0.0));
    qw_alias_.push_back(
      std::vector<std::tuple<int, int, double>>(K, std::make_tuple(0, 0, 0.0)));
    GenerateAlias(t);
  }
  for (int m = 0; m < M; ++m) {
    doc_topic_.push_back(std::set<int>());
    for (int k = 0; k < K; ++k)
      if (nmk_[m][k] > 0)
        doc_topic_[m].insert(k);
  }
  std::cout << "Finish initialize AliasLDA!" << std::endl;
}

void AliasLDA::Estimate(int max_iter) {
  std::srand(time(NULL));
  std::vector<double> p(K, 0);
  std::vector<int> t(K, 0);
  auto doc_list_ = docs_->GetDoclist();

  for (int iter = 0; iter < max_iter; ++iter) {
    for (int m = 0; m < M; ++m) {
      auto wlist = doc_list_[m].GetWords();
      for (int w = 0; w < wlist.size(); ++w) {
        int wid = word2id_[wlist[w]];
        int topic = z_[m][w];
        nmk_[m][topic] -= 1;
        nkt_[topic][wid] -= 1;
        nk_[topic] -= 1;
        if (nmk_[m][topic] == 0) {
          doc_topic_[m].erase(doc_topic_[m].find(topic));
        }

        int new_topic = topic;
        int tnum = 0;
        double Pdw = 0.0;
        //ignore topic with nmk = 0
        for (int k : doc_topic_[m]) {
          t[tnum] = k;
          Pdw += nmk_[m][k] * (nkt_[k][wid] + beta) / (nk_[k] + V * beta);
          p[tnum++] = Pdw;
        }
        double ratio = Pdw / (Pdw + Qw_[wid]);
        for (int sample = 0; sample < 2; ++sample) {
          double r = rand() * 1.0 / (RAND_MAX + 1.0);
          if (r < ratio) {
            double r2 = rand() * 1.0 / (RAND_MAX + 1.0) * Pdw;
            int tid = std::lower_bound(p.begin(), p.begin() + tnum, r2) - p.begin();
            new_topic = t[tid];
          } else {
            qnum_[wid]++;
            new_topic = SampleAlias(wid);
          }
          if (topic != new_topic) {
            double tmp_old = (nkt_[topic][wid] + beta) / (nk_[topic] + V * beta);
            double tmp_new = (nkt_[new_topic][wid] + beta) / (nk_[new_topic] + V * beta);
            double accept = tmp_new / tmp_old * (nmk_[m][new_topic] + alpha) / (nmk_[m][topic] + alpha) *
              (nmk_[m][topic] * tmp_old + Qw_[wid] * qw_[wid][topic]) /
              (nmk_[m][new_topic] * tmp_new + Qw_[wid] * qw_[wid][new_topic]);
            double r = rand() * 1.0 / (RAND_MAX + 1.0);
            if (r > accept)
              new_topic = topic;
          }
        }
        z_[m][w] = new_topic;
        nmk_[m][new_topic] += 1;
        nkt_[new_topic][wid] += 1;
        nk_[new_topic] += 1;
        doc_topic_[m].insert(new_topic);
        if (qnum_[wid] > K / 2) {
          GenerateAlias(wid);
        }
      }
    }
    double llh = Likelihood();
    std::cout << "The " << iter << "-th iteration finished, llh = " << llh << "."
      << std::endl;
  }
  this->UpdateParam();
}

void AliasLDA::GenerateAlias(int w) {
  qnum_[w] = 0;
  auto &A = qw_alias_[w];
  A.clear();
  auto &q = qw_[w];
  Qw_[w] = 0;
  for (int k = 0; k < K; ++k) {
    q[k] = alpha * (nkt_[k][w] + beta) / (nk_[k] + V * beta);
    Qw_[w] += q[k];
  }
  for (int k = 0; k < K; ++k) {
    q[k] /= Qw_[w];
  }
  util::GenerateAlias(A, q, K);
}

int AliasLDA::SampleAlias(int w) {
  return util::SampleAlias(qw_alias_[w], K);
}

void LightLDA::Init(Docs & docs, int K, double alpha, double beta) {
  LDA::Init(docs, K, alpha, beta);
  for (int t = 0; t < V; ++t) {
    qnum_.push_back(0);
    qw_.push_back(std::vector<double>(K, 0.0));
    qw_alias_.push_back(
      std::vector<std::tuple<int, int, double>>(K, std::make_tuple(0, 0, 0.0)));
    GenerateAlias(t);
  }
  std::cout << "Finish initialize LightLDA!" << std::endl;
}

void LightLDA::Estimate(int max_iter) {
  std::srand(time(NULL));
  auto doc_list_ = docs_->GetDoclist();

  for (int iter = 0; iter < max_iter; ++iter) {
    for (int m = 0; m < M; ++m) {
      auto wlist = doc_list_[m].GetWords();
      int wnum = wlist.size();
      for (int w = 0; w < wnum; ++w) {
        int wid = word2id_[wlist[w]];
        int topic = z_[m][w];
        nmk_[m][topic] -= 1;
        nkt_[topic][wid] -= 1;
        nk_[topic] -= 1;

        int new_topic = topic;
        for (int sample = 0; sample < 2; ++sample) {
          if (rand() % 2 == 0) {
            double r = rand() * 1.0 / (RAND_MAX + 1.0) * (wnum + alpha * K);
            if (r <= wnum) {
              int pos = rand() % wnum;
              new_topic = z_[m][pos];
            } else {
              new_topic = rand() % K;
            }

            if (new_topic != topic) {
              double tmp_old = (nkt_[topic][wid] + beta) * (nmk_[m][topic] + alpha) /
                (nk_[topic] + V * beta);
              double tmp_new = (nkt_[new_topic][wid] + beta) *(nmk_[m][new_topic] + alpha) /
                (nk_[new_topic] + V * beta);
              double accept = tmp_new / tmp_old * (nmk_[m][topic] + 1 + alpha) /
                (nmk_[m][new_topic] + alpha);
              double r = rand() * 1.0 / (RAND_MAX + 1.0);
              if (r > accept)
                new_topic = topic;
            }
          } else {
            qnum_[wid]++;
            new_topic = SampleAlias(wid);

            if (topic != new_topic) {
              double tmp_old = (nkt_[topic][wid] + beta) * (nmk_[m][topic] + alpha) /
                (nk_[topic] + V * beta);
              double tmp_new = (nkt_[new_topic][wid] + beta) *(nmk_[m][new_topic] + alpha) /
                (nk_[new_topic] + V * beta);
              double accept = tmp_new / tmp_old * qw_[wid][topic] / qw_[wid][new_topic];
              double r = rand() * 1.0 / (RAND_MAX + 1.0);
              if (r > accept)
                new_topic = topic;
            }
          }
        }
        z_[m][w] = new_topic;
        nmk_[m][new_topic] += 1;
        nkt_[new_topic][wid] += 1;
        nk_[new_topic] += 1;
        if (qnum_[wid] > K / 2) {
          GenerateAlias(wid);
        }
      }
    }
    double llh = Likelihood();
    std::cout << "The " << iter << "-th iteration finished, llh = " << llh << "."
      << std::endl;
  }
  this->UpdateParam();
}

void LightLDA::GenerateAlias(int w) {
  qnum_[w] = 0;
  auto &A = qw_alias_[w];
  A.clear();
  auto &q = qw_[w];
  double prob_sum = 0.0;
  for (int k = 0; k < K; ++k) {
    q[k] = (nkt_[k][w] + beta) / (nk_[k] + V * beta);
    prob_sum += q[k];
  }
  for (int k = 0; k < K; ++k) {
    q[k] /= prob_sum;
  }
  util::GenerateAlias(A, q, K);
}

int LightLDA::SampleAlias(int w) {
  return util::SampleAlias(qw_alias_[w], K);
}
