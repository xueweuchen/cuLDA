#include "cpulda.h"
#include "util.h"


void LDA::init(Docs& docs, int K, double alpha, double beta) {
  std::srand(std::time(NULL));
  this->K = K;
  this->alpha = alpha;
  this->beta = beta;
  this->docs = &docs;
  auto doc_list = docs.get_doclist();
  word2id = docs.get_word2id();
  id2word = docs.get_id2word();
  this->M = doc_list.size();
  this->V = word2id.size();

  for (int m = 0; m < M; ++m) {
    nmk.push_back(std::vector<int>(K, 0));
    nm.push_back(0);
    theta.push_back(std::vector<double>(K, 0.0));
  }
  for (int k = 0; k < K; ++k) {
    nkt.push_back(std::vector<int>(V, 0));
    nk.push_back(0);
    phi.push_back(std::vector<double>(V, 0.0));
  }
  for (int m = 0; m < M; ++m) {
    auto wlist = doc_list[m].get_words();
    int wnum = wlist.size();
    z.push_back(std::vector<int>(wnum, 0));
    for (int w = 0; w < wnum; w++) {
      int k = std::rand() % K;
      z[m][w] = k;
      nmk[m][k] += 1;
      nm[m] += 1;
      nkt[k][word2id[wlist[w]]] += 1;
      nk[k] += 1;
    }
  }
}


void LDA::infer_init(Doc &doc) {
  std::srand(std::time(NULL));
  test_nk.clear();
  test_z.clear();
  this->test_doc = &doc;
  auto wlist = doc.get_words();
  int wnum = wlist.size();
  for (int k = 0; k < K; ++k) {
    test_nk.push_back(0);
  }
  for (int w = 0; w < wnum; ++w) {
    test_z.push_back(0);
  }
  for (int w = 0; w < wnum; ++w) {
    int k = std::rand() % K;
    test_nk[k] += 1;
    test_z[w] = k;
  }
}

void LDA::inference(int max_iter, std::vector<double>& topics) {
  std::srand(std::time(NULL));
  std::vector<double> p(K, 0);
  auto wlist = test_doc->get_words();
  int wnum = wlist.size();

  for (int iter = 0; iter < max_iter; ++iter) {
    for (int w = 0; w < wnum; ++w) {
      int wid = word2id[wlist[w]];
      int topic = test_z[w];
      test_nk[topic] -= 1;

      double prob = 0.0;
      for (int k = 0; k < K; ++k) {
        prob += (test_nk[k] + alpha) * phi[k][wid];
        p[k] = prob;
      }
      double r = std::rand() * 1.0 / (RAND_MAX + 1) * prob;
      int new_topic = std::lower_bound(p.begin(), p.end(), r) - p.begin();

      test_z[w] = new_topic;
      test_nk[new_topic] += 1;
    }
    double llh = infer_likelihood();
    std::cout << "The " << iter << "-th iteration finished, llh = " << llh << "."
      << std::endl;
  }

  topics.clear();
  int norm = std::accumulate(test_nk.begin(), test_nk.end(), 0);
  for (int k = 0; k < K; ++k) {
    topics.push_back((test_nk[k] + alpha) / (norm + K * alpha));
  }
}

double LDA::infer_likelihood() {
  double logllh = 0.0;
  int word_num = 0;
  auto wlist = test_doc->get_words();

  int test_nm = std::accumulate(test_nk.begin(), test_nk.end(), 0);
  for (int w = 0; w < wlist.size(); ++w) {
    double prob = 0.0;
    int wid = word2id[wlist[w]];
    for (int k = 0; k < K; ++k) {
      prob += (test_nk[k] + alpha) / (test_nm + K * alpha) * phi[k][wid];
    }
    logllh += std::log(prob);
    word_num++;
  }
  return logllh / word_num;
}

double LDA::likelihood() {
  double logllh = 0.0;
  int word_num = 0;
  auto doc_list = docs->get_doclist();
  for (int m = 0; m < M; ++m) {
    auto wlist = doc_list[m].get_words();
    for (int w = 0; w < wlist.size(); ++w) {
      double prob = 0.0;
      int wid = word2id[wlist[w]];
      for (int k = 0; k < K; ++k) {
        prob += (nmk[m][k] + alpha) / (nm[m] + K * alpha) *
          (nkt[k][wid] + beta) / (nk[k] + V * beta);
      }
      logllh += std::log(prob);
      word_num++;
    }
  }
  return logllh / word_num;
}

void LDA::save_model(const std::string & path) {
  std::ofstream fout(path);
  fout << alpha << ' ' << beta << std::endl;
  fout << M << ' ' << K << ' ' << V << std::endl;
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      fout << theta[m][k] << ' ';
  for (int k = 0; k < K; ++k)
    for (int t = 0; t < V; ++t)
      fout << phi[k][t] << ' ';
  for (auto wi : word2id)
    fout << wi.first << ' ' << wi.second << ' ';
}

void LDA::load_model(const std::string & path) {
  std::ifstream fin(path);
  fin >> alpha >> beta;
  fin >> M >> K >> V;
  theta = std::vector<std::vector<double>>(M, std::vector<double>(K, 0));
  phi = std::vector<std::vector<double>>(K, std::vector<double>(V, 0));
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      fin >> theta[m][k];
  for (int k = 0; k < K; ++k)
    for (int t = 0; t < V; ++t)
      fin >> phi[k][t];
  std::string str;
  int index;
  for (int t = 0; t < V; ++t) {
    fin >> str >> index;
    word2id[str] = index;
    id2word[index] = str;
  }
}

void LDA::update_param() {
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      theta[m][k] = (nmk[m][k] + alpha) / (nm[m] + K * alpha);
  for (int k = 0; k < K; ++k)
    for (int t = 0; t < V; ++t)
      phi[k][t] = (nkt[k][t] + beta) / (nk[k] + V * beta);
}

void GibbsLDA::init(Docs & docs, int K, double alpha, double beta) {
  LDA::init(docs, K, alpha, beta);
}

void GibbsLDA::estimate(int max_iter) {
  std::srand(std::time(NULL));
  std::vector<double> p(K, 0);
  auto doc_list = docs->get_doclist();

  for (int iter = 0; iter < max_iter; ++iter) {
    for (int m = 0; m < M; ++m) {
      auto wlist = doc_list[m].get_words();
      for (int w = 0; w < wlist.size(); ++w) {
        int wid = word2id[wlist[w]];
        int topic = z[m][w];
        nmk[m][topic] -= 1;
        nm[m] -= 1;
        nkt[topic][wid] -= 1;
        nk[topic] -= 1;

        double prob = 0.0;
        for (int k = 0; k < K; ++k) {
          prob += (nmk[m][k] + alpha) / (nm[m] + K * alpha) *
            (nkt[k][wid] + beta) / (nk[k] + V * beta);
          p[k] = prob;
        }
        double r = std::rand() * 1.0 / (RAND_MAX + 1) * prob;
        int new_topic = std::lower_bound(p.begin(), p.end(), r) - p.begin();

        z[m][w] = new_topic;
        nmk[m][new_topic] += 1;
        nm[m] += 1;
        nkt[new_topic][wid] += 1;
        nk[new_topic] += 1;
      }
    }
    double llh = likelihood();
    std::cout << "The " << iter << "-th iteration finished, llh = " << llh << "."
      << std::endl;
  }
  this->update_param();
}

void AliasLDA::init(Docs & docs, int K, double alpha, double beta) {
  LDA::init(docs, K, alpha, beta);
  for (int t = 0; t < V; ++t) {
    Qw.push_back(0.0);
    qnum.push_back(0);
    qw.push_back(std::vector<double>(K, 0.0));
    qw_alias.push_back(
      std::vector<std::tuple<int, int, double>>(K, std::make_tuple(0, 0, 0.0)));
    generate_alias(t);
  }
  for (int m = 0; m < M; ++m) {
    doc_topic.push_back(std::set<int>());
    for (int k = 0; k < K; ++k)
      if (nmk[m][k] > 0)
        doc_topic[m].insert(k);
  }
  std::cout << "Finish initialize AliasLDA!" << std::endl;
}

void AliasLDA::estimate(int max_iter) {
  std::srand(time(NULL));
  std::vector<double> p(K, 0);
  std::vector<int> t(K, 0);
  auto doc_list = docs->get_doclist();

  for (int iter = 0; iter < max_iter; ++iter) {
    for (int m = 0; m < M; ++m) {
      auto wlist = doc_list[m].get_words();
      for (int w = 0; w < wlist.size(); ++w) {
        int wid = word2id[wlist[w]];
        int topic = z[m][w];
        nmk[m][topic] -= 1;
        nkt[topic][wid] -= 1;
        nk[topic] -= 1;
        if (nmk[m][topic] == 0) {
          doc_topic[m].erase(doc_topic[m].find(topic));
        }

        int new_topic = topic;
        int tnum = 0;
        double Pdw = 0.0;
        //ignore topic with nmk = 0
        for (int k : doc_topic[m]) {
          t[tnum] = k;
          Pdw += nmk[m][k] * (nkt[k][wid] + beta) / (nk[k] + V * beta);
          p[tnum++] = Pdw;
        }
        double ratio = Pdw / (Pdw + Qw[wid]);
        for (int sample = 0; sample < 2; ++sample) {
          double r = rand() * 1.0 / (RAND_MAX + 1.0);
          if (r < ratio) {
            double r2 = rand() * 1.0 / (RAND_MAX + 1.0) * Pdw;
            int tid = std::lower_bound(p.begin(), p.begin() + tnum, r2) - p.begin();
            new_topic = t[tid];
          } else {
            qnum[wid]++;
            new_topic = sample_alias(wid);
          }
          if (topic != new_topic) {
            double tmp_old = (nkt[topic][wid] + beta) / (nk[topic] + V * beta);
            double tmp_new = (nkt[new_topic][wid] + beta) / (nk[new_topic] + V * beta);
            double accept = tmp_new / tmp_old * (nmk[m][new_topic] + alpha) / (nmk[m][topic] + alpha) *
              (nmk[m][topic] * tmp_old + Qw[wid] * qw[wid][topic]) /
              (nmk[m][new_topic] * tmp_new + Qw[wid] * qw[wid][new_topic]);
            double r = rand() * 1.0 / (RAND_MAX + 1.0);
            if (r > accept)
              new_topic = topic;
          }
        }
        z[m][w] = new_topic;
        nmk[m][new_topic] += 1;
        nkt[new_topic][wid] += 1;
        nk[new_topic] += 1;
        doc_topic[m].insert(new_topic);
        if (qnum[wid] > K / 2) {
          generate_alias(wid);
        }
      }
    }
    double llh = likelihood();
    std::cout << "The " << iter << "-th iteration finished, llh = " << llh << "."
      << std::endl;
  }
  this->update_param();
}

void AliasLDA::generate_alias(int w) {
  qnum[w] = 0;
  auto &A = qw_alias[w];
  A.clear();
  auto &q = qw[w];
  Qw[w] = 0;
  for (int k = 0; k < K; ++k) {
    q[k] = alpha * (nkt[k][w] + beta) / (nk[k] + V * beta);
    Qw[w] += q[k];
  }
  for (int k = 0; k < K; ++k) {
    q[k] /= Qw[w];
  }
  util::generate_alias(A, q, K);
}

int AliasLDA::sample_alias(int w) {
  return util::sample_alias(qw_alias[w], K);
}

void LightLDA::init(Docs & docs, int K, double alpha, double beta) {
  LDA::init(docs, K, alpha, beta);
  for (int t = 0; t < V; ++t) {
    qnum.push_back(0);
    qw.push_back(std::vector<double>(K, 0.0));
    qw_alias.push_back(
      std::vector<std::tuple<int, int, double>>(K, std::make_tuple(0, 0, 0.0)));
    generate_alias(t);
  }
  std::cout << "Finish initialize LightLDA!" << std::endl;
}

void LightLDA::estimate(int max_iter) {
  std::srand(time(NULL));
  auto doc_list = docs->get_doclist();

  for (int iter = 0; iter < max_iter; ++iter) {
    for (int m = 0; m < M; ++m) {
      auto wlist = doc_list[m].get_words();
      int wnum = wlist.size();
      for (int w = 0; w < wnum; ++w) {
        int wid = word2id[wlist[w]];
        int topic = z[m][w];
        nmk[m][topic] -= 1;
        nkt[topic][wid] -= 1;
        nk[topic] -= 1;

        int new_topic = topic;
        for (int sample = 0; sample < 2; ++sample) {
          if (rand() % 2 == 0) {
            double r = rand() * 1.0 / (RAND_MAX + 1.0) * (wnum + alpha * K);
            if (r <= wnum) {
              int pos = rand() % wnum;
              new_topic = z[m][pos];
            } else {
              new_topic = rand() % K;
            }

            if (new_topic != topic) {
              double tmp_old = (nkt[topic][wid] + beta) * (nmk[m][topic] + alpha) /
                (nk[topic] + V * beta);
              double tmp_new = (nkt[new_topic][wid] + beta) *(nmk[m][new_topic] + alpha) /
                (nk[new_topic] + V * beta);
              double accept = tmp_new / tmp_old * (nmk[m][topic] + 1 + alpha) /
                (nmk[m][new_topic] + alpha);
              double r = rand() * 1.0 / (RAND_MAX + 1.0);
              if (r > accept)
                new_topic = topic;
            }
          } else {
            qnum[wid]++;
            new_topic = sample_alias(wid);

            if (topic != new_topic) {
              double tmp_old = (nkt[topic][wid] + beta) * (nmk[m][topic] + alpha) /
                (nk[topic] + V * beta);
              double tmp_new = (nkt[new_topic][wid] + beta) *(nmk[m][new_topic] + alpha) /
                (nk[new_topic] + V * beta);
              double accept = tmp_new / tmp_old * qw[wid][topic] / qw[wid][new_topic];
              double r = rand() * 1.0 / (RAND_MAX + 1.0);
              if (r > accept)
                new_topic = topic;
            }
          }
        }
        z[m][w] = new_topic;
        nmk[m][new_topic] += 1;
        nkt[new_topic][wid] += 1;
        nk[new_topic] += 1;
        if (qnum[wid] > K / 2) {
          generate_alias(wid);
        }
      }
    }
    double llh = likelihood();
    std::cout << "The " << iter << "-th iteration finished, llh = " << llh << "."
      << std::endl;
  }
  this->update_param();
}

void LightLDA::generate_alias(int w) {
  qnum[w] = 0;
  auto &A = qw_alias[w];
  A.clear();
  auto &q = qw[w];
  double prob_sum = 0.0;
  for (int k = 0; k < K; ++k) {
    q[k] = (nkt[k][w] + beta) / (nk[k] + V * beta);
    prob_sum += q[k];
  }
  for (int k = 0; k < K; ++k) {
    q[k] /= prob_sum;
  }
  util::generate_alias(A, q, K);
}

int LightLDA::sample_alias(int w) {
  return util::sample_alias(qw_alias[w], K);
}
