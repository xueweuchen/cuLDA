#include "util.h"


namespace util {

std::vector<std::string> split(const std::string &text, char sep) {
  std::vector<std::string> tokens;
  std::size_t start = 0, end = 0;
  while ((end = text.find(sep, start)) != std::string::npos) {
    tokens.push_back(text.substr(start, end - start));
    start = end + 1;
  }
  tokens.push_back(text.substr(start));
  return tokens;
}

void generate_alias(std::vector<std::tuple<int, int, double>> &A,
    std::vector<double> &q, int K) {
  std::vector<std::pair<int, double>> L, H;
  L.reserve(K);
  H.reserve(K);

  for (int k = 0; k < K; ++k) {
    if (q[k] <= 1.0 / K) {
      L.push_back(std::make_pair(k, q[k]));
    } else {
      H.push_back(std::make_pair(k, q[k]));
    }
  }
  while (!L.empty() && !H.empty()) {
    std::pair<int, double> lp = L.back();
    std::pair<int, double> hp = H.back();
    L.pop_back();
    H.pop_back();
    A.push_back(std::make_tuple(lp.first, hp.first, lp.second));
    double left = hp.second + lp.second - 1.0 / K;
    if (left > 1.0 / K) {
      H.push_back(std::make_pair(hp.first, left));
    } else {
      L.push_back(std::make_pair(hp.first, left));
    }
  }
  while (!H.empty()) {
    std::pair<int, double> hp = H.back();
    H.pop_back();
    A.push_back(std::make_tuple(hp.first, hp.first, 1.0));
  }
  while (!L.empty()) {
    std::pair<int, double> lp = L.back();
    L.pop_back();
    A.push_back(std::make_tuple(lp.first, lp.first, 1.0));
  }
}

int sample_alias(std::vector<std::tuple<int, int, double>> &A, int K) {
  int b = rand() % K;
  double p = rand() * 1.0 / (RAND_MAX + 1.0);
  auto t = A[b];
  int l = std::get<0>(t), h = std::get<1>(t);
  double prob = std::get<2>(t);
  if (p > prob)
    return h;
  else
    return l;
}


}; // namespace util