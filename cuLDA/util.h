#ifndef _UTIL_
#define _UTIL_


#include <vector>
#include <string>
#include <tuple>
#include <map>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ctime>
#include <numeric>
#include <algorithm>
#include <memory>
#include "cpulda.h"


namespace util {

std::vector<std::string> Split(const std::string &text, char sep);
void GenerateAlias(std::vector<std::tuple<int, int, double>> &A, std::vector<double> &q, int K);
int SampleAlias(std::vector<std::tuple<int, int, double>> &A, int K);

template <class T>
inline void DumpVector(const std::vector<T> &v, std::ofstream &fout) {
  for (auto e : v)
    fout << e << ' ';
  fout << std::endl;
}

}; //namespace util


#endif
