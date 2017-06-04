#include <fstream>
#include <sstream>
#include <iostream>
#include "documents.h"

Doc::Doc(const std::string &doc_line, const std::set<std::string>& stopwords_) {
  std::stringstream ss(doc_line); 
  std::string tmp;

  while (ss >> tmp)
    if (stopwords_.find(tmp) == stopwords_.end())
      words_.push_back(tmp);
}

Docs::Docs(const std::string file_name, const std::string stopwords_file) {
  std::ifstream infile(file_name);
  std::ifstream stopfile(stopwords_file);

  std::set<std::string> stopwords_;
  std::string doc_line;
  std::string word;

  while (stopfile >> word) {
    stopwords_.insert(word);
  }
  while (std::getline(infile, doc_line)) {
    //std::cout << doc_line << std::endl;
    doc_list_.push_back(Doc(doc_line, stopwords_));
  }

  std::map<std::string, int> word_count;
  for (auto doc : doc_list_) {
    auto wlist = doc.GetWords();
    for (auto w : wlist)
      word_count[w]++;
  }

  int id = 0;
  for (auto wc : word_count) {
      word2id_.insert(make_pair(wc.first, id));
      id2word_.insert(make_pair(id, wc.first));
      id++;
  }

}
