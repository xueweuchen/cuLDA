#ifndef _DOCUMENTS_
#define _DOCUMENTS_

#include <vector>
#include <string>
#include <map>
#include <set>

class Doc {
public:
  Doc(const std::string &doc_line, const std::set<std::string> &stopwords_);
  ~Doc() {}

  const std::vector<std::string>& GetWords() const {
    return words_;
  }

private:
  std::vector<std::string> words_;
};

class Docs {
public:
  Docs(const std::string file_name, const std::string stopwords_file);
  ~Docs() {}

  const std::vector<Doc>& GetDoclist() const {
    return doc_list_;
  }
  const std::map<int, std::string>& GetIdToWord() const {
    return id2word_;
  }
  const std::map<std::string, int>& GetWordToId() const {
    return word2id_;
  }

private:
  std::set<std::string> stopwords_;
  std::vector<Doc> doc_list_;
  std::map<int, std::string> id2word_;
  std::map<std::string, int> word2id_;

};

#endif // !_DOCUMENTS_

