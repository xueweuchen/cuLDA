#ifndef _DOCUMENTS_
#define _DOCUMENTS_

#include <vector>
#include <string>
#include <map>
#include <set>

class Doc {
public:
  Doc(const std::string &doc_line, const std::set<std::string> &stopwords);
  ~Doc() {}

  const std::vector<std::string>& get_words() const {
    return words;
  }

private:
  std::vector<std::string> words;
};

class Docs {
public:
  Docs(const std::string file_name, const std::string stopwords_file);
  ~Docs() {}

  const std::vector<Doc>& get_doclist() const {
    return doc_list;
  }
  const std::map<int, std::string>& get_id2word() const {
    return id2word;
  }
  const std::map<std::string, int>& get_word2id() const {
    return word2id;
  }

private:
  std::set<std::string> stopwords;
  std::vector<Doc> doc_list;
  std::map<int, std::string> id2word;
  std::map<std::string, int> word2id;

};

#endif // !_DOCUMENTS_

