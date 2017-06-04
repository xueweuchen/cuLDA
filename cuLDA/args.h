#ifndef _ARGS_
#define _ARGS_

#include "util.h"

namespace util {

class OptionBase {
public:
  virtual void SetValue(const std::string &v) = 0;
};

template <class T>
class Option : public OptionBase {
public:
  Option(T v) {
    value_ = v;
  }
  void SetValue(const std::string &v) {
    std::istringstream ss(v);
    ss >> value_;
  }
  T GetValue() {
    return value_;
  }
private:
  T value_;
};

class ArgParse {
public:
  ArgParse() {}
  ~ArgParse() {
    for (auto p : kv_) {
      delete p.second;
      p.second = nullptr;
    }
  }

  void Parse(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
      std::string option = argv[i];
      if (option[0] == '-') {
        auto tokens = Split(option.substr(2), '=');
        if (kv_.find(tokens[0]) == kv_.end())
          continue;
        this->SetOption(tokens[0], tokens[1]);
      }
    }
  }

  template <class T>
  void AddOption(const std::string &name, T default) {
    kv_[name] = new Option<T>(default);
  }

  void SetOption(const std::string &name, const std::string &v) {
    kv_[name]->SetValue(v);
  }

  template <class T>
  T GetOption(const std::string &name) {
    return ((Option<T>*)kv_[name])->GetValue();
  }
private:
  std::map<std::string, OptionBase*> kv_;
};

}; //namespace util


#endif
