#ifndef _ARGS_
#define _ARGS_

#include "util.h"

namespace util {

class OptionBase {
public:
  virtual void set_value(const std::string &v) = 0;
};

template <class T>
class Option : public OptionBase {
public:
  Option(T v) {
    value = v;
  }
  void set_value(const std::string &v) {
    std::istringstream ss(v);
    ss >> value;
  }
  T get_value() {
    return value;
  }
private:
  T value;
};

class ArgParse {
public:
  ArgParse() {}
  ~ArgParse() {
    for (auto p : kv) {
      delete p.second;
      p.second = nullptr;
    }
  }

  void parse(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
      std::string option = argv[i];
      if (option[0] == '-') {
        auto tokens = split(option.substr(2), '=');
        if (kv.find(tokens[0]) == kv.end())
          continue;
        this->set_option(tokens[0], tokens[1]);
      }
    }
  }

  template <class T>
  void add_option(const std::string &name, T default) {
    kv[name] = new Option<T>(default);
  }

  void set_option(const std::string &name, const std::string &v) {
    kv[name]->set_value(v);
  }

  template <class T>
  T get_option(const std::string &name) {
    return ((Option<T>*)kv[name])->get_value();
  }
private:
  std::map<std::string, OptionBase*> kv;
};

}; //namespace util


#endif
