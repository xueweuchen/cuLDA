#ifndef _FACTORY_
#define _FACTORY_

#include "util.h"

class Registry {
public:
  typedef std::shared_ptr<LDA>(*Creator)();
  typedef std::map<std::string, Creator> CreatorRegistry;

  static CreatorRegistry& GetRegistry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  static void AddEntry(const std::string &type, Creator creator) {
    CreatorRegistry& registry = GetRegistry();
    registry[type] = creator;
  }

  static std::shared_ptr<LDA> Create(const std::string &type) {
    CreatorRegistry& registry = GetRegistry();
    return registry[type]();
  }
};

class Register {
public:
  Register(const std::string& type,
    std::shared_ptr<LDA>(*creator)()) {
    Registry::AddEntry(type, creator);
  }
};


#define REGISTER_CLASS(name, type) \
std::shared_ptr<LDA> Create_##type##() { \
  return std::shared_ptr<LDA>(new type##());  \
}               \
static Register register_##type##(name, Create_##type##);


#endif // !_FACTORY_
