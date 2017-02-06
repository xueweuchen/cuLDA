#ifndef _FACTORY_
#define _FACTORY_

#include "util.h"

class Registry {
public:
  typedef std::shared_ptr<LDA>(*Creator)();
  typedef std::map<std::string, Creator> CreatorRegistry;

  static CreatorRegistry& getRegistry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  static void addEntry(const std::string &type, Creator creator) {
    CreatorRegistry& registry = getRegistry();
    registry[type] = creator;
  }

  static std::shared_ptr<LDA> create(const std::string &type) {
    CreatorRegistry& registry = getRegistry();
    return registry[type]();
  }
};

class Register {
public:
  Register(const std::string& type,
    std::shared_ptr<LDA>(*creator)()) {
    Registry::addEntry(type, creator);
  }
};


#define REGISTER_CLASS(name, type) \
std::shared_ptr<LDA> create_##type##() { \
  return std::shared_ptr<LDA>(new type##());  \
}               \
static Register register_##type##(name, create_##type##);


#endif // !_FACTORY_
