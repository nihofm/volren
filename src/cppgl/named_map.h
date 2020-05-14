#pragma once

#include <map>
#include <memory>
#include <string>

template <typename T> class NamedMap {
public:
    NamedMap(const std::string& name) : name(name) {
        if (map.count(name))
            throw std::runtime_error("ERROR: Key \"" + name + "\" not unique in NamedMap!");
        map[name] = (T*) this;
    }

    virtual ~NamedMap() {
        map.erase(name);
    }

    // check if mapping for given name exists
    static bool valid(const std::string& name) { return map.count(name); }
    // return (non-ownership) pointer to mapping for given name
    static T* find(const std::string& name) { return map[name]; }

    // iterators to iterate over all entries
    static typename std::map<std::string, T*>::iterator begin() { return map.begin(); }
    static typename std::map<std::string, T*>::iterator end() { return map.end(); }

    const std::string name;
    static std::map<std::string, T*> map;
};

// definition of static member (compiler magic)
template <typename T> std::map<std::string, T*> NamedMap<T>::map;
