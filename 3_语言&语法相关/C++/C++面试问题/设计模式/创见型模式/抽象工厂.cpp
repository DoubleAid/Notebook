#include <iostream>
#include <string>
#include <memory>

using namespace std;

class Movie {
public:
    virtual void get_name() = 0;
};

class CNMovie : public Movie {
public:
    virtual void get_name() override {
        cout << "cn movie name" << endl;
    }
};

class USMovie : public Movie {
public:
    virtual void get_name() override {
        cout << "us movie name" << endl;
    }
};

class Factory {
public:
    virtual shared_ptr<Movie> get_movie() = 0;
};

class USFactory : public Factory {
public:
    virtual shared_ptr<Movie> get_movie() override {
        return make_shared<USMovie>();
    }
};

class CNFactory : public Factory {
public:
    virtual shared_ptr<Movie> get_movie() override {
        return make_shared<CNMovie>();
    }
};

int main() {
    string config = "cn";
    shared_ptr<Factory> factory;
    if(config == "cn") {
        factory.reset(new CNFactory());
    } else {
        factory.reset(new USFactory());
    }
    shared_ptr<Movie> movie = factory->get_movie();
    movie->get_name();
}