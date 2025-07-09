#include <iostream>
#include <memory>

template<class T>
class Base {
public:
    virtual void print() { std::cout << "Base" << std::endl; }
};

template<class T>
class Host : public Base<T> {
public:
    void print() override { std::cout << "Host" << std::endl; }
    virtual T at() { return (T) 100; }
};

template<class T>
class HostOther : public Host<T> {
public:
    void print() override { std::cout << "HostOther" << std::endl; }
    virtual T at() override { return (T) 200; }
};

template<class T>
class Device : public Base<T> {
public:
    void print() override { std::cout << "Device" << std::endl; }
};

template<class T>
class AbstractMatrix {
protected:
    Base<T> *u;

    explicit AbstractMatrix(Base<T> *u) {
        this->u = u;
    }
public:
    virtual void print() = 0;
};

template<class T, class U>
class Matrix : public AbstractMatrix<T> {
protected:
    explicit Matrix(Base<T> *u) : AbstractMatrix<T>(u) { ; }
public:
    Matrix() : AbstractMatrix<T>(new U){ ; }

    void print() override { this->u->print(); }
};

template<class T>
class Matrix<T, Host<T>> : public AbstractMatrix<T> {
protected:
    explicit Matrix(Host<T>* u) : AbstractMatrix<T>(u) {

    }
public:

    Matrix() : Matrix(new Host<T>) {
    }

    void print() override { this->u->print(); }
    virtual T at() {
        return static_cast<Host<T>*>(this->u)->at();
    }
};

template<class T>
class Matrix<T, HostOther<T>> : public Matrix<T, Host<T>> {
protected:

public:
    Matrix() : Matrix<T, Host<T>>(new HostOther<T>) { ; }

    T at() override {
    return static_cast<HostOther<T>*>(this->u)->at();
    }
};

template<class T>
AbstractMatrix<T>* factoryMethod(int x) {
    if(x == 0)
        return new Matrix<T, Base<T>>;
    else if(x == 1)
        return new Matrix<T, Host<T>>;
    else if(x == 2)
        return new Matrix<T, HostOther<T>>;
    else if(x == 3)
        return new Matrix<T, Device<T>>;
}

template<class T, class U, class V = int>
class Foo {
public:
        V v = (V) 200;
    };

template<class T, template<class, class> class U, class V>
        class Bar {
        public:
            U<T,V> u;
        };


int main() {
    auto x = factoryMethod<int>(0);
    x->print();

    auto y = factoryMethod<int>(1);
    y->print();

    auto yy = factoryMethod<int>(2);
    yy->print();

    auto z = factoryMethod<int>(3);
    z->print();

    Bar<int, Foo, float> a;
    std::cout << a.u.v << std::endl;
}

void print() {
    int x;
}


