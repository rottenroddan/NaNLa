//
// Created by Steven Roddan on 10/1/2025.
//

#ifndef NANLA_LAYER_H
#define NANLA_LAYER_H

#include <concepts>
#include "NaNLA/Matrix/AbstractMatrix.h"


namespace NaNLA::NN {
    template<class T, class U = T>
    class Layer {
    public:
        virtual auto forward(T t) -> U = 0;
        virtual auto backward(T t) -> U = 0;
        virtual auto update(float lr) -> void = 0;
    };
}

#endif //NANLA_LAYER_H
