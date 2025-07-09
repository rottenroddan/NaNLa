//
// Created by Steven Roddan on 8/11/2024.
//

#include "InvalidDimensionError.h"

namespace NaNLa:: Exceptions {

    DECLSPEC InvalidDimensionError::InvalidDimensionError(const std::string &message) : runtime_error(message) {
        ;
    }
} // NaNLa