//
// Created by Steven Roddan on 8/11/2024.
//

#ifndef CUPYRE_INVALIDDIMENSIONERROR_H
#define CUPYRE_INVALIDDIMENSIONERROR_H

#include "Common.h"
#include <stdexcept>

namespace NaNLa::Exceptions {
    class InvalidDimensionError : public std::runtime_error {
    public:
        DECLSPEC InvalidDimensionError(const std::string& message);
    };
} // NaNLa

#endif //CUPYRE_INVALIDDIMENSIONERROR_H
