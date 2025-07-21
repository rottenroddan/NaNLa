//
// Created by Steven Roddan on 7/17/2025.
//

#ifndef NANLA_THREECLASSDATAGENERATOR_H
#define NANLA_THREECLASSDATAGENERATOR_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#include "DataPoint.h"

int classify_by_angle(float x, float y);

std::vector<DataPoint> generate_dataset(int num_points, float spread);

#endif //NANLA_THREECLASSDATAGENERATOR_H
