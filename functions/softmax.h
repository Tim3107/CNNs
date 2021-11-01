//
// Created by tim on 05.10.21.
//

#ifndef CNNS_SOFTMAX_H
#define CNNS_SOFTMAX_H

#include <math.h>
#include <vector>

/**
 * \brief This function implements the softmax function
 * @param x : Vector of interest
 * @return softmax(x)
 */
std::vector<double> softmax(std::vector<double> x);

/**
 * \brief This function implements the derivative of the softmax function
 * @param x : Vector of interest
 * @return softmax(x)
 */
std::vector<double> d_softmax(std::vector<double> x);

#endif //CNNS_SOFTMAX_H
