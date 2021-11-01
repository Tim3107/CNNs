//
// Created by tim on 05.10.21.
//

#ifndef CNNS_RELU_H
#define CNNS_RELU_H

#include <math.h>
#include "vector"

/**
 * \brief This function implements the ReLu-function
 * @param x : value of interest
 * @return returns value of ReLu applied to x
 */
double reLu(double x);
/**
 * \brief This function implements the derivative of the ReLu-function
 * @param x : value of interest
 * @return returns value of ReLu applied to x
 */
double d_reLu(double x);

/** \brief is needed when relu is needed to be applied to vector
 *
 * @param input_vector : input vector after applying affin linear trafo
 * @param[out] result
 */
std::vector<double> d_reLu_list(std::vector<double> input_vector);

/** \brief is needed when d_relu is needed to be applied to vector
 *
 * @param input_vector : input vector after applying affin linear trafo
 * @param[out] result
 */
std::vector<double> reLu_list(std::vector<double> input_vector);

#endif //CNNS_RELU_H
