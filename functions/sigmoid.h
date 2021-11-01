//
// Created by tim on 05.10.21.
//

#ifndef CNNS_SIGMOID_H
#define CNNS_SIGMOID_H

#include <math.h>
#include "vector"

/**
 * \brief This function implements the sigmoid-function
 * @param x : value of interest
 * @return returns value of sigoid applied to x
 */
double sigmoid(double x);
/**
 * \brief This function implements the derivative of the sigmoid-function
 * @param x : value of interest
 * @return returns value of sigoid applied to x
 */
double d_sigmoid(double x);

/** \brief is needed when sigmoid is needed to be applied to vector
 *
 * @param input_vector : input vector after applying affin linear trafo
 * @param[out] result
 */
std::vector<double> sigmoid_list(std::vector<double> input_vector);

/** \brief is needed when d_sigmoid is needed to be applied to vector
 *
 * @param input_vector : input vector after applying affin linear trafo
 * @param[out] result
 */
std::vector<double> d_sigmoid_list(std::vector<double> input_vector);

#endif //CNNS_SIGMOID_H
