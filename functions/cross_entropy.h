//
// Created by tim on 09.10.21.
//

#ifndef CNNS_CROSS_ENTROPY_H
#define CNNS_CROSS_ENTROPY_H

#include <math.h>
#include "vector"

/**
 * \brief This function implements the cross entropy loss-function
 * @param x : value of interest
 * @return returns value of cross enetropy applied to x
 */
double cross_entropy(double x,double label);
/**
 * \brief This function implements the derivative of the cross entropy loss-function
 * @param x : value of interest
 * @return returns value of cross entropy loss-function applied to x
 */
double d_cross_entropy(double x,double label);

/** \brief is needed when cross entropy loss-function is needed to be applied to vector
 *
 * @param input_vector : input vector after applying affin linear trafo
 * @param[out] result
 */
double cross_entropy_list(std::vector<double> input_vector, std::vector<double> label);

/** \brief is needed when d_cross entropy loss-function is needed to be applied to vector
 *
 * @param input_vector : input vector after applying affin linear trafo
 * @param[out] result
 */
std::vector<double> d_cross_entropy_list(std::vector<double> input_vector, std::vector<double> label);

#endif //CNNS_CROSS_ENTROPY_H