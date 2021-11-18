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
/**This routine computes relu for arrays
 *
 * @param input_array : input image/array
 * @return return_array : output after relu is performed
 */
std::vector<std::vector<double>> reLu_array(std::vector<std::vector<double>> input_array);

/**This routine computes d_relu for arrays
 *
 * @param input_array : input image/array
 * @return return_array : output after relu is performed
 */
std::vector<std::vector<double>> d_reLu_array(std::vector<std::vector<double>> input_array);

#endif //CNNS_RELU_H
