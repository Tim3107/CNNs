//
// Created by tim on 04.11.21.
//

#ifndef CNNS_MSE_H
#define CNNS_MSE_H

#include "vector"
#include "../Tools/Matrix_computations.h"
#include "iostream"
#include "math.h"



/**This method computes the MSE-Error
 *
 * @param output : Output of FCC
 * @param target_label : Target_values
 * @return MEAN-SQUARED ERROR
 */
double mse_compute_error(std::vector<double> output,std::vector<double> target_label);

#endif //CNNS_MSE_H
