//
// Created by tim on 11.11.21.
//

#ifndef CNNS_FILTER_OPERATIONS_H
#define CNNS_FILTER_OPERATIONS_H

#include "Matrix_computations.h"
#include "iostream"
#include "vector"

/**This method performs a filter operation between two arrays
 * @param stride : stride
 * @param padding : padded zeros on each side
 * @param input_array_1 : input array
 * @param input_array_2 : second input array
 */

std::vector<std::vector<double>> Filter_2D(int padding, int stride, std::vector<std::vector<double>> input_array_1,std::vector<std::vector<double>> input_array_2);


/**This method is in charge to pad an image
 *
 * @param input_array : image to be padded
 * @param padding : number of zeros on each side
 * @return padded array
 */
std::vector<std::vector<double>> apply_padding(std::vector<std::vector<double>> input_array, int padding);


#endif //CNNS_FILTER_OPERATIONS_H
