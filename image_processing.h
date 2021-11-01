//
// Created by tim on 05.10.21.
//

#ifndef CNNS_IMAGE_PROCESSING_H
#define CNNS_IMAGE_PROCESSING_H
#include <vector>
#include "opencv2/opencv.hpp"
#include <string>
#include <iterator>
#include <sstream>

/** @brief function that converts 2D-std::vector to cv::Mat image
 * @param input_array : Input Array of cv::Mat type is given
 * @param[out] output_array : 2D-std::vector is returned
 */

std::vector<std::vector<double>> conversion_to_std_vector(cv::Mat input_array);
/**
 * \brief function that converts cv::Mat object to 2D std::vector
 * @param input_array : Input image of 2D std:vector shape
 * @return output_array : Output image of cv::Mat type
 */

cv::Mat1f conversion_to_Mat(std::vector<std::vector<double>> input_array);

/**
 * \brief This function simply prints the entries of a 2D std::vector row-wise on the console.
 * @param input_array : Input array
 */
void display_array(std::vector<std::vector<double>> input_array);

/** @brief method which converts std::vector of strings to doubles
 *
 * @param pointLine
 * @return
 */
std::vector<double> getVertexIndices(std::string const& pointLine);



#endif //CNNS_IMAGE_PROCESSING_H


