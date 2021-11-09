//
// Created by tim on 30.09.21.
//

#ifndef CNNS_FILTER_H
#define CNNS_FILTER_H

#include <iostream>
#include <vector>
#include "algorithm"
#include "functions/sigmoid.h"
#include "functions/ReLu.h"
#include "Matrix_computations.h"

class Filter {
    int dim_Filter;
    int padding;
    int stride;
    std::vector<std::vector<double>> Filter_grid;
    std::string activation_function;

private:


public:
    /**\brief Default Constructor needed for array initialization
     *
     */
    Filter();
    /** \brief Constructor of a Filter
    * @param dim_Filter: Dimension of Filter
    * @param padding : Number of added zeros on each edge
    * @param stride : Number of overjumped elements of each Filter operation
    */
    Filter(int dim_Filter, int padding, int stride);

    /** \brief Constructor of a Filter
    * @param dim_Filter: Dimension of Filter
    * @param padding : Number of added zeros on each edge
    * @param stride : Number of overjumped elements of each Filter operation
    * @param activation_function : activation_function for Filter
    */
    Filter(int dim_Filter, int padding,int stride,std::string activation_function);

    /** \brief Constructor of a Filter
    * @param dim_Filter: Dimension of Filter
    * @param padding : Number of added zeros on each edge
    * @param stride : Number of overjumped elements of each Filter operation
     *@param activation_function : activation_function for Filter
    * @param Filter_map : Default-Filter is given
    */
    Filter(int dim_Filter, int padding,int stride,std::string activation_function,std::vector<std::vector<double>> Filter_map);
    /** Filter function
    * @param image_array : Array which is going to be filtered
     *@param [out] filtered_array : filtered Array
    */
    std::vector<std::vector<double>>  run_Filter(std::vector<std::vector<double>> image_array);


    /** Function which checks whether current pair of indices is relevant for padding
     * @param i : current row index of filtered image
     * @param j : current column index of filtered image
     * @param inner_i : current row index of Filter
     * @param inner_j : current column index of Filter
     * @param rows : rows of input array
     * @param cols : columns of input array
     */
    bool check_for_padding(int i, int j, int inner_i, int inner_j,int rows,int cols);
    /**
     *
     * @param input_array : input image which needs to get padded
     * @return output_array : returns array which got padded edges
     */
    std::vector<std::vector<double>> apply_padding(std::vector<std::vector<double>> input_array);

    /**Function to fill rows of 2D Array with random values
     *
     * @param row
     */
    void static fill_row(std::vector<double> & row);

    /**Function to fill 2D array with random values
     *
     * @param mat
     */
    void fill_matrix(std::vector<std::vector<double>> & mat);


};


#endif //CNNS_FILTER_H
