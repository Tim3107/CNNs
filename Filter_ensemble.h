//
// Created by tim on 11.10.21.
//
#include "Filter.h"
#include "Matrix_computations.h"

#ifndef CNNS_FILTER_ENSEMBLE_H
#define CNNS_FILTER_ENSEMBLE_H

class Filter_set{
    int anzahl_filter;
    int dim_filters;
    int padding;
    int stride;
    std::string activation_function;
    Filter* filters;
private:

public:
    /**Default Constructor needed for declaration of Filter_layer
     *
     */
    Filter_set();
    /** \brief Constructor which creates batches of filters
     *
     * @param anzahl_filter : Number of Filters
     * @param dim_filter : size of Filters
     * @param padding : size of padding
     * @param stride : stride
     * @param activation_function : activation function after filter operation
     */
    Filter_set(int anzahl_filter, int dim_filter, int padding, int stride, std::string activation_function);

    /** Filter function, all filters are applied to same image. This function takes the methods of each Filter
    * @param image_array : Array which is going to be filtered
    * @param [out] filtered_array : filtered Array
    */
    std::vector<std::vector<double>>  run_Filters(std::vector<std::vector<std::vector<double>>> image_array);
    /** Filter function, all filters are applied to same image. This function takes the methods of each Filter
    * @param image_array : Array which is going to be filtered
    * @param [out] filtered_array : filtered Array
    */
    std::vector<std::vector<double>>  run_Filters(std::vector<std::vector<double>> image_array);

};

#endif //CNNS_FILTER_ENSEMBLE_H
