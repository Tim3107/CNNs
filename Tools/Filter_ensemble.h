//
// Created by tim on 11.10.21.
//

#ifndef CNNS_FILTER_ENSEMBLE_H
#define CNNS_FILTER_ENSEMBLE_H

#include "Filter.h"
#include "Matrix_computations.h"
#include "../functions/sigmoid.h"
#include "../functions/ReLu.h"
#include "Matrix_computations.h"
#include "Filter_Operations.h"

class Filter_set{
    int anzahl_filter;
    int dim_filters;
    int padding;
    int stride;
    double learning_rate = 1.0;
    double bias;
    std::string activation_function;
    Filter* filters;
    std::vector<std::vector<double>> temp_affin;
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

    /**This routine computes the backpropagation step for one Filter_set
     *
     * @param input_successor : BP input gradient from successor
     * @param index_of_channel : This index is needed to know to which Filter in Filterset this gradient belongs. I.e. which Input is filtered with which Filter?
     * @return output_array : Gradient of
     */
    std::vector<std::vector<double>> backward_step_filter_set(std::vector<std::vector<double>> input_successor,int index_of_channel);

    /**This method helps to compute the filter gradients and update the filters with the BP and Gradient decent scheme
     *
     * @param input : Input for this filter ensemble
     * @param input_successor : Backpropagated Input from successor layer of BP
     * @param filter_index : which filter in this ensemble is of interest
     */
    void backward_step_update_filters(std::vector<std::vector<double>> input,std::vector<std::vector<double>> input_successor,int filter_index);

    /**This setter method allows to change the learning rate.
     *
     * @param learning_rate
     */
    void setter_learning_rate(double learning_rate);

    /**This method initializes the bias randomly
     *
     *
     */
    void initialize_bias();
};

#endif //CNNS_FILTER_ENSEMBLE_H
