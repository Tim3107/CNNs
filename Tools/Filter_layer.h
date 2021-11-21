//
// Created by tim on 13.10.21.
//

#ifndef CNNS_FILTER_LAYER_H
#define CNNS_FILTER_LAYER_H

#include "vector"
#include "Filter_ensemble.h"


class Filter_layer{
    int Filter_dim;
    int padding;
    int stride;
    double learning_rate = 1.0;
    std::string activation_function;
    int input_channels;
    int output_channels;
    Filter_set* filter_sets;
    std::vector<std::vector<std::vector<double>>> input;

private:
public:
    /**Default constructor for initializations
     *
     */
    Filter_layer();

    /**@brief Constructor. Instances are objects which consist of couple Filter_ensembles. One for each output_channel
     *
     * @param Filter_dim : Dim of the Filters which are used.
     * @param padding : number of added zeros at edges
     * @param stride : stride
     * @param activation_function : activation function after filter operation
     * @param input_channels : Channels of input image
     * @param output_channels : Channels of output channel
     */
    Filter_layer(int Filter_dim, int padding, int stride,std::string activation_function,int input_channels, int output_channels);


    /**This routine gets a 3D image and uses a 4D Filter to get a 3D image
     *
     * @param input_image : input image which consists of height, width and number of input channels
     * @return output_image : 3D output with height, width and output_channels
     */
    std::vector<std::vector<std::vector<double>>> run_Filter_one_channel(std::vector<std::vector<std::vector<double>>> input_image);

    /**This routine gets a 2D image and uses a 4D Filter to get a 3D image
 *
 * @param input_image : input image which consists of height, width and number of input channels
 * @return output_image : 3D output with height, width and output_channels
 */
    std::vector<std::vector<std::vector<double>>> run_Filter_one_channel(std::vector<std::vector<double>> input_image);

    /**This routine computes the backpropagation step for one Filter_set
     *
     * @param input_successor : BP input gradient from successor
     * @return output_array : Gradient of
     */
    std::vector<std::vector<std::vector<double>>> backward_step_filter_set(std::vector<std::vector<std::vector<double>>> input_successor);

    /**This method allows to set the learning rate for all filter_sets.
     *
     * @param learning_rate
     */
    void setter_learning_rate(double learning_rate);
};


#endif //CNNS_FILTER_LAYER_H
