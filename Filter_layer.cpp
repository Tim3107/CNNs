//
// Created by tim on 13.10.21.
//

#include "Filter_layer.h"


Filter_layer::Filter_layer(int Filter_dim, int padding, int stride,std::string activation_function, int input_channels, int output_channels) {
    this->Filter_dim = Filter_dim;
    this->padding = padding;
    this->stride = stride;
    this->activation_function = activation_function;
    this->input_channels = input_channels;
    this->output_channels = output_channels;

    this->filter_sets = new Filter_set[output_channels];
    for (int i = 0;i<this->output_channels; i++){
        filter_sets[i] = Filter_set(this->input_channels,this->Filter_dim , this->padding, this->stride,this->activation_function);
    }

}

std::vector<std::vector<std::vector<double>>> Filter_layer::run_Filter_one_channel(std::vector<std::vector<std::vector<double>>> input_image) {
    int rows = input_image[0].size();
    int cols = input_image[0][0].size();

    int new_rows = (rows - this->Filter_dim + 2 * this->padding) / this->stride + 1;
    int new_cols = (cols - this->Filter_dim + 2 * this->padding) / this->stride + 1;
    std::vector<std::vector<std::vector<double>>> output_image(this->output_channels, std::vector<std::vector<double>>(new_rows, std::vector<double>(new_cols, 0)));

    for (int i = 0;i<this->output_channels;i++){
        output_image[i] = this->filter_sets[i].run_Filters(input_image);
    }

    return output_image;
}

std::vector<std::vector<std::vector<double>>> Filter_layer::run_Filter_one_channel(std::vector<std::vector<double>> input_image) {
    int rows = input_image.size();
    int cols = input_image[0].size();

    int new_rows = (rows - this->Filter_dim + 2 * this->padding) / this->stride + 1;
    int new_cols = (cols - this->Filter_dim + 2 * this->padding) / this->stride + 1;
    std::vector<std::vector<std::vector<double>>> output_image(this->output_channels, std::vector<std::vector<double>>(new_rows, std::vector<double>(new_cols, 0)));

    for (int i = 0;i<this->output_channels;i++){
        output_image[i] = this->filter_sets[i].run_Filters(input_image);
    }

    return output_image;
}