//
// Created by tim on 13.10.21.
//

#include "Filter_layer.h"

Filter_layer::Filter_layer() {

}

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
    this->input = input_image;
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
    this->input.push_back(input_image);

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


std::vector<std::vector<std::vector<double>>> Filter_layer::backward_step_filter_set(std::vector<std::vector<std::vector<double>>> input_successor){
    int channels = input_successor.size();
    int rows = input_successor[0].size();
    int cols = input_successor[0][0].size();

    int new_rows = (rows-1)*this->stride- 2*this->padding+this->Filter_dim;
    int new_cols = (cols-1)*this->stride- 2*this->padding+this->Filter_dim;

    std::vector<std::vector<std::vector<double>>> return_gradients(this->input_channels, std::vector<std::vector<double>>(new_rows, std::vector<double>(new_cols, 0)));

    for (int i = 0; i< this->input_channels;i++){
        for (int j = 0;j<this->output_channels;j++){
            return_gradients[i] = add_matrices(return_gradients[i],this->filter_sets[j].backward_step_filter_set(input_successor[j],i));
        }
    }

    for (int i = 0;i<this->output_channels;i++){
        this->filter_sets[i].setter_learning_rate(this->learning_rate);
        for (int j = 0;j < this->input_channels;j++){
            this->filter_sets[i].backward_step_update_filters(this->input[j],input_successor[i],j);
        }
    }



    return  return_gradients;
}

void Filter_layer::setter_learning_rate(double learning_rate) {
    this->learning_rate = learning_rate;
}