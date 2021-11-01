//
// Created by tim on 09.10.21.
//

#include "output_layer.h"



Output_layer::Output_layer(int input_dim, int output_dim, std::string loss_function, std::string classification_function){
    this->input_dim = input_dim;
    this->output_dim = output_dim;
    this->loss_function = loss_function;
    this->classification_function = classification_function;
    this->labels = std::vector<std::string>(this->input_dim,"");
    this->input = std::vector<double>(input_dim,0);
    this->output = std::vector<double>(output_dim,0);
}

std::vector<double> Output_layer::forward_step(std::vector<double> input_vector) {
    std::vector<double> output(this->output_dim,0);
    if(this->classification_function == "softmax"){
        output = softmax(input_vector);
    }
    this->output = output;

    return output;
}

void Output_layer::compute_loss(std::vector<double> target_values) {
    if(this->loss_function == "cross_entropy"){
        this->loss = cross_entropy_list(this->output,target_values);
    }
}

