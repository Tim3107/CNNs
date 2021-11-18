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
    if(this->classification_function == "softmax" && this->loss_function == "cross_entropy"){
        output = softmax(input_vector);
        this->output = output;
        return output;
    }
    else if(this->classification_function == "none" && this->loss_function == "MSE"){
        this->output = input_vector;
        return input_vector;
    }
    else{
        this->output = input_vector;
        return input_vector;

    }
    this->output = output;

    return output;
}

double Output_layer::compute_loss(std::vector<double> target_values) {
    if(this->loss_function == "cross_entropy" && this->classification_function == "softmax"){
        this->loss = cross_entropy_list(this->output,target_values);
        return this->loss;
    }

    if(this->loss_function == "MSE" && this->classification_function == "none"){
        this->loss = mse_compute_error(this->output,target_values);
        return this->loss;
    }
}

double Output_layer::compute_loss(std::vector<double> target_values,std::vector<double> given_output) {
    if(this->loss_function == "cross_entropy" && this->classification_function == "softmax"){
        this->loss = cross_entropy_list(given_output,target_values);
        return this->loss;
    }

    if(this->loss_function == "MSE" && this->classification_function == "none"){
        this->loss = mse_compute_error(given_output,target_values);
        return this->loss;
    }
}

std::vector<double> Output_layer::backward_step(std::vector<double> label){

    if (this->loss_function == "cross_entropy" && this->classification_function == "softmax"){
        return vector_additions(this->output,label,-1);

    }

    else if(this->loss_function == "MSE" && this->classification_function == "none"){
        return vector_additions(this->output,label,-1);
    }

    else if(this->loss_function == "MSE"){
        return vector_additions(this->output,label,-1);
    }
}

