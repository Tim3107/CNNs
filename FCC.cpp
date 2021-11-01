//
// Created by tim on 05.10.21.
//

#include "FCC.h"
#include "opencv2/opencv.hpp"
#include "functions/sigmoid.h"
#include "functions/ReLu.h"



FCC::FCC(int input_dim, int output_dim) {
    this->input_dim = input_dim;
    this->output_dim = output_dim;
    this->activation_function = "sigmoid";
    this->weight_matrix = std::vector<std::vector<double>>(output_dim,std::vector<double> (input_dim ,1)); //need random values here
    this->bias = std::vector<double>(output_dim,0);
}


FCC::FCC(int input_dim, int output_dim, std::string activation_function) {
    this->input_dim = input_dim;
    this->output_dim = output_dim;
    this->activation_function = activation_function;
    this->weight_matrix = std::vector<std::vector<double>>(output_dim,std::vector<double> (input_dim ,1)); //need random values here
    this->bias = std::vector<double>(output_dim,0);
}

FCC::FCC(int input_dim, int output_dim, std::string activation_function,std::vector<std::vector<double>> default_matrix, std::vector<double> default_bias){
    this->input_dim = input_dim;
    this->output_dim = output_dim;
    this->activation_function = activation_function;
    this->weight_matrix = default_matrix; //need random values here
    this->bias = default_bias;
}

std::vector<double> FCC::forward_step(std::vector<double> input_vector) {
    std::vector<double> temp (std::vector<double>(output_dim,0));
    temp = matrix_Vector_multiplication(this->weight_matrix,input_vector);// + this->bias;
    std::transform(temp.begin(),temp.end(),this->bias.begin(),temp.begin(),std::plus<double>());
    if(this->activation_function == "sigmoid"){
        temp = sigmoid_list(temp);
    }
    else if(this->activation_function == "relu"){
        temp = reLu_list(temp);
    }
    this->input = input_vector;
    this->output = temp;
    return temp;
}


std::vector<double> FCC::forward_step(std::vector<std::vector<std::vector<double>>> input_vector) {
    int size = input_vector.size();
    std::vector<double> input_vector_1D(std::vector<double>(size,0));
    for(int i = 0;i<size;i++){
        input_vector_1D[i] = input_vector[i][0][0];
    }
    std::vector<double> temp (std::vector<double>(output_dim,0));
    return this->forward_step(input_vector_1D);
}


//std::tuple<std::vector<std::vector<double>>,std::vector<double>>
std::vector<double> FCC::backward_step(std::vector<double> input_successor){
    std::vector<double> output_vector(this->input_dim);
    std::vector<double> temp (std::vector<double>(output_dim,0));
    temp = matrix_Vector_multiplication(this->weight_matrix,input_successor);// + this->bias;
    std::transform(temp.begin(),temp.end(),bias.begin(),temp.begin(),std::plus<double>());
    std::vector<double> temp_2 (std::vector<double>(output_dim,0));
    if(this->activation_function == "sigmoid")
        temp_2 = d_sigmoid_list(temp);
    else if(this->activation_function == "relu")
        temp_2 = d_reLu_list(temp);
    std::transform(temp_2.begin(),temp_2.end(),input_successor.begin(),temp_2.begin(),std::plus<double>());

    update_weights(dyadic_product(input_successor,this->input),input_successor);

    return matrix_Vector_multiplication_transpose(this->weight_matrix,temp_2);

}

std::tuple<std::vector<std::vector<double>>,std::vector<double>> FCC::getter(){
    return {this->weight_matrix,this->bias};
}


void FCC::setter_learning_rate(double learning_rate){
    this->learning_rate = learning_rate;
}

void FCC::update_weights(std::vector<std::vector<double>> update_weight_matrix,std::vector<double> update_bias_vector){
    this->weight_matrix = add_matrices(this->weight_matrix, matrix_scalar(update_weight_matrix,-1*this->learning_rate));
    this->bias = vector_computations(this->bias,update_bias_vector,-1*this->learning_rate);
}