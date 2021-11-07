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
    this->weight_matrix = random_matrix(output_dim,input_dim,1);//std::vector<std::vector<double>>(output_dim,std::vector<double> (input_dim ,1)); //need random values here
    this->bias = random_vector(output_dim,1);//std::vector<double>(output_dim,0);
    this->learning_rate = 1.0;
}


FCC::FCC(int input_dim, int output_dim, std::string activation_function) {
    this->input_dim = input_dim;
    this->output_dim = output_dim;
    this->activation_function = activation_function;
    this->weight_matrix = random_matrix(output_dim,input_dim,1);//std::vector<std::vector<double>>(output_dim,std::vector<double> (input_dim ,1)); //need random values here
    this->bias = random_vector(output_dim,1);//std::vector<double>(output_dim,0);
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
    this->temp_affin = temp;
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
    //temp = matrix_Vector_multiplication_transpose(this->weight_matrix,input_successor);// + this->bias;
    //std::transform(temp.begin(),temp.end(),bias.begin(),temp.begin(),std::plus<double>());
    std::vector<double> temp_2 (std::vector<double>(output_dim,0));
    if(this->activation_function == "sigmoid")
        temp_2 = d_sigmoid_list(this->temp_affin);
    else if(this->activation_function == "relu")
        temp_2 = d_reLu_list(this->temp_affin);
    else if(this->activation_function == "none"){
        temp_2 = temp;
    }

    temp_2 = vector_multiplication_elementwise(input_successor,temp_2,1.0);
    output_vector = matrix_Vector_multiplication_transpose(this->weight_matrix,temp_2);
    this->output = output_vector;
    //std::cout << "dyd" <<std::endl;
    //display_array(dyadic_product(input_successor,this->input));
    //for (int i = 0;i<this->input.size();i++){
    //    std::cout << this->input[i] << "HHOHOHO"<< std::endl;
    // }
    this->update_weights(dyadic_product(temp_2,this->input),temp_2);

    return output_vector;

}


std::vector<double> FCC::backward_step(std::vector<double> input_successor,std::vector<double> input_vec){
    std::vector<double> output_vector(this->input_dim);
    std::vector<double> temp (std::vector<double>(output_dim,0));
    temp = matrix_Vector_multiplication(this->weight_matrix,input_successor);// + this->bias;
    std::transform(temp.begin(),temp.end(),bias.begin(),temp.begin(),std::plus<double>());
    std::vector<double> temp_2 (std::vector<double>(output_dim,0));
    if(this->activation_function == "sigmoid")
        temp_2 = d_sigmoid_list(temp);
    else if(this->activation_function == "relu")
        temp_2 = d_reLu_list(temp);
    else if(this->activation_function == "none"){
        temp_2 = temp;
    }
    std::transform(temp_2.begin(),temp_2.end(),input_successor.begin(),temp_2.begin(),std::plus<double>());
    //std::cout << "dyd" <<std::endl;
    //display_array(dyadic_product(input_successor,input_vec));
    //for (int i = 0;i<this->input.size();i++){
    //    std::cout << this->input[i] << "HHOHOHO"<< std::endl;
    // }
    //std::cout << "Dyadic Product"<< std::endl;
    //display_array(dyadic_product(input_successor,input_vec));
    //std::cout << " "<< std::endl;
    //std::cout << "Vector_1"<< std::endl;
    //display_vector(input_successor);
    //std::cout << " "<< std::endl;

    //std::cout << "Vector_2"<< std::endl;
    //display_vector(input_vec);
    //std::cout << " "<< std::endl;
    this->update_weights(dyadic_product(input_successor,input_vec),input_successor);

    return matrix_Vector_multiplication_transpose(this->weight_matrix,temp_2);

}

std::vector<double> FCC::getter_vector(){
    return this->bias;
}

std::vector<std::vector<double>> FCC::getter_matrix(){
    return this->weight_matrix;
}


void FCC::setter_learning_rate(double learning_rate){
    this->learning_rate = learning_rate;
}

void FCC::update_weights(std::vector<std::vector<double>> update_weight_matrix,std::vector<double> update_bias_vector){
    /*
    std::cout << "" << std::endl;
    std::cout << "-------------------------" << std::endl;
    std::cout << "" << std::endl;
    display_array(update_weight_matrix);
    std::cout << "" << std::endl;
    display_vector(update_bias_vector);
    std::cout << "" << std::endl;
    std::cout << "-------------------------" << std::endl;
    std::cout << "" << std::endl;
     */
    this->weight_matrix = add_matrices(this->weight_matrix, matrix_scalar(update_weight_matrix,-1*this->learning_rate));
    this->bias = vector_additions(this->bias,update_bias_vector,-1*this->learning_rate);
}