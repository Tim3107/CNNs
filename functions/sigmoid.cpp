//
// Created by tim on 05.10.21.
//

#include "sigmoid.h"


using namespace std;

double sigmoid(double x){
    return 1.0/(1+exp(-x));
}
double d_sigmoid(double x){
    return sigmoid(x)*(1- sigmoid(x));
}

std::vector<double> sigmoid_list(std::vector<double> input_vector){
    int size = input_vector.size();
    std::vector<double> return_vec(size,0);
    for (int i = 0;i<size;i++){
        return_vec[i] = sigmoid(input_vector[i]);
    }
    return return_vec;
}

std::vector<double> d_sigmoid_list(std::vector<double> input_vector){
    int size = input_vector.size();
    std::vector<double> return_vec(size,0);
    for (int i = 0;i<size;i++){
        return_vec[i] = d_sigmoid(input_vector[i]);
    }
    return return_vec;
}

std::vector<std::vector<double>> sigmoid_array(std::vector<std::vector<double>> input_array){
    int rows = input_array.size();
    int cols = input_array[0].size();
    std::vector<std::vector<double>> return_array(rows,std::vector<double>(cols,0));
    for (int i = 0;i<rows;i++){
        return_array[i] = sigmoid_list(input_array[i]);
    }
    return return_array;
}

std::vector<std::vector<double>> d_sigmoid_array(std::vector<std::vector<double>> input_array){
    int rows = input_array.size();
    int cols = input_array[0].size();
    std::vector<std::vector<double>> return_array(rows,std::vector<double>(cols,0));
    for (int i = 0;i<rows;i++){
        return_array[i] = d_sigmoid_list(input_array[i]);
    }
    return return_array;
}