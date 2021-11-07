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
