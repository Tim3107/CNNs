//
// Created by tim on 05.10.21.
//

#include "ReLu.h"

double reLu(double x){
    return std::fmax(0,x);
}
double d_reLu(double x){
    if(x>0){
        return 1;
    }
    else{
        return 0;
    }
}


std::vector<double> reLu_list(std::vector<double> input_vector){
    int size = input_vector.size();
    std::vector<double> return_vec(size,0);
    for (int i = 0;i<size;i++){
        return_vec[i] = reLu(input_vector[i]);
    }
    return return_vec;
}

std::vector<double> d_reLu_list(std::vector<double> input_vector){
    int size = input_vector.size();
    std::vector<double> return_vec(size,0);
    for (int i = 0;i<size;i++){
        return_vec[i] = d_reLu(input_vector[i]);
    }
    return return_vec;
}

std::vector<std::vector<double>> reLu_array(std::vector<std::vector<double>> input_array){
    int rows = input_array.size();
    int cols = input_array[0].size();
    std::vector<std::vector<double>> return_array(rows,std::vector<double>(cols,0));
    for (int i = 0;i<rows;i++){
        return_array[i] = reLu_list(input_array[i]);
    }
    return return_array;
}

std::vector<std::vector<double>> d_reLu_array(std::vector<std::vector<double>> input_array){
    int rows = input_array.size();
    int cols = input_array[0].size();
    std::vector<std::vector<double>> return_array(rows,std::vector<double>(cols,0));
    for (int i = 0;i<rows;i++){
        return_array[i] = d_reLu_list(input_array[i]);
    }
    return return_array;
}
