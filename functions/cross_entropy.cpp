//
// Created by tim on 09.10.21.
//




/*********************
 *
 * Tooooooooodoooooooooooooooooooo
 *
 */
#include "cross_entropy.h"

using namespace std;

double cross_entropy(double x, double label){
    return -label* log(x);
}
double d_cross_entropy(double x, double label){
    return label*1/(x);
}

double cross_entropy_list(std::vector<double> input_vector, std::vector<double> target_values){
    int size = input_vector.size();
    double cross_entropy_return = 0;
    for (int i = 0;i<size;i++){
        cross_entropy_return += cross_entropy(input_vector[i],target_values[i]);
    }
    return cross_entropy_return;
}

std::vector<double> d_cross_entropy_list(std::vector<double> input_vector, std::vector<double> target_values){
    int size = input_vector.size();
    std::vector<double> return_vec(size,0);
    for (int i = 0;i<size;i++){
        return_vec[i] = d_cross_entropy(input_vector[i],target_values[i]);
    }
    return return_vec;
}