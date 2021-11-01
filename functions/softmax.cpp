//
// Created by tim on 05.10.21.
//

#include "softmax.h"

std::vector<double> softmax(std::vector<double> x){
    double sum_of_components = 0;
    int size = x.size();
    std::vector<double> y(size,0);
    for (int i=0; i<size; i++){
        y[i] = exp(x[i]);
    }

    for(std::vector<double>::iterator it = y.begin(); it != y.end(); ++it)
        sum_of_components += *it;

    for (int i=0; i<size; i++){
        y[i] = y[i]/sum_of_components;
    }

    return y;

}


std::vector<double> d_softmax(std::vector<double> x){

}