//
// Created by tim on 04.11.21.
//

#include "MSE.h"



double mse_compute_error(std::vector<double> output,std::vector<double> target_label){
    //display_vector(output);
    //display_vector(target_label);
    int rows = target_label.size();
    int rows_test = output.size();
    //std::cout << rows << std::endl;
    //std::cout << rows_test << std::endl;

    assert(rows==rows_test);
    double mse = 0;

    for (int i = 0; i<rows;i++){
        mse += (output[i]-target_label[i])*(output[i]-target_label[i]);
    }

    return 0.5*mse;
}