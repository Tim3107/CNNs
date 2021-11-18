//
// Created by tim on 11.11.21.
//

#include "Filter_Operations.h"





std::vector<std::vector<double>> Filter_2D(int padding, int stride, std::vector<std::vector<double>> input_array_1,std::vector<std::vector<double>> input_array_2){
    int rows = input_array_1.size();
    int cols = input_array_1[0].size();

    int new_rows = (rows - input_array_2.size() + 2 * padding) / stride + 1;
    int new_cols = (cols - input_array_2[0].size() + 2 * padding) / stride + 1;

    std::vector<std::vector<double>> temp = apply_padding(input_array_1,padding);

    std::vector<std::vector<double>> output_array(new_rows, std::vector<double>(new_cols, 0));

    for (int i = 0; i < new_rows; i++) {
        for (int j = 0; j < new_cols; j++) {
            for (int inner_i = 0; inner_i < input_array_2.size(); inner_i++) {
                for (int inner_j = 0; inner_j < input_array_2[0].size(); inner_j++) {
                    output_array[i][j] += temp[i*stride+inner_i][j*stride+inner_j]*input_array_2[inner_i][inner_j];
                }
            }
        }
    }

    return output_array;
}

std::vector<std::vector<double>> apply_padding(std::vector<std::vector<double>> input_array, int padding) {
    int rows = input_array.size();
    int cols = input_array[0].size();
    std::vector<std::vector<double>> output_array(rows+2*padding, std::vector<double>(cols + 2*padding, 0));           //sufficient for zero padding
    for (int i=padding;i<padding+rows;i++){
        for (int j = padding;j<padding+cols;j++){
            output_array[i][j] = input_array[i-padding][j-padding];
        }
    }
    return output_array;

}
