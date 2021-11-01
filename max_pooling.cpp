//
// Created by tim on 01.10.21.
//

#include "max_pooling.h"

Max_pooling::Max_pooling(int dim_Filter, int stride, int padding) {
    this->dim_Filter = dim_Filter;
    this->stride = stride;
    this->padding = padding;
}

std::vector<std::vector<double>> Max_pooling::run_max_pooling(std::vector<std::vector<double>> input_image) {
    int rows = input_image.size();
    int cols = input_image[0].size();

    int new_rows = (rows - dim_Filter + 2 * padding) / stride + 1;
    int new_cols = (cols - dim_Filter + 2 * padding) / stride + 1;

    std::vector<std::vector<double>> temp = apply_padding(input_image);
    std::vector<std::vector<double>> output_array(new_rows, std::vector<double>(new_cols, 0));

    for (int i = 0; i < new_rows; i++) {
        for (int j = 0; j < new_cols; j++) {
            double max_value = 0;
            for (int inner_i = 0; inner_i < dim_Filter; inner_i++) {
                for (int inner_j = 0; inner_j < dim_Filter; inner_j++) {
                    max_value = std::max(max_value,temp[i*this->stride+inner_i][j*this->stride+inner_j]);
                }
            }
            output_array[i][j] = max_value;
        }
    }

    return output_array;

}

std::vector<std::vector<std::vector<double>>> Max_pooling::run_max_pooling_3D(std::vector<std::vector<std::vector<double>>> input_image){
    int channels = input_image.size();
    int rows = input_image[0].size();
    int cols = input_image[0][0].size();

    int new_rows = (rows - this->dim_Filter + 2 * this->padding) / this->stride + 1;
    int new_cols = (cols - this->dim_Filter + 2 * this->padding) / this->stride + 1;

    std::vector<std::vector<std::vector<double>>> output_image(channels,std::vector<std::vector<double>>(new_rows, std::vector<double>(new_cols, 0)));

    for (int i = 0; i<channels;i++){
        output_image[i] = this->run_max_pooling(input_image[i]);
    }
    return output_image;
}

std::vector<std::vector<double>> Max_pooling::apply_padding(std::vector<std::vector<double>> input_array) {
    int rows = input_array.size();
    int cols = input_array[0].size();
    std::vector<std::vector<double>> output_array(rows + 2 * padding, std::vector<double>(cols + 2 * padding,
                                                                                          0));           //sufficient for zero padding
    for (int i = padding; i < padding + rows; i++) {
        for (int j = padding; j < padding + cols; j++) {
            output_array[i][j] = input_array[i - padding][j - padding];
        }
    }
    return output_array;
}