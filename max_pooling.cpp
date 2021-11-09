//
// Created by tim on 01.10.21.
//

#include "max_pooling.h"

Max_pooling::Max_pooling(int dim_Filter, int stride, int padding) {
    this->dim_Filter = dim_Filter;
    this->stride = stride;
    this->padding = padding;
}

std::vector<std::vector<double>> Max_pooling::run_max_pooling(std::vector<std::vector<double>> input_image, int channel) {
    int rows = input_image.size();
    int cols = input_image[0].size();

    int new_rows = (rows - dim_Filter + 2 * padding) / stride + 1;
    int new_cols = (cols - dim_Filter + 2 * padding) / stride + 1;

    std::vector<std::vector<double>> temp = apply_padding(input_image);
    std::vector<std::vector<double>> output_array(new_rows, std::vector<double>(new_cols, 0));
    int index_1 = 0;
    int index_2 = 0;
    double temps = 0;

    for (int i = 0; i < new_rows; i++) {
        for (int j = 0; j < new_cols; j++) {
            double max_value = 0;
            for (int inner_i = 0; inner_i < dim_Filter; inner_i++) {
                for (int inner_j = 0; inner_j < dim_Filter; inner_j++) {
                    temps = temp[i*this->stride+inner_i][j*this->stride+inner_j];
                    if(max_value<temps){
                        max_value = temps;
                        this->index_x = i*this->stride+inner_i - this->padding;
                        this->index_y = j*this->stride+inner_j - this->padding;

                    }
                }
            }
            output_array[i][j] = max_value;
            this->max_array_3D[channel][index_x][index_y] = 1;
            index_1 = 0;
            index_2 = 0;
        }
    }
    std::cout <<"---------" << std::endl;
    display_array(this->max_array_3D[channel]);

    return output_array;

}

std::vector<std::vector<std::vector<double>>> Max_pooling::run_max_pooling_3D(std::vector<std::vector<std::vector<double>>> input_image){
    int channels = input_image.size();
    int rows = input_image[0].size();
    int cols = input_image[0][0].size();

    std::vector<std::vector<std::vector<int>>> kkk(channels,std::vector<std::vector<int>>(rows,std::vector<int>(cols,0)));
    this->max_array_3D = kkk;

    int new_rows = (rows - this->dim_Filter + 2 * this->padding) / this->stride + 1;
    int new_cols = (cols - this->dim_Filter + 2 * this->padding) / this->stride + 1;

    std::vector<std::vector<std::vector<double>>> output_image(channels,std::vector<std::vector<double>>(new_rows, std::vector<double>(new_cols, 0)));

    for (int i = 0; i<channels;i++){
        output_image[i] = this->run_max_pooling(input_image[i],i);
    }
    return output_image;
}

std::vector<std::vector<std::vector<double>>> Max_pooling::backward_pooler(std::vector<std::vector<std::vector<double>>> input_successor){
    int channels = input_successor.size();
    int rows = input_successor[0].size();
    int cols = input_successor[0][0].size();

    int channels_test = this->max_array_3D.size();
    int rows_test = this->max_array_3D[0].size();
    int cols_test = this->max_array_3D[0][0].size();

    std::cout << "-----------------"<< std ::endl;

    assert(channels == channels_test);
    assert(this->stride*rows == rows_test);
    assert(this->stride*cols == cols_test);

    std::vector<std::vector<std::vector<double>>> return_array(channels,std::vector<std::vector<double>>(rows,std::vector<double>(cols,0)));
    for (int i = 0;i<channels;i++){
        return_array[i] = this->backward_pooler_2D(input_successor[i],i);
    }
    return return_array;
}

std::vector<std::vector<double>> Max_pooling::backward_pooler_2D(std::vector<std::vector<double>> input_successor_2D,int channel){
    int rows = input_successor_2D.size();
    int cols = input_successor_2D[0].size();

    std::vector<std::vector<double>> output_array(this->stride*rows,std::vector<double>(this->stride*cols,0));
    for (int i = 0;i<rows;i++){
        for (int j = 0;j<cols;j++){
            for (int inner_i = 0;inner_i<this->stride;inner_i++){
                for (int inner_j = 0;inner_j<this->stride;inner_j++){
                    output_array[stride*i+inner_i][stride*j+inner_j] = this->max_array_3D[channel][stride*i+inner_i][stride*j+inner_j]*input_successor_2D[i][j];
                }
            }
        }
    }
    return output_array;
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