//
// Created by tim on 30.09.21.
//

#include "Filter.h"

Filter::Filter(){

}

Filter::Filter(int dim_Filter, int padding, int stride) {
    this->dim_Filter = dim_Filter;
    this->padding = padding;
    this->stride = stride;
    this->Filter_grid = std::vector<std::vector<double>>(dim_Filter,std::vector<double> (dim_Filter ,1)); //---> need random values
    this->fill_matrix(this->Filter_grid);
}

Filter::Filter(int dim_Filter, int padding, int stride,std::vector<std::vector<double>> Filter_map) {
    this->dim_Filter = dim_Filter;
    this->padding = padding;
    this->stride = stride;
    this->Filter_grid = Filter_map;
}

std::vector<std::vector<double>> Filter::run_Filter(std::vector<std::vector<double>> image_array) {
    int rows = image_array.size();
    int cols = image_array[0].size();

    int new_rows = (rows - dim_Filter + 2 * padding) / stride + 1;
    int new_cols = (cols - dim_Filter + 2 * padding) / stride + 1;

    std::vector<std::vector<double>> temp = apply_padding(image_array);

    std::vector<std::vector<double>> output_array(new_rows, std::vector<double>(new_cols, 0));

    for (int i = 0; i < new_rows; i++) {
        for (int j = 0; j < new_cols; j++) {
            for (int inner_i = 0; inner_i < dim_Filter; inner_i++) {
                for (int inner_j = 0; inner_j < dim_Filter; inner_j++) {
                        output_array[i][j] += temp[i*stride+inner_i][j*stride+inner_j]*Filter_grid[inner_i][inner_j];
                }
            }
        }
    }

    return output_array;
}

std::vector<std::vector<double>> Filter::apply_padding(std::vector<std::vector<double>> input_array) {
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

bool Filter::check_for_padding(int i, int j, int inner_i,int inner_j,int rows,int cols){
    int new_rows = (rows-dim_Filter+2*padding)/stride +1;
    int new_cols = (cols-dim_Filter+2*padding)/stride +1;
    if(i - (dim_Filter - 1) / 2 + inner_i<0)
        return false;
    if(j - (dim_Filter - 1) / 2 + inner_j<0)
        return false;
    if(i - (dim_Filter - 1) / 2 + inner_i>=rows)
        return false;
    if(j - (dim_Filter - 1) / 2 + inner_j>=cols)
        return false;
    return true;

}


void Filter::fill_row(std::vector<double> & row){
    std::generate(row.begin(), row.end(), [](){ return (rand()%100)/50; });
}

void Filter::fill_matrix(std::vector<std::vector<double>> & mat){
    std::for_each(mat.begin(), mat.end(), fill_row);
}


