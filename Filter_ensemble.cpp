//
// Created by tim on 11.10.21.
//

#include "Filter_ensemble.h"
Filter_set::Filter_set()  {}

Filter_set::Filter_set(int anzahl_filter, int dim_filter, int padding, int stride,std::string activation_function) {
    this->anzahl_filter = anzahl_filter;
    this->dim_filters = dim_filter;
    this->padding = padding;
    this->stride = stride;
    this->activation_function = activation_function;

    this->filters = new Filter[this->anzahl_filter];
    for (int i = 0;i<this->anzahl_filter;i++){
        this->filters[i] = Filter(this->dim_filters,this->padding,this->stride, this->activation_function);
    }


}

std::vector<std::vector<double>> Filter_set::run_Filters(std::vector<std::vector<std::vector<double>>> image_array) {
    int rows = image_array[0].size();
    int cols = image_array[0][0].size();

    int new_rows = (rows - this->dim_filters + 2 * padding) / stride + 1;
    int new_cols = (cols - this->dim_filters + 2 * padding) / stride + 1;

    std::vector<std::vector<double>> output_filtered_image(new_rows,std::vector<double>(new_cols,0));
    for (int i = 0;i<this->anzahl_filter;i++){
        add_matrices(output_filtered_image,this->filters[i].run_Filter(image_array[i]));
    }
    return output_filtered_image;
}

std::vector<std::vector<double>> Filter_set::run_Filters(std::vector<std::vector<double>> image_array) {
    int rows = image_array.size();
    int cols = image_array[0].size();

    int new_rows = (rows - this->dim_filters + 2 * padding) / stride + 1;
    int new_cols = (cols - this->dim_filters + 2 * padding) / stride + 1;

    std::vector<std::vector<double>> output_filtered_image(new_rows,std::vector<double>(new_cols,0));
    output_filtered_image = this->filters[0].run_Filter(image_array);

    return output_filtered_image;
}
