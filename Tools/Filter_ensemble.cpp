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
    this->initialize_bias();
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

    std::vector<std::vector<double>> output_filtered_image(new_rows,std::vector<double>(new_cols,this->bias));
    for (int i = 0;i<this->anzahl_filter;i++){
        output_filtered_image = add_matrices(output_filtered_image,this->filters[i].run_Filter(image_array[i]));
    }

    this->temp_affin = output_filtered_image;

    for (int i = 0;i<new_rows;i++){
        for (int j = 0;j<new_cols;j++){
            if(this->activation_function == "sigmoid") {
                output_filtered_image[i][j] = sigmoid(output_filtered_image[i][j]);
            }
            else if(this->activation_function == "relu") {
                output_filtered_image[i][j] = reLu(output_filtered_image[i][j]);
            }
        }
    }
    return output_filtered_image;
}

std::vector<std::vector<double>> Filter_set::run_Filters(std::vector<std::vector<double>> image_array) {
    int rows = image_array.size();
    int cols = image_array[0].size();

    int new_rows = (rows - this->dim_filters + 2 * padding) / stride + 1;
    int new_cols = (cols - this->dim_filters + 2 * padding) / stride + 1;

    std::vector<std::vector<double>> output_filtered_image(new_rows,std::vector<double>(new_cols,this->bias));
    output_filtered_image = add_matrices(output_filtered_image,this->filters[0].run_Filter(image_array));

    this->temp_affin = output_filtered_image;

    for (int i = 0;i<new_rows;i++){
        for (int j = 0;j<new_cols;j++){
            if(this->activation_function == "sigmoid") {
                output_filtered_image[i][j] = sigmoid(output_filtered_image[i][j]);
            }
            else if(this->activation_function == "relu") {
                output_filtered_image[i][j] = reLu(output_filtered_image[i][j]);
            }
        }
    }

    return output_filtered_image;
}

std::vector<std::vector<double>> Filter_set::backward_step_filter_set(std::vector<std::vector<double>> input_successor,int index_of_channel){

    int rows = input_successor.size();
    int cols = input_successor[0].size();

    int new_rows = this->temp_affin.size();
    int new_cols = this->temp_affin[0].size();


    std::vector<std::vector<double>> return_gradients(new_rows, std::vector<double>(new_cols, 0));
    std::vector<std::vector<double>> temp;
    if(this->activation_function == "relu") {
        temp = matrix_multiplication_elementwise(input_successor, d_reLu_array(this->temp_affin),1.0);
    }
    else if (this->activation_function == "sigmoid"){
        temp = matrix_multiplication_elementwise(input_successor, d_sigmoid_array(this->temp_affin),1.0);
    }
    //std::cout <<"----b----"<< std ::endl;
    //display_array(temp);
    //std::cout <<"----b----"<< std ::endl;

    return_gradients = filters[index_of_channel].run_Filter_flipped(temp);

    return return_gradients;
}


void Filter_set::backward_step_update_filters(std::vector<std::vector<double>> input,std::vector<std::vector<double>> input_successor,int filter_index){
    std::vector<std::vector<double>> temp;
    if(this->activation_function == "relu"){
        temp = matrix_multiplication_elementwise(input_successor, d_reLu_array(this->temp_affin),1.0);
    }
    else if (this->activation_function == "sigmoid"){
        temp = matrix_multiplication_elementwise(input_successor, d_sigmoid_array(this->temp_affin),1.0);
    }

    std::vector<std::vector<double>> filter_grad;

    filter_grad = Filter_2D(this->padding,this->stride,input,temp);
    this->filters[filter_index].setter_learning_rate(learning_rate);
    this->filters[filter_index].update_filter(filter_grad);

    double grad_bias = 0;
    for(int i = 0;i<temp.size();i++){
        for(int j = 0;j<temp[0].size();j++){
            grad_bias += temp[i][j];
        }
    }

    //this->bias -= this->learning_rate*grad_bias;
    //display_array(temp_affin);
    //std::cout<< grad_bias<< std::endl;
}


void Filter_set::setter_learning_rate(double learning_rate){
    this->learning_rate = learning_rate;
}

void Filter_set::initialize_bias() {
    srand((double) time(NULL));
    static std::random_device generator;
    std::uniform_real_distribution<double> distribution (-0.3,0.3);
    this->bias = 0;//distribution(generator);
}
