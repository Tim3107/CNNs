//
// Created by tim on 08.11.21.
//

#ifndef CNNS_TRAINER_H
#define CNNS_TRAINER_H

#include "vector"
#include "iostream"
#include "string"
#include "../Tools/max_pooling.h"
#include "../Tools/Filter_layer.h"
#include "../Tools/Matrix_computations.h"
#include "../Tools/FCC.h"
#include "../Tools/image_processing.h"
#include "../Tools/output_layer.h"






class Trainer{

    std::string path_images;
    std::vector<std::string> suffix_features;
    std::vector<std::vector<int>> indices_of_images;
    int image_size;
    int classes;
    double loss;
    int iteration_steps;
    double error_tolerance;
    Output_layer outputLayer;
    std::vector<Filter_layer> filter_layers;
    std::vector<Max_pooling> max_poolers;
    std::vector<FCC> fullys;
    std::vector<std::vector<double>> current_input;
    std::vector<double> current_label;
    std::vector<double> current_output;
    std::vector<std::vector<std::vector<std::vector<double>>>> feature_maps_forward;
    std::vector<std::vector<double>> fully_layers_forward;
    std::vector<std::vector<std::vector<std::vector<double>>>> feature_maps_backward;
    std::vector<std::vector<double>> fully_layers_backward;
    std::vector<std::vector<std::vector<std::vector<double>>>> features;
    std::vector<std::vector<double>> labels;


private:

public:


    Trainer(std::string path_images,std::vector<std::string> suffix_features,std::vector<std::vector<int>> indices_of_images,
            int image_size, int classes,int iteration_steps,double error_tolerance,Output_layer outputLayer);

    void get_features();

    void set_labels(std::vector<std::vector<double>> labels);

    void start_Training();

    void setter_error_tolerance(double error_tolerance);

    void setter_iteration_steps(int iteration_steps);

    void setter_learning_rate(double learning_rate);

    void initialize_filter_layers(std::vector<int> sizes,std::vector<int> paddings,std::vector<int> strides, std::vector<std::string> activation_functions, std::vector<int> input_channels, std::vector<int> output_channels) ;

    void initialize_max_poolers(std::vector<int> sizes,std::vector<int> paddings,std::vector<int> strides);

    void initialize_fullys(std::vector<int> input,std::vector<int> output,std::vector<std::string> activation_functions);

    void forward_algorithm();

    void backward_algorithm();

    std::vector<double> extract_vector(std::vector<std::vector<std::vector<double>>> intput);

    std::vector<std::vector<std::vector<double>>> convert_to_3D_array(std::vector<double> input_vector);

    double compute_overall_loss();

    std::vector<double> classification(std::vector<std::vector<double>> input_image);


};

#endif //CNNS_TRAINER_H
