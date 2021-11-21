//
// Created by tim on 08.11.21.
//

#include "Trainer.h"



Trainer::Trainer(std::string path_images, std::vector<std::string> suffix_features,
                 std::vector<std::vector<int>> indices_of_images, int image_size, int classes, int iteration_steps,
                 double error_tolerance,Output_layer outputLayer) {
    this->path_images = path_images;
    this->suffix_features = suffix_features;
    this->indices_of_images = indices_of_images;
    this->image_size = image_size;
    this->classes = classes;
    this->iteration_steps = iteration_steps;
    this->error_tolerance = error_tolerance;
    this->outputLayer = outputLayer;
}

void Trainer::get_features() {
    cv::Mat grey_image;
    cv::Mat grey_feature;
    std::vector<std::vector<double>> feature;
    for (int i = 0;i<this->classes;i++){
        std::vector<std::vector<std::vector<double>>> temp(this->indices_of_images[i][1]-this->indices_of_images[i][0]+1,std::vector<std::vector<double>>(this->image_size,std::vector<double>(this->image_size,0)));
        for (int j = this->indices_of_images[i][0];j<=this->indices_of_images[i][1];j++) {
            cv::Mat image = imread(this->path_images + this->suffix_features[i] + std::to_string(j)+".png", cv::IMREAD_COLOR);
            //std::string test = this->path_images + this->suffix_features[i] + std::to_string(j)+".png";
            cvtColor(image, grey_image, cv::COLOR_BGR2GRAY);
            resize(grey_image, grey_feature, cv::Size(this->image_size, this->image_size), cv::INTER_LINEAR);
            feature = conversion_to_std_vector(grey_feature);
            //cv::Mat tests = conversion_to_Mat(feature);
            temp[j - this->indices_of_images[i][0]] = feature;
        }
        this->features.push_back(temp);
    }


    /*
    for (int i = 0;i<this->classes;i++){
        for (int j = this->indices_of_images[i][0];j<=this->indices_of_images[i][1];j++) {
            cv::Mat tests = conversion_to_Mat(this->features[i][j-this->indices_of_images[i][0]]);
            cv::imshow(std::to_string(i)+std::to_string(j),tests);
        }
    }
     */

}

void Trainer::setter_error_tolerance(double error_tolerance) {
    this->error_tolerance = error_tolerance;
}

void Trainer::setter_iteration_steps(int iteration_steps) {
    this->iteration_steps = iteration_steps;
}

void Trainer::initialize_filter_layers(std::vector<int> sizes, std::vector<int> paddings, std::vector<int> strides,
                                       std::vector<std::string> activation_functions,std::vector<int> input_channels, std::vector<int> output_channels) {
    int number_of_filter_layers = paddings.size();

    this->filter_layers = std::vector<Filter_layer>(number_of_filter_layers);

    for (int i = 0;i<number_of_filter_layers;i++){
        this->filter_layers[i] = Filter_layer(sizes[i],paddings[i],strides[i],activation_functions[i],input_channels[i],output_channels[i]);
    }
}

void Trainer::initialize_max_poolers(std::vector<int> sizes, std::vector<int> paddings, std::vector<int> strides) {
    int number_of_max_poolers = paddings.size();

    this->max_poolers = std::vector<Max_pooling>(number_of_max_poolers);

    for (int i = 0;i<number_of_max_poolers;i++){
        this->max_poolers[i] = Max_pooling(sizes[i],paddings[i],strides[i]);
    }
}

void Trainer::initialize_fullys(std::vector<int> input, std::vector<int> output,
                                std::vector<std::string> activation_functions) {
    int number_of_fullys = input.size();

    this->fullys = std::vector<FCC>(number_of_fullys);

    for (int i = 0;i<number_of_fullys;i++){
        this->fullys[i] = FCC(input[i],output[i],activation_functions[i]);
    }

}

void Trainer::forward_algorithm() {

}
