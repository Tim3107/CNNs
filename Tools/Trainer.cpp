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

void Trainer::set_labels(std::vector<std::vector<double>> labels) {
    this->labels = labels;
}

void Trainer::start_Training() {
    int random_number = 0;
    int random_number_2 = 0;
    for (int i = 0;i<this->iteration_steps;i++) {
        random_number = rand() % 2;
        random_number_2 = rand() % 5;
        this->current_label = this->labels[random_number];
        this->current_input = this->features[random_number][random_number_2];
        this->forward_algorithm();
        this->backward_algorithm();
        this->loss =  this->compute_overall_loss();
        std::cout << "Full error is: " << this->loss << std::endl;
        if(this->loss<this->error_tolerance){
            i = this->iteration_steps;
        }
    }
}

void Trainer::setter_error_tolerance(double error_tolerance) {
    this->error_tolerance = error_tolerance;
}

void Trainer::setter_iteration_steps(int iteration_steps) {
    this->iteration_steps = iteration_steps;
}

void Trainer::setter_learning_rate(double learning_rate) {
    for (int i = 0;i<this->filter_layers.size();i++){
        filter_layers[i].setter_learning_rate(learning_rate);
    }

    for (int i = 0;i<this->fullys.size();i++){
        fullys[i].setter_learning_rate(learning_rate);
    }
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
        this->max_poolers[i] = Max_pooling(sizes[i],strides[i],paddings[i]);
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
    std::vector<std::vector<std::vector<std::vector<double>>>> feature_maps_forward_temp;
    std::vector<std::vector<std::vector<double>>> tempo = this->filter_layers[0].run_Filter_one_channel(this->current_input);
    feature_maps_forward_temp.push_back(this->max_poolers[0].run_max_pooling_3D(tempo));

    for (int i = 1;i<this->filter_layers.size();i++){
        feature_maps_forward_temp.push_back(this->max_poolers[i].run_max_pooling_3D(this->filter_layers[i].run_Filter_one_channel(feature_maps_forward_temp[i-1])));
    }
    this->feature_maps_forward = feature_maps_forward_temp;
    int a = feature_maps_forward_temp.size()-1;
    std::vector<double> convert = this->extract_vector(feature_maps_forward_temp[feature_maps_forward_temp.size()-1]);
    std::vector<std::vector<double>> fully_layers_forward_temp;
    fully_layers_forward_temp.push_back(this->fullys[0].forward_step(convert));

    for (int i = 1;i<this->fullys.size();i++){
        fully_layers_forward_temp.push_back(this->fullys[i].forward_step(fully_layers_forward_temp[i-1]));
    }
    this->fully_layers_forward = fully_layers_forward_temp;
    this->current_output = this->outputLayer.forward_step(this->fully_layers_forward[this->fullys.size()-1]);


}


void Trainer::backward_algorithm() {

    std::vector<double> output_backward = this->outputLayer.backward_step(this->current_label);
    std::vector<std::vector<double>> fully_layers_backward_temp;

    fully_layers_backward_temp.push_back(this->fullys[this->fullys.size()-1].backward_step(output_backward));

    for(int i = this->fullys.size()-2;i>=0;i--){
        fully_layers_backward_temp.push_back(this->fullys[i].backward_step(fully_layers_backward_temp[fully_layers_backward_temp.size()-1]));
    }
    this->fully_layers_backward = fully_layers_backward_temp;
    std::vector<std::vector<std::vector<std::vector<double>>>> feature_maps_backward_temp;
    feature_maps_backward_temp.push_back(this->filter_layers[this->filter_layers.size()-1].backward_step_filter_set(this->max_poolers[this->max_poolers.size()-1].backward_pooler(this->convert_to_3D_array(fully_layers_backward_temp[fully_layers_backward_temp.size()-1]))));

    for (int i = this->filter_layers.size()-2;i>=0;i--){
        feature_maps_backward_temp.push_back(this->filter_layers[i].backward_step_filter_set(this->max_poolers[i].backward_pooler(feature_maps_backward_temp[feature_maps_backward_temp.size()-1])));
    }
    this->feature_maps_backward = feature_maps_backward_temp;
}


std::vector<double> Trainer::extract_vector(std::vector<std::vector<std::vector<double>>> input) {
    int dim = input.size();
    std::vector<double> out;
    for (int i = 0;i<dim;i++){
        out.push_back(input[i][0][0]);
    }

    return out;
}

std::vector<std::vector<std::vector<double>>> Trainer::convert_to_3D_array(std::vector<double> input_vector) {
    int rows = input_vector.size();
    std::vector<std::vector<std::vector<double>>> output_array(rows,std::vector<std::vector<double>>(1,std::vector<double>(1,0)));
    for (int i = 0;i<rows;i++){
        output_array[i][0][0] = input_vector[i];
    }

    return output_array;
}

double Trainer::compute_overall_loss() {
    double loss = 0;
    for (int i = 0;i<this->classes;i++){
        for(int j = 0;j<this->features[i].size();j++){
            this->current_label = this->labels[i];
            this->current_input = features[i][j];
            this->forward_algorithm();
            loss += this->outputLayer.compute_loss(this->current_label);
        }
    }

    return loss;
}

std::vector<double> Trainer::classification(std::vector<std::vector<double>> input_image) {
    this->current_input = input_image;
    this->forward_algorithm();
    display_vector(this->current_output);
    return this->current_output;
}
