//
// Created by tim on 10.10.21.
//

#include "image_label_pair.h"

Image_label_pair::Image_label_pair() {}

Image_label_pair::Image_label_pair(std::vector<std::vector<double>> image,std::string label){

    this->image = image;
    this->label = label;

}

std::vector<std::vector<double>> Image_label_pair::get_Image() {
    return this->image;
}

std::string Image_label_pair::get_Label() {
    return this->label;
}