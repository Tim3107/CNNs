//
// Created by tim on 02.10.21.
//

#include "extract_images.h"

using namespace cv;

Image_extractor::Image_extractor() {
}

Image_extractor::Image_extractor(std::string folder, std::string filename_trunk, int first_pos, int last_pos, int image_size_x, int image_size_y) {
    this->folder = folder;
    this->filename_trunk = filename_trunk;
    this->first_pos = first_pos;
    this->last_pos = last_pos;
    this->image_size_x = image_size_x;
    this->image_size_y = image_size_y;
}

void Image_extractor::run_extractor(Image_label_pair* image_label_array) {
    //image_label_array = new Image_label_pair[this->last_pos-this->first_pos+1];
    for(int i = this->first_pos;i<=this->last_pos;i++){
        //from string get openCV image
        // convert to std::double
        //create image_label_pairs and store in image_label_array
        std::string name = std::to_string(i);
        name = this->folder + this->filename_trunk + "/" + name + ".png";
        Mat image = imread(name,IMREAD_COLOR);
        Mat image_2;
        resize(image,image_2,Size(this->image_size_x,this->image_size_y),INTER_LINEAR);
        Mat grey_image;
        cvtColor(image_2,grey_image,COLOR_BGR2GRAY);
        std::vector<std::vector<double>> image_vector = conversion_to_std_vector(grey_image);
        Image_label_pair temp = Image_label_pair(image_vector,"Hi");
        image_label_array[i-this->first_pos] = temp;
    }
}

void Image_extractor::schrott() {
    std::cout << "Pfusch" << std::endl;
}