//
// Created by tim on 11.10.21.
//

#include "Save_load_image_label_pairs.h"


void save_image_label_pairs(std::string filename, Image_label_pair* image_label_object) {

    std::ofstream output_file(filename + ".txt");

    int rows = image_label_object->get_Image().size();
    int cols = image_label_object->get_Image()[0].size();

    output_file << std::to_string(rows) << std::endl;
    output_file << std::to_string(cols) << std::endl;

    output_file << image_label_object->get_Label()<< std::endl;

    for (int i = 0; i<rows; i++){
        for (int j = 0; j<cols;j++){
           output_file << " ";
            output_file << std::to_string(int (image_label_object->get_Image()[i][j]));
        }
        output_file << " " << std::endl;
    }
    output_file.close();


    //std::ofstream output_file(filename + ".txt");
    //output_file << std::setprecision(10);
    //for (int i = 0; i < rows; i++) {
    //    for (auto const &x: image_label_object->get_Image()[0]){
    //        output_file << x;
      //  }
    //    output_file <<""<<std::endl;

    //}

}

Image_label_pair load_image_label_pairs(std::string filename){
    std::string label;
    std::vector<std::vector<double>> output_array;
    std::vector<double> temp;
    int rows;
    int cols;
    std::ifstream input_file(filename+".txt");
    std::string line;
    int i = 0;
    while (std::getline(input_file,line)){
        if(i == 0){
            rows = std::stoi(line);
        }
        else if(i==1){
            cols = std::stoi(line);
        }
        else if(i==2){
            label = line;
            output_array = std::vector<std::vector<double>>(rows,std::vector<double> (cols ,0));
        }

        else{
            std::istringstream iss(line);
            std::vector<std::string> results((std::istream_iterator<std::string>(iss)),std::istream_iterator<std::string>());
            temp = getVertexIndices(line);
            output_array[i-3] = temp;
        }
        i++;
    }
    return Image_label_pair(output_array,label);
}