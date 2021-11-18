//
// Created by tim on 05.10.21.
//

#include "image_processing.h"
#include "Matrix_computations.h"
using namespace cv;


std::vector<std::vector<double>> conversion_to_std_vector(cv::Mat input_array){
    int rows = input_array.rows;
    int cols = input_array.cols;

    std::vector<std::vector<double>> output_array(rows, std::vector<double>(cols, 0));
    for(int i = 0;i<rows;i++){
        input_array.row(i).copyTo(output_array[i]);
    }

    for (int i = 0;i<rows;i++){
        for (int j = 0;j<cols;j++){
            output_array[i][j] = output_array[i][j]/150;
        }
    }
    return output_array;

}

cv::Mat1f conversion_to_Mat(std::vector<std::vector<double>> input_array){

    int rows = input_array.size();
    int cols = input_array[0].size();

    cv::Mat output_array(rows,cols,CV_64FC1);

    for (int i = 0;i<rows;i++){
        for (int j = 0; j<cols;j++){
            output_array.at<double>(i,j) = input_array[i][j]/255;
        }

    }

    return output_array;
}

void display_array(std::vector<std::vector<double>> input_array){
    int rows = input_array.size();
    int cols = input_array[0].size();
    for (int i = 0;i<rows;i++){
        for (int j = 0;j<cols;j++){
            std::cout << input_array[i][j]<< " ";
        }
        std::cout << "   "<< std::endl;
    }
}

void display_array(std::vector<std::vector<int>> input_array){
    int rows = input_array.size();
    int cols = input_array[0].size();
    for (int i = 0;i<rows;i++){
        for (int j = 0;j<cols;j++){
            std::cout << input_array[i][j]<< " ";
        }
        std::cout << "   "<< std::endl;
    }
}

void display_vector(std::vector<double> input_vector){
    int rows = input_vector.size();
    for (int i = 0;i<rows;i++){
            std::cout << input_vector[i]<< " ";
    }
}

std::vector<double> getVertexIndices(std::string const& pointLine)
{
    std::istringstream iss(pointLine);

    return std::vector<double>{
            std::istream_iterator<double>(iss),
            std::istream_iterator<double>()
    };
}