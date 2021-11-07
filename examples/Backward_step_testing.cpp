//
// Created by tim on 03.11.21.
//

#include "vector"
#include "../Matrix_computations.h"
#include "../FCC.h"
#include "../output_layer.h"
#include "../image_label_pair.h"
#include "../image_processing.h"

int main(){
    srand((unsigned int)time(NULL));
    FCC fully_1(3,4);
    FCC fully_2(4,2);
    FCC fully_3(2,2);
    Output_layer outputLayer(2,2,"cross_entropy","softmax");

    display_array(fully_1.getter_matrix());

    std::vector<std::vector<double>> feature_1 = {{0.,0.,0.}};
    std::vector<std::vector<double>> feature_2 = {{1.,0.,0.}};
    std::vector<std::vector<double>> feature_3 = {{0.,1.,0.}};
    std::vector<std::vector<double>> feature_4 = {{1.,1.,0.}};
    std::vector<std::vector<double>> feature_5 = {{0.,1.,1.}};
    std::vector<std::vector<double>> feature_6 = {{1.,1.,1.}};

    std::vector<double> label_1 = {1.,0.};
    std::vector<double> label_2 = {1.,0.};
    std::vector<double> label_3 = {1.,0.};
    std::vector<double> label_4 = {0.,1.};
    std::vector<double> label_5 = {0.,1.};
    std::vector<double> label_6 = {0.,1.};

    return 0;
};

