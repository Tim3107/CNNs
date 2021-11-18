//
// Created by tim on 05.11.21.
//

#include "vector"
#include "../Tools/Matrix_computations.h"
#include "../Tools/FCC.h"
#include "../Tools/output_layer.h"
#include "../Tools/image_label_pair.h"
#include "../Tools/image_processing.h"
#include <stdlib.h>
#include "random"

double kkkk(std::vector<std::vector<double>> features, std::vector<std::vector<double>> labels, FCC fully_1, Output_layer outputLayer){
    double  loss = 0;
    for (int i = 0;i<4;i++) {
        std::vector<double> temp1 = fully_1.forward_step(features[i]);
        std::vector<double> out = outputLayer.forward_step(temp1);


        loss += outputLayer.compute_loss(labels[i], out);

    }
    return loss;
}

int main(){


    int random_number = 0;
    std::vector<double> feature_1 = {0,0};
    std::vector<double> feature_2 = {1,0};
    std::vector<double> feature_3 = {0,1};
    std::vector<double> feature_4 = {1,1};


    std::vector<std::vector<double>> features = {feature_1,feature_2,feature_3,feature_4};

    FCC fully_1(2,2,"sigmoid",{{0.1,0.5},{0.1,-0.3}},{0.8,0.5});
    //FCC fully_1(2,2,"sigmoid");
    fully_1.setter_learning_rate(1);

    Output_layer outputLayer(2,2,"MSE","none");

    std::vector<double> temp1 = fully_1.forward_step(feature_4);
    std::vector<double> out = outputLayer.forward_step(temp1);
    double loss = outputLayer.compute_loss(feature_4,out);

    std::cout << "loss: " <<loss << std::endl;
    display_vector(out);

    std::vector<double> backprop_1 = outputLayer.backward_step(feature_4);
    std::vector<double> backprop_2 = fully_1.backward_step(backprop_1);

    for (int i = 0;i<25000;i++) {
        fully_1.setter_learning_rate(10);
        random_number = rand() % 4;
        temp1 = fully_1.forward_step(features[random_number]);
        out = outputLayer.forward_step(temp1);


        loss = outputLayer.compute_loss(features[random_number], out);

        //std::cout << "current loss: " << loss << std::endl;
        //display_vector(out);
        std::cout << "The overall loss is: " << kkkk(features,features,fully_1,outputLayer) << std::endl;


        backprop_1 = outputLayer.backward_step(features[random_number]);
        backprop_2 = fully_1.backward_step(backprop_1);
    }

    std::cout << "Displaying Arrays + bias and stuff:" << std::endl;


    display_array(fully_1.getter_matrix());
    display_vector(fully_1.getter_vector());


    std::cout << "printing errors" <<std::endl;

    std::cout << " " <<std::endl;

    display_vector(fully_1.forward_step(feature_1));

    std::cout << " " <<std::endl;

    display_vector(fully_1.forward_step(feature_2));

    std::cout << " " <<std::endl;

    display_vector(fully_1.forward_step(feature_3));

    std::cout << " " <<std::endl;

    display_vector(fully_1.forward_step(feature_4));

    std::cout << " " <<std::endl;


    return 0;
}

