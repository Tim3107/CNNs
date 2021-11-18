//
// Created by tim on 04.11.21.
//

#include "vector"
#include "../Tools/Matrix_computations.h"
#include "../Tools/FCC.h"
#include "../Tools/output_layer.h"
#include "../Tools/image_label_pair.h"
#include "../Tools/image_processing.h"
#include <stdlib.h>
#include "random"
double kkkk(std::vector<std::vector<double>> features, std::vector<std::vector<double>> labels, FCC fully_1,FCC fully_2, Output_layer outputLayer){
    double  loss = 0;
    for (int i = 0;i<4;i++) {

        std::vector<double> temp1 = fully_1.forward_step(features[i]);
        std::vector<double> temp2 = fully_2.forward_step(temp1);
        double out = outputLayer.forward_step(temp2)[0];
        std::vector<double> vect;
        vect.push_back(2);
        vect[0] = out;
        loss += outputLayer.compute_loss(labels[i], vect);

    }
    return loss;
}

int main(){

    srand((unsigned int)time(NULL));
    int random_number = 0 ;

    std::vector<double> feature_1 = {0,0};
    std::vector<double> feature_2 = {1,0};
    std::vector<double> feature_3 = {0,1};
    std::vector<double> feature_4 = {1,1};

    std::vector<double> label_1 = {0};
    std::vector<double> label_2 = {1};
    std::vector<double> label_3 = {1};
    std::vector<double> label_4 = {0};

    std::vector<std::vector<double>> features = {feature_1,feature_2,feature_3,feature_4};
    std::vector<std::vector<double>> labels = {label_1,label_2,label_3,label_4};

    FCC fully_1(2,2,"relu");
    fully_1.setter_learning_rate(0.001);
    FCC fully_2(2,1,"relu");
    fully_2.setter_learning_rate(0.001);
    Output_layer outputLayer(1,1,"MSE","none");


    std::vector<double> temp1 = fully_1.forward_step(feature_1);
    std::vector<double> temp2 = fully_2.forward_step(temp1);
    double out = outputLayer.forward_step(temp2)[0];
    std::vector<double > vect;
    vect.push_back(out);
    double loss = outputLayer.compute_loss(label_1,vect);

    std::cout << "loss: " <<loss << std::endl;
    std::cout << "out: " << out  << std::endl;

    std::vector<double> backprop_1 = outputLayer.backward_step(label_1);
    std::vector<double> backprop_2 = fully_2.backward_step(backprop_1);
    std::vector<double> backprop_3 = fully_1.backward_step(backprop_2);

    for (int i = 0;i<5000;i++) {
        fully_1.setter_learning_rate(0.01);
        fully_2.setter_learning_rate(0.01);
        random_number = rand() % 4;
        temp1 = fully_1.forward_step(features[random_number]);
        temp2 = fully_2.forward_step(temp1);
        out = outputLayer.forward_step(temp2)[0];

        vect[0] = out;
        loss = outputLayer.compute_loss(labels[random_number], vect);

        std::cout << "loss: " << loss << std::endl;
        //std::cout << "out: " << out << std::endl;
        std::cout << "The overall loss is: " << kkkk(features,labels,fully_1,fully_2,outputLayer) << std::endl;


        backprop_1 = outputLayer.backward_step(labels[random_number]);
        backprop_2 = fully_2.backward_step(backprop_1);
        backprop_3 = fully_1.backward_step(backprop_2);
    }

    std::cout << "Displaying Arrays + bias and stuff:" << std::endl;

    std::cout << "fully_1" <<std::endl;
    display_array(fully_1.getter_matrix());
    display_vector(fully_1.getter_vector());

    std::cout << "fully_2" <<std::endl;
    display_array(fully_2.getter_matrix());
    display_vector(fully_2.getter_vector());

    std::cout << "printing errors" <<std::endl;

    for (int i = 0; i<4;i++){
        std::cout << "printing errors of "<<i  <<std::endl;
        temp1 = fully_1.forward_step(features[i]);
        temp2 = fully_2.forward_step(temp1);
        out = outputLayer.forward_step(temp2)[0];

        vect[0] = out;
        loss = outputLayer.compute_loss(labels[i], vect);

        std::cout << "loss: " << loss << std::endl;
        std::cout << "out: " << out << std::endl;
    }



    return 0;


}




