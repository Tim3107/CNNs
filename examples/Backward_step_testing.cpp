//
// Created by tim on 03.11.21.
//

#include "vector"
#include "../Matrix_computations.h"
#include "../FCC.h"
#include "../output_layer.h"
#include "../image_label_pair.h"
#include "../image_processing.h"

double kkkk(std::vector<std::vector<double>> features, std::vector<std::vector<double>> labels,FCC fully_1,FCC fully_2,FCC fully_3,Output_layer outputLayer){
    double  loss = 0;
    for (int i = 0;i<6;i++) {

        std::vector<double> temp1 = fully_1.forward_step(features[i]);
        std::vector<double> temp2 = fully_2.forward_step(temp1);
        std::vector<double> temp3 = fully_3.forward_step(temp2);
        std::vector<double> vect = outputLayer.forward_step(temp3);

        loss += outputLayer.compute_loss(labels[i], vect);
    }
    return loss;
}

int main(){

    srand((double)time(NULL));
    FCC fully_1(3,6,"relu");
    fully_1.setter_learning_rate(0.01);
    FCC fully_2(6,3,"relu");
    fully_2.setter_learning_rate(0.01);
    FCC fully_3(3,3,"relu");
    fully_3.setter_learning_rate(0.01);
    Output_layer outputLayer(3,3,"MSE","none");

    int random_number = 0;

    display_array(fully_1.getter_matrix());


    std::vector<double> feature_1 = {0.,0.,0.};
    std::vector<double> feature_2 = {1.,0.,0.};
    std::vector<double> feature_3 = {0.,1.,0.};
    std::vector<double> feature_4 = {1.,1.,0.};
    std::vector<double> feature_5 = {0.,1.,1.};
    std::vector<double> feature_6 = {1.,1.,1.};


    std::vector<double> label_1 = {1.,1.,1.};
    std::vector<double> label_2 = {0.,1.,1.};
    std::vector<double> label_3 = {1.,0.,1.};
    std::vector<double> label_4 = {0.,0.,1.};
    std::vector<double> label_5 = {1.,0.,0.};
    std::vector<double> label_6 = {0.,0.,0.};

    std::vector<std::vector<double>> features = {feature_1,feature_2,feature_3,feature_4,feature_5,feature_6};
    std::vector<std::vector<double>> labels = {label_1,label_2,label_3,label_4,label_5,label_6};

    std::vector<double> temp1 = fully_1.forward_step(feature_1);
    std::vector<double> temp2 = fully_2.forward_step(temp1);
    std::vector<double> temp3 = fully_3.forward_step(temp2);
    std::vector<double> vect = outputLayer.forward_step(temp3);

    double loss = outputLayer.compute_loss(label_1,vect);

    std::cout << "loss: " <<loss << std::endl;

    std::vector<double> backprop_1 = outputLayer.backward_step(label_1);
    std::vector<double> backprop_2 = fully_3.backward_step(backprop_1);
    std::vector<double> backprop_3 = fully_2.backward_step(backprop_2);
    std::vector<double> backprop_4 = fully_1.backward_step(backprop_3);

    for (int i = 0;i<20000;i++) {
        fully_1.setter_learning_rate(0.01);
        fully_2.setter_learning_rate(0.01);
        fully_3.setter_learning_rate(0.01);
        random_number = rand() % 6;
        temp1 = fully_1.forward_step(features[random_number]);
        temp2 = fully_2.forward_step(temp1);
        temp3 = fully_3.forward_step(temp2);
        vect = outputLayer.forward_step(temp3);

        loss = outputLayer.compute_loss(labels[random_number], vect);

        std::cout << "loss: " << loss << std::endl;
        std::cout << "The overall loss is: " << kkkk(features,labels,fully_1,fully_2,fully_3,outputLayer) << std::endl;


        backprop_1 = outputLayer.backward_step(labels[random_number]);
        backprop_2 = fully_3.backward_step(backprop_1);
        backprop_3 = fully_2.backward_step(backprop_2);
        backprop_4 = fully_1.backward_step(backprop_3);
    }

    std::cout <<"Testing"<<std::endl;

    temp1 = fully_1.forward_step({0.,1.,0.});
    temp2 = fully_2.forward_step(temp1);
    temp3 = fully_3.forward_step(temp2);
    vect = outputLayer.forward_step(temp3);

    display_vector(vect);

/*
    for (int i = 0; i<6;i++){
        std::cout << "printing errors of "<<i  <<std::endl;
        temp1 = fully_1.forward_step(features[i]);
        temp2 = fully_2.forward_step(temp1);
        temp3 = fully_3.forward_step(temp2);
        vect = outputLayer.forward_step(temp3);

        std::cout << " "  << std::endl;
        std::cout << " "  << std::endl;
        std::cout << " " << std::endl;

        display_vector(vect);
        std::cout << " "  << std::endl;
        std::cout << " "  << std::endl;
        std::cout << " "  << std::endl;

        std::cout << " "  << std::endl;

        loss = outputLayer.compute_loss(labels[i], vect);

        std::cout << "loss: " << loss << std::endl;
    }
    */

    return 0;
};

