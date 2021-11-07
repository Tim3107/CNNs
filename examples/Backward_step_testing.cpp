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
    for (int i = 0;i<4;i++) {

        std::vector<double> temp1 = fully_1.forward_step(features[i]);
        std::vector<double> temp2 = fully_2.forward_step(temp1);
        std::vector<double> temp3 = fully_3.forward_step(temp2);
        std::vector<double> vect = outputLayer.forward_step(temp3);

        loss += outputLayer.compute_loss(labels[i], vect);
    }
    return loss;
}

int main(){
    srand((unsigned int)time(NULL));
    FCC fully_1(3,4);
    FCC fully_2(4,6);
    FCC fully_3(6,3);
    Output_layer outputLayer(3,3,"MSE","none");

    int random_number = 0;

    display_array(fully_1.getter_matrix());


    std::vector<double> feature_1 = {{0.,0.,0.}};
    std::vector<double> feature_2 = {{1.,0.,0.}};
    std::vector<double> feature_3 = {{0.,1.,0.}};
    std::vector<double> feature_4 = {{1.,1.,0.}};
    std::vector<double> feature_5 = {{0.,1.,1.}};
    std::vector<double> feature_6 = {{1.,1.,1.}};

    std::vector<double> label_1 = {1.,1.,1.};
    std::vector<double> label_2 = {0.,0.,0.};
    std::vector<double> label_3 = {0.,1.,0.};
    std::vector<double> label_4 = {0.,0.,1.};
    std::vector<double> label_5 = {1.,0.,0.};
    std::vector<double> label_6 = {0.,0.,0.};

    std::vector<std::vector<double>> features = {feature_1,feature_2,feature_3,feature_4,feature_5,feature_6};
    std::vector<std::vector<double>> labels = {label_1,label_2,label_3,label_4,label_5,label_6};

    std::vector<double> temp1 = fully_1.forward_step(feature_1);
    std::vector<double> temp2 = fully_2.forward_step(temp1);
    std::vector<double> temp3 = fully_3.forward_step(temp2);
    std::vector<double> vect = outputLayer.forward_step(temp2);

    double loss = outputLayer.compute_loss(label_1,vect);

    std::cout << "loss: " <<loss << std::endl;

    std::vector<double> backprop_1 = outputLayer.backward_step(label_1);
    std::vector<double> backprop_2 = fully_2.backward_step(backprop_1);
    std::vector<double> backprop_3 = fully_2.backward_step(backprop_2);
    std::vector<double> backprop_4 = fully_1.backward_step(backprop_3);

    for (int i = 0;i<100;i++) {
        fully_1.setter_learning_rate(1);
        fully_2.setter_learning_rate(1);
        random_number = rand() % 4;
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

    return 0;
};

