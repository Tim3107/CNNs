//
// Created by tim on 08.11.21.
//


#include "vector"
#include "../Matrix_computations.h"
#include "../FCC.h"
#include "../output_layer.h"
#include "../image_label_pair.h"
#include "../image_processing.h"
#include <stdlib.h>
#include "random"

double kkkk(std::vector<std::vector<double>> features, std::vector<std::vector<double>> labels, FCC fully_1,FCC fully_2,FCC fully_3, Output_layer outputLayer){
    double  loss = 0;
    for (int i = 0;i<8;i++) {

        std::vector<double> temp1 = fully_1.forward_step(features[i]);
        std::vector<double> temp2 = fully_2.forward_step(temp1);
        std::vector<double> temp3 = fully_3.forward_step(temp2);
        double out = outputLayer.forward_step(temp3)[0];
        std::vector<double> vect;
        vect.push_back(out);
        //vect[0] = out;
        loss += outputLayer.compute_loss(labels[i], vect);

    }
    return loss;
}

int main(){

    srand((unsigned int)time(NULL));

    int random_number = 0 ;

    std::vector<double> feature_1 = {1,1,0,1};
    std::vector<double> feature_2 = {1,0,0,1};
    std::vector<double> feature_3 = {0,0,0,1};
    std::vector<double> feature_4 = {0,1,0,1};
    std::vector<double> feature_5 = {0,0,0,1};
    std::vector<double> feature_6 = {1,0,0,0};
    std::vector<double> feature_7 = {2,2};
    std::vector<double> feature_8 = {3,3};

    std::vector<double> label_1 = {1};
    std::vector<double> label_2 = {3};
    std::vector<double> label_3 = {6};
    std::vector<double> label_4 = {15};
    std::vector<double> label_5 = {5};
    std::vector<double> label_6 = {12};
    std::vector<double> label_7 = {4};
    std::vector<double> label_8 = {9};

    std::vector<std::vector<double>> features = {feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7,feature_8};
    std::vector<std::vector<double>> labels = {label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8};

    FCC fully_1(2,2,"sigmoid");//,{{0.9,0.7},{1.2,0},{0,1.8}},{-0.3,-3.3,-2});
    fully_1.setter_learning_rate(0);
    FCC fully_2(2,2,"sigmoid");//,{{1.4,2,0.7},{-1.7,0.7,2.3}},{-2.4,-1.6});
    fully_2.setter_learning_rate(0);
    FCC fully_3(2,1,"sigmoid");
    fully_2.setter_learning_rate(0);
    Output_layer outputLayer(1,1,"MSE","none");


    std::vector<double> temp1 = fully_1.forward_step(feature_1);
    std::vector<double> temp2 = fully_2.forward_step(temp1);
    std::vector<double> temp3 = fully_3.forward_step(temp2);
    double out = outputLayer.forward_step(temp3)[0];
    std::vector<double> vect;
    vect.push_back(out);
    double loss = outputLayer.compute_loss(label_1,vect);

    std::cout << "loss: " <<loss << std::endl;
    //std::cout << "out: " << out  << std::endl;

    std::vector<double> backprop_1 = outputLayer.backward_step(label_1);
    std::vector<double> backprop_2 = fully_3.backward_step(backprop_1);
    std::vector<double> backprop_3 = fully_2.backward_step(backprop_2);
    std::vector<double> backprop_4 = fully_1.backward_step(backprop_3);

    for (int i = 0;i<80000;i++) {

        fully_1.setter_learning_rate(0.0001);
        fully_2.setter_learning_rate(0.0001);
        fully_3.setter_learning_rate(0.0001);
        if(i>50000){
            fully_1.setter_learning_rate(0.0001);
            fully_2.setter_learning_rate(0.0001);
            fully_3.setter_learning_rate(0.0001);
        }
        random_number = rand() % 8;
        temp1 = fully_1.forward_step(features[random_number]);
        temp2 = fully_2.forward_step(temp1);
        temp3 = fully_3.forward_step(temp2);
        out = outputLayer.forward_step(temp3)[0];

        vect[0] = out;
        loss = outputLayer.compute_loss(labels[random_number], vect);

        std::cout << "loss: " << loss << "at"<<random_number<< std::endl;

        //std::cout << "out: " << out << std::endl;
        std::cout << "The overall loss is: " << kkkk(features,labels,fully_1,fully_2,fully_3,outputLayer) <<std::endl;



        backprop_1 = outputLayer.backward_step(labels[random_number]);
        backprop_2 = fully_3.backward_step(backprop_1);
        backprop_3 = fully_2.backward_step(backprop_2);
        backprop_4 = fully_1.backward_step(backprop_3);
    }

    std::cout << "Displaying Arrays + bias and stuff:" << std::endl;

    std::cout << "fully_1" <<std::endl;
    display_array(fully_1.getter_matrix());
    display_vector(fully_1.getter_vector());

    std::cout << "fully_2" <<std::endl;
    display_array(fully_2.getter_matrix());
    display_vector(fully_2.getter_vector());

    std::cout << "printing errors" <<std::endl;

    for (int i = 0; i<8;i++){
        std::cout << "printing errors of "<<i  <<std::endl;
        temp1 = fully_1.forward_step(features[i]);
        temp2 = fully_2.forward_step(temp1);
        temp3 = fully_3.forward_step(temp2);
        vect = outputLayer.forward_step(temp3);


        loss = outputLayer.compute_loss(labels[i], vect);

        std::cout << "loss: " << loss << std::endl;
        //std::cout << "out: " << out << std::endl;
    }

    temp1 = fully_1.forward_step({3,1});
    temp2 = fully_2.forward_step(temp1);
    temp3 = fully_3.forward_step(temp2);
    vect = outputLayer.forward_step(temp3);


    loss = outputLayer.compute_loss(label_1,vect);

    std::cout << "loss: " <<loss << std::endl;
    //std::cout << "out: " << out  << std::endl;
    display_vector(vect);

    return 0;


}




