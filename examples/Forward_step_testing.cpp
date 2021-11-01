//
// Created by tim on 13.10.21.

#include <iostream>
#include "vector"
#include "../max_pooling.h"
#include "opencv2/opencv.hpp"
#include "../image_processing.h"
#include "../functions/softmax.h"
#include "../Matrix_computations.h"
#include "../image_label_pair.h"
#include "../Save_load_image_label_pairs.h"
#include "../extract_images.h"
#include "../Filter.h"
#include "../Filter_layer.h"
#include "../FCC.h"
#include "../output_layer.h"

using namespace std;

int main(){
    std::cout << "Examples are tested " << std::endl;
    Image_extractor image_extractor("/home/tim/Tim/CNNs/Pictures_Testing/","",1,1,40,40);
    Image_label_pair* imageandlabel= new Image_label_pair[1];
    image_extractor.run_extractor(imageandlabel);

    Filter filter_1(3,1,1);
    Max_pooling pooler_1(4,4,0);

    Filter_layer filter_2(3,1,1,1,3);
    //Filter filter_2(3,2,1);
    //Filter filter_3(3,2,1);
    Filter_layer filter_3(3,1,1,3,6);

    Max_pooling pooler_2(10,10,0);

    FCC fully_layer_1(6,3);

    FCC fully_layer_2(3,3);

    Output_layer out_layer(3,2,"cross_entropy_loss","softmax");

    std::vector<std::vector<double>> x = filter_1.run_Filter(imageandlabel[0].get_Image());
    std::cout << x.size()<<" "<< x[0].size()<<" "<<"Hier Sizes"<< "initial" <<std::endl;
    cv::Mat m = conversion_to_Mat(x);

    //cv::imshow("m",m);

    x = pooler_1.run_max_pooling(x);

    cv::Mat n = conversion_to_Mat(x);

    //cv::imshow("n",n);

    //cv::waitKey(0);
    std::cout << x.size()<<" "<< x[0].size()<<" " <<"Hier Sizes"<< "x" <<std::endl;
    std::vector<std::vector<std::vector<double>>> xx = filter_2.run_Filter_one_channel(x);
    std::cout << xx.size()<<" "<< xx[0].size()<<" "<<xx[0][0].size()<<"Hier Sizes"<< "xx" <<std::endl;
    std::vector<std::vector<std::vector<double>>> xxx = filter_3.run_Filter_one_channel(xx);


    std::cout << xxx.size()<<" "<< xxx[0].size()<<" "<<xxx[0][0].size()<<"Hier Sizes"<< "xxx" <<std::endl;


    std::vector<std::vector<std::vector<double>>> xxxx = pooler_2.run_max_pooling_3D(xxx);


    std::cout << xxxx.size()<<" "<< xxxx[0].size()<<" "<<xxxx[0][0].size()<<"Hier Sizes"<< "xxxx" <<std::endl;

    std::vector<double> fully_1 = fully_layer_1.forward_step(xxxx);
    for (int i = 0;i<fully_1.size();i++){
        std::cout << fully_1[i]<<" ";
    }
    std::vector<double> fully_2 = fully_layer_2.forward_step(fully_1);

    for (int i = 0;i<fully_2.size();i++){
        std::cout << fully_2[i]<<" ";
    }
    return 0;
};