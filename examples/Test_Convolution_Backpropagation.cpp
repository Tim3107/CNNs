//
// Created by tim on 11.11.21.
//

#include "vector"
#include "../Tools/Matrix_computations.h"
#include "../Tools/FCC.h"
#include "../Tools/output_layer.h"
#include "../Tools/image_label_pair.h"
#include "../Tools/image_processing.h"
#include "../Tools/Filter_layer.h"
#include "opencv2/opencv.hpp"
#include "../Tools/max_pooling.h"

double full_error(Filter_layer filter_layer_1,Filter_layer filter_layer_2,Filter_layer filter_layer_3,Max_pooling pooler_1,Max_pooling pooler_2,Max_pooling pooler_3,FCC fully_1,FCC fully_2,Output_layer outputLayer,
                                                std::vector<std::vector<std::vector<double>>> bars,
                                                std::vector<std::vector<std::vector<double>>> circles,
                                                std::vector<std::vector<double>> label_circles,
                                                std::vector<std::vector<double>> label_bars){
    double full_loss = 0;
    std::vector<std::vector<std::vector<double>>> forward_1 = filter_layer_1.run_Filter_one_channel(circles[0]);
    std::vector<std::vector<std::vector<double>>> pooled_1 = pooler_1.run_max_pooling_3D(forward_1);
    std::vector<std::vector<std::vector<double>>> forward_2 = filter_layer_2.run_Filter_one_channel(pooled_1);
    std::vector<std::vector<std::vector<double>>> pooled_2 = pooler_2.run_max_pooling_3D(forward_2);
    std::vector<std::vector<std::vector<double>>> forward_3 = filter_layer_3.run_Filter_one_channel(pooled_2);
    std::vector<std::vector<std::vector<double>>> pooled_3 = pooler_3.run_max_pooling_3D(forward_3);
    std::vector<double> forward_4 = fully_1.forward_step(pooled_3);
    std::vector<double> forward_5 = fully_2.forward_step(forward_4);
    std::vector<double> output_end = outputLayer.forward_step(forward_5);
    double loss = outputLayer.compute_loss(label_circles[0]);
    full_loss += loss;

    for (int i = 1;i<5;i++){
        forward_1 = filter_layer_1.run_Filter_one_channel(circles[i]);
        pooled_1 = pooler_1.run_max_pooling_3D(forward_1);
        forward_2 = filter_layer_2.run_Filter_one_channel(pooled_1);
        pooled_2 = pooler_2.run_max_pooling_3D(forward_2);
        forward_3 = filter_layer_3.run_Filter_one_channel(pooled_2);
        pooled_3 = pooler_3.run_max_pooling_3D(forward_3);
        forward_4 = fully_1.forward_step(pooled_3);
        forward_5 = fully_2.forward_step(forward_4);
        output_end = outputLayer.forward_step(forward_5);
        loss = outputLayer.compute_loss(label_circles[i]);
        full_loss += loss;
    }

    for (int i = 0;i<5;i++){
        forward_1 = filter_layer_1.run_Filter_one_channel(bars[i]);
        pooled_1 = pooler_1.run_max_pooling_3D(forward_1);
        forward_2 = filter_layer_2.run_Filter_one_channel(pooled_1);
        pooled_2 = pooler_2.run_max_pooling_3D(forward_2);
        forward_3 = filter_layer_3.run_Filter_one_channel(pooled_2);
        pooled_3 = pooler_3.run_max_pooling_3D(forward_3);
        forward_4 = fully_1.forward_step(pooled_3);
        forward_5 = fully_2.forward_step(forward_4);
        output_end = outputLayer.forward_step(forward_5);
        loss = outputLayer.compute_loss(label_bars[i]);
        full_loss += loss;
    }

    return full_loss;
}

int main(){

    std::vector<double> iniial_loss;
    double lossss = 0;
    std::vector<double> iniial_step_label;
    std::vector<std::vector<double>> current_image;
    std::vector<double> curent_label;

    srand((unsigned int)time(NULL));
    int random_number = 0 ;

    cv::Mat grey_image;

    cv::Mat grey_circle_1;
    cv::Mat grey_circle_2;
    cv::Mat grey_circle_3;
    cv::Mat grey_circle_5;
    cv::Mat grey_circle_6;

    cv::Mat grey_bars_1;
    cv::Mat grey_bars_2;
    cv::Mat grey_bars_3;
    cv::Mat grey_bars_5;
    cv::Mat grey_bars_6;

    std::vector<std::vector<double>> circle_1;
    std::vector<std::vector<double>> circle_2;
    std::vector<std::vector<double>> circle_3;
    std::vector<std::vector<double>> circle_5;
    std::vector<std::vector<double>> circle_6;

    std::vector<std::vector<double>> bar_1;
    std::vector<std::vector<double>> bar_2;
    std::vector<std::vector<double>> bar_3;
    std::vector<std::vector<double>> bar_4;
    std::vector<std::vector<double>> bar_6;

    std::vector<std::vector<double>> pipe_1;
    std::vector<std::vector<double>> pipe_2;
    std::vector<std::vector<double>> pipe_3;
    std::vector<std::vector<double>> pipe_4;
    std::vector<std::vector<double>> pipe_5;

    cv::Mat image = imread("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/circle_1.png",cv::IMREAD_COLOR);
    cvtColor(image,grey_image,cv::COLOR_BGR2GRAY);
    resize(grey_image,grey_circle_1,cv::Size(50,50),cv::INTER_LINEAR);
    circle_1 = conversion_to_std_vector(grey_circle_1);

    image = imread("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/circle_2.png",cv::IMREAD_COLOR);
    cvtColor(image,grey_image,cv::COLOR_BGR2GRAY);
    resize(grey_image,grey_circle_2,cv::Size(50,50),cv::INTER_LINEAR);
    circle_2 = conversion_to_std_vector(grey_circle_2);

    image = imread("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/circle_3.png",cv::IMREAD_COLOR);
    cvtColor(image,grey_image,cv::COLOR_BGR2GRAY);
    resize(grey_image,grey_circle_3,cv::Size(50,50),cv::INTER_LINEAR);
    circle_3 = conversion_to_std_vector(grey_circle_3);

    image = imread("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/circle_5.png",cv::IMREAD_COLOR);
    cvtColor(image,grey_image,cv::COLOR_BGR2GRAY);
    resize(grey_image,grey_circle_2,cv::Size(50,50),cv::INTER_LINEAR);
    circle_5 = conversion_to_std_vector(grey_circle_2);

    image = imread("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/circle_6.png",cv::IMREAD_COLOR);
    cvtColor(image,grey_image,cv::COLOR_BGR2GRAY);
    resize(grey_image,grey_circle_3,cv::Size(50,50),cv::INTER_LINEAR);
    circle_6 = conversion_to_std_vector(grey_circle_3);

    image = imread("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/bars_1.png",cv::IMREAD_COLOR);
    cvtColor(image,grey_image,cv::COLOR_BGR2GRAY);
    resize(grey_image,grey_bars_1,cv::Size(50,50),cv::INTER_LINEAR);
    bar_1 = conversion_to_std_vector(grey_bars_1);

    image = imread("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/bars_2.png",cv::IMREAD_COLOR);
    cvtColor(image,grey_image,cv::COLOR_BGR2GRAY);
    resize(grey_image,grey_bars_2,cv::Size(50,50),cv::INTER_LINEAR);
    bar_2 = conversion_to_std_vector(grey_bars_2);

    image = imread("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/bars_3.png",cv::IMREAD_COLOR);
    cvtColor(image,grey_image,cv::COLOR_BGR2GRAY);
    resize(grey_image,grey_bars_3,cv::Size(50,50),cv::INTER_LINEAR);
    bar_3 = conversion_to_std_vector(grey_bars_3);

    image = imread("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/bars_4.png",cv::IMREAD_COLOR);
    cvtColor(image,grey_image,cv::COLOR_BGR2GRAY);
    resize(grey_image,grey_bars_2,cv::Size(50,50),cv::INTER_LINEAR);
    bar_4 = conversion_to_std_vector(grey_bars_2);

    image = imread("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/bars_6.png",cv::IMREAD_COLOR);
    cvtColor(image,grey_image,cv::COLOR_BGR2GRAY);
    resize(grey_image,grey_bars_3,cv::Size(50,50),cv::INTER_LINEAR);
    bar_6 = conversion_to_std_vector(grey_bars_3);

    image = imread("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/pipe_1.png",cv::IMREAD_COLOR);
    cvtColor(image,grey_image,cv::COLOR_BGR2GRAY);
    resize(grey_image,grey_bars_1,cv::Size(50,50),cv::INTER_LINEAR);
    pipe_1 = conversion_to_std_vector(grey_bars_1);

    image = imread("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/pipe_2.png",cv::IMREAD_COLOR);
    cvtColor(image,grey_image,cv::COLOR_BGR2GRAY);
    resize(grey_image,grey_bars_2,cv::Size(50,50),cv::INTER_LINEAR);
    pipe_2 = conversion_to_std_vector(grey_bars_2);

    image = imread("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/pipe_3.png",cv::IMREAD_COLOR);
    cvtColor(image,grey_image,cv::COLOR_BGR2GRAY);
    resize(grey_image,grey_bars_3,cv::Size(50,50),cv::INTER_LINEAR);
    pipe_3 = conversion_to_std_vector(grey_bars_3);

    image = imread("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/pipe_4.png",cv::IMREAD_COLOR);
    cvtColor(image,grey_image,cv::COLOR_BGR2GRAY);
    resize(grey_image,grey_bars_2,cv::Size(50,50),cv::INTER_LINEAR);
    pipe_4 = conversion_to_std_vector(grey_bars_2);

    image = imread("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/pipe_5.png",cv::IMREAD_COLOR);
    cvtColor(image,grey_image,cv::COLOR_BGR2GRAY);
    resize(grey_image,grey_bars_3,cv::Size(50,50),cv::INTER_LINEAR);
    pipe_5 = conversion_to_std_vector(grey_bars_3);

    std::vector<std::vector<std::vector<double>>> bars = {bar_1,bar_2,bar_3,bar_4,bar_6};
    std::vector<std::vector<std::vector<double>>> circles = {circle_1,circle_2,circle_3,circle_5,circle_6};
    std::vector<std::vector<std::vector<double>>> pipes = {pipe_1,pipe_2,pipe_3,pipe_4,pipe_5};

    std::vector<std::vector<double>> label_circles = {{1,0},{1,0},{1,0},{1,0},{1,0}};
    std::vector<std::vector<double>> label_bars = {{0,1},{0,1},{0,1},{0,1},{0,1}};
    std::vector<std::vector<double>> label_pipes = {{0,1},{0,1},{0,1},{0,1},{0,1}};

    //image = imread("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/7.jpeg",cv::IMREAD_COLOR);
    grey_image;
    //imshow("keks",image);
    //cvtColor(image,grey_image,cv::COLOR_BGR2GRAY);
    //resize(grey_image,grey_image,cv::Size(50,50),cv::INTER_LINEAR);



    //imshow("grey_keksi",grey_image);
    //std::vector<std::vector<double>> test_vector = conversion_to_std_vector(grey_image);



    Filter_layer filterLayer_1(3,1,1,"sigmoid",1,4);
    Filter_layer filterLayer_2(3,1,1,"sigmoid",4,8);
    Filter_layer filterLayer_3(3,1,1,"sigmoid",8,10);
    filterLayer_1.setter_learning_rate(0.005);
    filterLayer_2.setter_learning_rate(0.005);
    filterLayer_3.setter_learning_rate(0.005);

    Max_pooling max_pooler_1(10,10,0);
    Max_pooling max_pooler_2(5,5,0);
    Max_pooling max_pooler_3(1,1,0);

    FCC fully_1(10,8,"relu");
    FCC fully_2(8,2,"relu");
    fully_1.setter_learning_rate(0.005);
    fully_2.setter_learning_rate(0.005);
    Output_layer outputLayer(2,2,"MSE","none");
    //display_array(circles[0]);
    std::vector<std::vector<std::vector<double>>> forward_1 = filterLayer_1.run_Filter_one_channel(circles[0]);

    std::cout << "sizes"<<forward_1.size() <<" "<< forward_1[0].size() <<" "<<forward_1[0][0].size() <<" forward" <<std::endl;
    std::vector<std::vector<std::vector<double>>> pooled_1 = max_pooler_1.run_max_pooling_3D(forward_1);
    std::cout << "sizes"<<pooled_1.size() <<" "<< pooled_1[0].size() <<" "<<pooled_1[0][0].size() <<" pooled_1" <<std::endl;
    std::vector<std::vector<std::vector<double>>> forward_2 = filterLayer_2.run_Filter_one_channel(pooled_1);
    std::cout << "sizes"<<forward_2.size() <<" "<< forward_2[0].size() <<" "<<forward_2[0][0].size() <<" forward_2" <<std::endl;
    std::vector<std::vector<std::vector<double>>> pooled_2 = max_pooler_2.run_max_pooling_3D(forward_2);
    std::cout << "sizes"<<pooled_2.size() <<" "<< pooled_2[0].size() <<" "<<pooled_2[0][0].size() <<" pooled_2" <<std::endl;
    std::vector<std::vector<std::vector<double>>> forward_3 = filterLayer_3.run_Filter_one_channel(pooled_2);
    std::cout << "sizes"<<forward_3.size() <<" "<< forward_3[0].size() <<" "<<forward_3[0][0].size() <<" forward_3" <<std::endl;
    std::vector<std::vector<std::vector<double>>> pooled_3 = max_pooler_3.run_max_pooling_3D(forward_3);
    std::cout << "sizes"<<pooled_3.size() <<" "<< pooled_3[0].size() <<" "<<pooled_3[0][0].size() <<" pooled_3" <<std::endl;

    std::vector<double> forward_4 = fully_1.forward_step(pooled_3);
    std::vector<double> forward_5 = fully_2.forward_step(forward_4);
    std::vector<double> output_end = outputLayer.forward_step(forward_5);
    double loss = outputLayer.compute_loss(label_circles[0]);

    //iniial_step_label.push_back()

    display_vector(output_end);

    std::vector<double> backwards_1 = outputLayer.backward_step(label_circles[0]);
    std::vector<double> backwards_2 = fully_2.backward_step(backwards_1);
    std::vector<double> backwards_3 = fully_1.backward_step(backwards_2);
    std::vector<std::vector<std::vector<double>>> backwards_3_convert(backwards_3.size(),std::vector<std::vector<double>>(1,std::vector<double>(1,0)));
    for (int i = 0;i<backwards_3.size();i++){
        backwards_3_convert[i][0][0] = backwards_3[i];
    }
    std::vector<std::vector<std::vector<double>>> backwards_3_1 = max_pooler_3.backward_pooler(backwards_3_convert);
    std::vector<std::vector<std::vector<double>>> backwards_temp = filterLayer_3.backward_step_filter_set(backwards_3_1);
    std::vector<std::vector<std::vector<double>>> backwards_4 = max_pooler_2.backward_pooler(backwards_temp);
    std::vector<std::vector<std::vector<double>>> backwards_5 = filterLayer_2.backward_step_filter_set(backwards_4);
    std::vector<std::vector<std::vector<double>>> backwards_6 = max_pooler_1.backward_pooler(backwards_5);
    std::vector<std::vector<std::vector<double>>> backwards_7 = filterLayer_1.backward_step_filter_set(backwards_6);


    for (int i = 0; i<2000;i++){

        random_number = rand() % 10;

        if(random_number>=5){
            current_image = circles[random_number-5];
            curent_label =  label_circles[random_number-5];
        }
        else{
            current_image = pipes[random_number];
            curent_label = label_pipes[random_number];
        }


        forward_1 = filterLayer_1.run_Filter_one_channel(current_image);
        pooled_1 = max_pooler_1.run_max_pooling_3D(forward_1);
        forward_2 = filterLayer_2.run_Filter_one_channel(pooled_1);
        pooled_2 = max_pooler_2.run_max_pooling_3D(forward_2);
        forward_3 = filterLayer_3.run_Filter_one_channel(pooled_2);
        pooled_3 = max_pooler_3.run_max_pooling_3D(forward_3);

        forward_4 = fully_1.forward_step(pooled_3);
        forward_5 = fully_2.forward_step(forward_4);
        output_end = outputLayer.forward_step(forward_5);
        loss = outputLayer.compute_loss(curent_label);

        //display_vector(output_end);
        //std::cout << "Current loss is: " << loss << std::endl;

        backwards_1 = outputLayer.backward_step(curent_label);
        backwards_2 = fully_2.backward_step(backwards_1);
        backwards_3 = fully_1.backward_step(backwards_2);
        for (int i = 0;i<backwards_3.size();i++){
            backwards_3_convert[i][0][0] = backwards_3[i];
        }
        backwards_3_1 = max_pooler_3.backward_pooler(backwards_3_convert);
        backwards_temp = filterLayer_3.backward_step_filter_set(backwards_3_1);
        backwards_4 = max_pooler_2.backward_pooler(backwards_temp);
        backwards_5 = filterLayer_2.backward_step_filter_set(backwards_4);
        backwards_6 = max_pooler_1.backward_pooler(backwards_5);
        backwards_7 = filterLayer_1.backward_step_filter_set(backwards_6);
        lossss = full_error(filterLayer_1,filterLayer_2,filterLayer_3,max_pooler_1,max_pooler_2,max_pooler_3,fully_1,fully_2,outputLayer,pipes,circles,label_circles,label_pipes);
        std::cout << "overall Error is: "<<lossss<<std::endl;
        if(lossss<0.05 || abs(lossss-6.93147)<0.0001) {
            i = 100000;
        }

/*
        if(lossss < 5){
            filterLayer_1.setter_learning_rate(0.00015);
            filterLayer_2.setter_learning_rate(0.00015);
            filterLayer_3.setter_learning_rate(0.00015);

            fully_1.setter_learning_rate(0.000015);
            fully_2.setter_learning_rate(0.000015);
        }
*/

    }

    double full_loss = 0;

    for (int i = 0;i<5;i++){
        forward_1 = filterLayer_1.run_Filter_one_channel(circles[i]);
        pooled_1 = max_pooler_1.run_max_pooling_3D(forward_1);
        forward_2 = filterLayer_2.run_Filter_one_channel(pooled_1);
        pooled_2 = max_pooler_2.run_max_pooling_3D(forward_2);
        forward_3 = filterLayer_3.run_Filter_one_channel(pooled_2);
        pooled_3 = max_pooler_3.run_max_pooling_3D(forward_3);
        forward_4 = fully_1.forward_step(pooled_3);
        forward_5 = fully_2.forward_step(forward_4);
        output_end = outputLayer.forward_step(forward_5);
        loss = outputLayer.compute_loss(label_circles[i]);
        full_loss += loss;
        std::cout << "------circles--------" << i+1<< std::endl;
        display_vector(output_end);
        std::cout << loss << std::endl;
        std::cout << "..............." << i+1<< std::endl;
    }

    for (int i = 0;i<5;i++){
        forward_1 = filterLayer_1.run_Filter_one_channel(pipes[i]);
        pooled_1 = max_pooler_1.run_max_pooling_3D(forward_1);
        forward_2 = filterLayer_2.run_Filter_one_channel(pooled_1);
        pooled_2 = max_pooler_2.run_max_pooling_3D(forward_2);
        forward_3 = filterLayer_3.run_Filter_one_channel(pooled_2);
        pooled_3 = max_pooler_3.run_max_pooling_3D(forward_3);
        forward_4 = fully_1.forward_step(pooled_3);
        forward_5 = fully_2.forward_step(forward_4);
        output_end = outputLayer.forward_step(forward_5);
        loss = outputLayer.compute_loss(label_pipes[i]);
        full_loss += loss;
        std::cout << "------pipes--------" << i+1<< std::endl;
        display_vector(output_end);
        std::cout << loss << std::endl;
        std::cout << "..............." << i+1<< std::endl;
    }




    image = imread("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/circle_4.png",cv::IMREAD_COLOR);
    cvtColor(image,grey_image,cv::COLOR_BGR2GRAY);
    resize(grey_image,grey_circle_2,cv::Size(50,50),cv::INTER_LINEAR);
    std::vector<std::vector<double>> circle_4 = conversion_to_std_vector(grey_circle_2);

    forward_1 = filterLayer_1.run_Filter_one_channel(circle_4);
    pooled_1 = max_pooler_1.run_max_pooling_3D(forward_1);
    forward_2 = filterLayer_2.run_Filter_one_channel(pooled_1);
    pooled_2 = max_pooler_2.run_max_pooling_3D(forward_2);
    forward_3 = filterLayer_3.run_Filter_one_channel(pooled_2);
    pooled_3 = max_pooler_3.run_max_pooling_3D(forward_3);

    forward_4 = fully_1.forward_step(pooled_3);
    forward_5 = fully_2.forward_step(forward_4);
    output_end = outputLayer.forward_step(forward_5);

    std::cout << "output classification last circle: "<< std::endl;

    display_vector(output_end);


    image = imread("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/pipe_6.png",cv::IMREAD_COLOR);
    cvtColor(image,grey_image,cv::COLOR_BGR2GRAY);
    resize(grey_image,grey_circle_2,cv::Size(50,50),cv::INTER_LINEAR);
    std::vector<std::vector<double>> pipe_6 = conversion_to_std_vector(grey_circle_2);

    forward_1 = filterLayer_1.run_Filter_one_channel(pipe_6);
    pooled_1 = max_pooler_1.run_max_pooling_3D(forward_1);
    forward_2 = filterLayer_2.run_Filter_one_channel(pooled_1);
    pooled_2 = max_pooler_2.run_max_pooling_3D(forward_2);
    forward_3 = filterLayer_3.run_Filter_one_channel(pooled_2);
    pooled_3 = max_pooler_3.run_max_pooling_3D(forward_3);

    forward_4 = fully_1.forward_step(pooled_3);
    forward_5 = fully_2.forward_step(forward_4);
    output_end = outputLayer.forward_step(forward_5);

    std::cout << "output classification last pipe: "<< std::endl;

    display_vector(output_end);

    //cv::waitKey(0);
    return 0;
}