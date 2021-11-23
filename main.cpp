#include <iostream>
#include "Tools/Filter.h"
#include "vector"
#include "Tools/max_pooling.h"
#include "opencv2/opencv.hpp"
#include "Tools/image_processing.h"
#include "functions/softmax.h"
#include "Tools/Matrix_computations.h"
#include "Tools/image_label_pair.h"
#include "Tools/Save_load_image_label_pairs.h"
#include "Tools/extract_images.h"
#include "Tools/FCC.h"
#include "Tools/Filter_layer.h"
#include "Tools/Filter_Operations.h"
#include "Tools/Trainer.h"

using namespace cv;

int main() {
    Mat image = imread("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/7.jpeg",IMREAD_COLOR);
    Mat grey_image;
    imshow("keks",image);
    cvtColor(image,grey_image,COLOR_BGR2GRAY);

    imshow("grey_keksi",grey_image);
    std::vector<std::vector<double>> test_vector = conversion_to_std_vector(grey_image);

    Mat return_array = conversion_to_Mat(test_vector);


    std::vector<std::vector<double>> y_grad = {
            {-1,-1,-1},
            {-1,8,-1},
            {-1,-1,-1}
    };

    std::vector<std::vector<double>> y_grad_4 = {
            {0.15,0.15,0.15,0.15,0.15,0.15},
            {0.15,0.15,0.15,0.15,0.15,0.15},
            {0.15,0.15,0.15,0.15,0.15,0.15},
            {0.15,0.15,0.15,0.15,0.15,0.15},
            {0.15,0.15,0.15,0.15,0.15,0.15},
            {0.15,0.15,0.15,0.15,0.15,0.15},
    };

    std::vector<std::vector<double>> y_grad_2 = {
            {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
            {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
            {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
            {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
            {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
            {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
            {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
            {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
            {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
            {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
            {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
            {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
            {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
            {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
            {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
            {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
            {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
            {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
            {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
            {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}



    };

    Filter filter_grad(3,1,1,"none",y_grad);
    std::vector<std::vector<double>> test_array = conversion_to_std_vector(grey_image);
    std::vector<std::vector<double>> filtered_array = filter_grad.run_Filter(test_vector);
    Mat return_Arrray = conversion_to_Mat(filtered_array);

    imshow("filtered",return_Arrray);

    //waitKey(0);

    Mat image_2;
    resize(image,image_2,Size(100,100),INTER_LINEAR);

    Mat greyed;
    cvtColor(image_2,greyed,COLOR_BGR2GRAY);
    std::vector<std::vector<double>> small = conversion_to_std_vector(greyed);
    Image_label_pair test(small,"Testing");
    save_image_label_pairs("/home/tim/Tim/CNN_folder/CNNs/saved_image_label_pairs/coozu", &test);

    Image_label_pair testing_one = load_image_label_pairs("/home/tim/Tim/CNN_folder/CNNs/saved_image_label_pairs/coozu");

    Image_extractor image_extractor("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/","",1,6,10,10);

    Image_label_pair* labels = new Image_label_pair[6];

    image_extractor.run_extractor(labels);

    Filter filter(4,2,1);
    filter.run_Filter(test_array);

    FCC test_fully(3,2,"relu",{{1,2,1},{2,3,1}},{1,2});
    //display_vector(test_fully.forward_step({1,1,1}));


    Filter_layer layer_testing(3,1,1,"relu",3,4);

    std::vector<std::vector<std::vector<double>>> images = {{{0,1,1,0}  ,{0,0,0,1},  {0,0,0,0},  {1,0,1,0}},
                                                            {{0,0,0,0}  ,{0,0,0,0},  {2,0,0,0},{0,0,0,1}},
                                                            {{1,0,0,0} ,{0,0,0,0},  {0,0,1,0}, {1,1,1,0}}
    };

    std::vector<std::vector<std::vector<double>>> gradients = { {{0, 0, 0, 1},{0,0,1,0},{0,0,0,0},{1,0,0,0}},
                                                                {{1, 0, 0, 1},{1,0,1,0},{0,0,0,0},{1,0,0,1}},
                                                                {{0, 1, 0, 1},{0,0,1,0},{0,1,0,0},{1,0,0,0}},
                                                                {{0, 0, 0, 1},{0,0,1,0},{0,0,1,0},{1,0,1,0}}};

    //std::vector<std::vector<std::vector<double>>> output_layer = layer_testing.run_Filter_one_channel(images);


    //std::vector<std::vector<std::vector<double>>> backprop_grad = layer_testing.backward_step_filter_set(gradients);

    /*for(int i = 0;i<backprop_grad.size();i++){
        display_array(backprop_grad[i]);
        std::cout <<"--------"<< std ::endl;
    }
    */

   // display_array(matrix_multiplication_elementwise(images[0],images[0],1));


    std::vector<std::vector<double>> padding = {
            {-2,-1,-3},
            {-1,8,-1},
            {-1,-1,-1}
    };

    //std::vector<std::vector<double>> result = Filter_2D(1,1,padding,{{1,1,1},{1,1,1},{1,1,1}});

    //display_array(result);


    Output_layer outputLayer(2,2,"MSE","none");
    Trainer trainer("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/",{"circle_","pipe_"},{{1,5},{1,5}},50,2,3000,0.3,outputLayer);

    std::vector<std::vector<double>> jjas = {{1,2},{1,2,3}};

    trainer.initialize_max_poolers({10,5,1},{0,0,0},{10,5,1});
    trainer.initialize_filter_layers({3,3,3},{1,1,1},{1,1,1},{"sigmoid","sigmoid","sigmoid"},{1,4,8},{4,8,10});
    trainer.initialize_fullys({10,8},{8,2},{"relu","relu"});
    trainer.setter_learning_rate(0.05);
    trainer.get_features();
    trainer.set_labels({{1,0},{0,1}});

    trainer.start_Training();

    image = imread("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/circle_6.png", cv::IMREAD_COLOR);
    cv::Mat grey_feature;
    cvtColor(image, grey_image, cv::COLOR_BGR2GRAY);
    resize(grey_image, grey_feature, cv::Size(50,50), cv::INTER_LINEAR);
    trainer.classification(conversion_to_std_vector(grey_feature));






    return 0;
}