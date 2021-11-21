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

    for (int i = 0; i<6; i++){

        display_array(labels[i].get_Image());
        std::cout << labels[i].get_Label() << std::endl;

    }

    Filter filter(4,2,1);
    filter.run_Filter(test_array);

    //


    Max_pooling max_pooler_Testing(2,2,0);

    std::vector<std::vector<std::vector<double>>> test_array_pooler = {{{200,3,5,3}  ,{2,3,6,4},  {2,4,5,3},  {1,9,17,3}},
                                                                       {{2,3,5,3}  ,{2,3,6,4},  {2,14,15,3},{1,9,7,3}},
                                                                       {{12,3,5,3} ,{2,3,6,4},  {2,4,15,3}, {1,19,17,3}},
                                                                       {{2,3,15,3} ,{2,3,6,4},  {2,14,5,3}, {1,9,7,13}}};

    std::vector<std::vector<std::vector<double>>> Testoutput = max_pooler_Testing.run_max_pooling_3D(test_array_pooler);

    std::vector<std::vector<std::vector<double>>> Testeror = max_pooler_Testing.backward_pooler(Testoutput);

    for(int i = 0;i<Testeror.size();i++){
        display_array(Testeror[i]);
        std::cout <<"--------"<< std ::endl;
    }
    for(int i = 0;i<Testoutput.size();i++){
        display_array(Testoutput[i]);
        std::cout <<"--------"<< std ::endl;
    }

    FCC test_fully(3,2,"relu",{{1,2,1},{2,3,1}},{1,2});
    display_vector(test_fully.forward_step({1,1,1}));


    Filter_layer layer_testing(3,1,1,"relu",3,4);

    std::vector<std::vector<std::vector<double>>> images = {{{0,1,1,0}  ,{0,0,0,1},  {0,0,0,0},  {1,0,1,0}},
                                                            {{0,0,0,0}  ,{0,0,0,0},  {2,0,0,0},{0,0,0,1}},
                                                            {{1,0,0,0} ,{0,0,0,0},  {0,0,1,0}, {1,1,1,0}}
    };

    std::vector<std::vector<std::vector<double>>> gradients = { {{0, 0, 0, 1},{0,0,1,0},{0,0,0,0},{1,0,0,0}},
                                                                {{1, 0, 0, 1},{1,0,1,0},{0,0,0,0},{1,0,0,1}},
                                                                {{0, 1, 0, 1},{0,0,1,0},{0,1,0,0},{1,0,0,0}},
                                                                {{0, 0, 0, 1},{0,0,1,0},{0,0,1,0},{1,0,1,0}}};

    std::vector<std::vector<std::vector<double>>> output_layer = layer_testing.run_Filter_one_channel(images);

    for(int i = 0;i<output_layer.size();i++){
        display_array(output_layer[i]);
        std::cout <<"--------"<< std ::endl;
    }

    std::vector<std::vector<std::vector<double>>> backprop_grad = layer_testing.backward_step_filter_set(gradients);

    for(int i = 0;i<backprop_grad.size();i++){
        display_array(backprop_grad[i]);
        std::cout <<"--------"<< std ::endl;
    }


    display_array(matrix_multiplication_elementwise(images[0],images[0],1));


    std::vector<std::vector<double>> padding = {
            {-2,-1,-3},
            {-1,8,-1},
            {-1,-1,-1}
    };

    std::vector<std::vector<double>> result = Filter_2D(1,1,padding,{{1,1,1},{1,1,1},{1,1,1}});

    display_array(result);


    Output_layer outputLayer(2,2,"MSE","none");
    Trainer trainer("/home/tim/Tim/CNN_folder/CNNs/Pictures_Testing/",{"bars_","circle_","pipe_"},{{1,4},{1,5},{1,6}},400,3,1000,0.1,outputLayer);

    std::vector<std::vector<double>> jjas = {{1,2},{1,2,3}};

    trainer.initialize_max_poolers({1,2,3},{1,2,3},{1,2,3});
    trainer.initialize_filter_layers({3,3,3},{1,1,1},{1,1,1},{"sigmoid","sigmoid","sigmoid"},{1,5,10},{5,10,15});
    trainer.initialize_fullys({15,10,5},{10,5,2},{"relu","relu","relu"});

    trainer.get_features();






    return 0;
}