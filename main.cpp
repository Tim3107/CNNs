#include <iostream>
#include "Filter.h"
#include "vector"
#include "max_pooling.h"
#include "opencv2/opencv.hpp"
#include "image_processing.h"
#include "functions/softmax.h"
#include "Matrix_computations.h"
#include "image_label_pair.h"
#include "Save_load_image_label_pairs.h"
#include "extract_images.h"

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




    Filter filter_grad(3,1,1,y_grad);
    std::vector<std::vector<double>> test_array = conversion_to_std_vector(grey_image);
    std::vector<std::vector<double>> filtered_array = filter_grad.run_Filter(test_vector);
    Mat return_Arrray = conversion_to_Mat(filtered_array);

    imshow("filtered",return_Arrray);

    waitKey(0);

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
    return 0;
}