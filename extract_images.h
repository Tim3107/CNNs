//
// Created by tim on 02.10.21.
//

#ifndef CNNS_EXTRACT_IMAGES_H
#define CNNS_EXTRACT_IMAGES_H
#include <vector>
#include "string"
#include "opencv2/opencv.hpp"
#include "image_label_pair.h"
#include "image_processing.h"

class Image_extractor{

    std::string folder;
    std::string filename_trunk = "";
    int first_pos;
    int last_pos;
    int image_size_x;
    int image_size_y;


public:

    /**Default Constructor
     *
     */
    Image_extractor();
    /**Constructor of Image_extractor which extracts only grey scaled images.
     *
     * @param folder : Here are the images located
     * @param filename_trunk : The images are named and labeled similarly. The trunk of this specification is set here
     * @param first_pos : We want to numerate the images by numbers. These numbers are sorted straight. And so it is sufficient to point out the first number
     * @param last_pos : similar to @param first_pos do we need to set an ending number
     * @param image_size_x : number of pixels in horizontal direction. Avoids to large images
     * @param image_size_y : number of pixels in vertical direction. Avoids to large images
     */

    Image_extractor(std::string folder, std::string filename_trunk, int first_pos, int last_pos, int image_size_x, int image_size_y);

    /**This function extracts all predefined images and returns them in pointer array
     *
     * @param image_label_array : pointer to array with all the Image-Label pairs
     */
    void run_extractor(Image_label_pair* image_label_array);

    void schrott();

};


#endif //CNNS_EXTRACT_IMAGES_H
