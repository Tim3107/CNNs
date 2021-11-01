//
// Created by tim on 10.10.21.
//

#ifndef CNNS_IMAGE_LABEL_PAIR_H
#define CNNS_IMAGE_LABEL_PAIR_H

#include "vector"
#include "opencv2/opencv.hpp"
#include "string"

class Image_label_pair{
private:

public:
    std::vector<std::vector<double>> image;
    std::string label;

    /**Default constructor
     *
     */
    Image_label_pair();

    /**Constructor of image class
     *
     * @param image : Image is represented as std::vector
     * @param label : label is string
     */


    Image_label_pair(std::vector<std::vector<double>> image,std::string label);

    /** Getter for images
     *
     * @return Image as std::vector<std::vector<double>>
     */

    std::vector<std::vector<double>> get_Image();

    /** Getter for labels
     *
     * @return Label as std::string
     */

    std::string get_Label();

};

#endif //CNNS_IMAGE_LABEL_PAIR_H
