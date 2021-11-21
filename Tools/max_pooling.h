//
// Created by tim on 01.10.21.
//

#ifndef CNNS_MAX_POOLING_H
#define CNNS_MAX_POOLING_H

#include "vector"
#include "image_processing.h"

class Max_pooling{
    int dim_Filter;
    int stride;
    int padding;
    int index_x = 0;
    int index_y = 0;
    std::vector<std::vector<std::vector<int>>> max_array_2D;
    std::vector<std::vector<std::vector<int>>> max_array_3D;
private:

public:

    /**Default Constructor
     *
     */
    Max_pooling();

    /** \brief Constructor of a Filter
    * @param dim_Filter: Dimension of Filter
    * @param padding : Number of added zeros on each edge
    * @param stride : Number of overjumped elements of each Filter operation
    */
    Max_pooling(int dim_Filter, int stride, int padding);
    /**
     *
     * @param input_image : Input image on which we want to max_pool
     * @param channel : sets which channel is now pooled. Important for BP when one needs to set index of max value in max_array_3D
     * @return output_array : pooled array
     */
    std::vector<std::vector<double>> run_max_pooling(std::vector<std::vector<double>> input_image, int channel);

    /**This routine should be called when one wants to max-pool an image with more than one channel
     *
     * @param input_image : a 3D Input image
     * @return output_image : a 3D max-pooled image
     */
    std::vector<std::vector<std::vector<double>>> run_max_pooling_3D(std::vector<std::vector<std::vector<double>>> input_image);

    /**This method performs the BP step for a pooling layer
     *
     * @param input_successor : backpropagated gradient from successor layer.
     * @return gradient
     */
    std::vector<std::vector<std::vector<double>>> backward_pooler(std::vector<std::vector<std::vector<double>>> input_successor);

    /**This routine does the BP step of the max-pooler for a single 2D gradient
     *
     * @param input_successor_2D : The 2D input_gradient of BP scheme
     * @param channel : Which channel of 3D Image is used in this method call
     * @return Gradient distributed on one hot positions of this->max_Array_3D
     */
    std::vector<std::vector<double>> backward_pooler_2D(std::vector<std::vector<double>> input_successor_2D,int channel);

    /**
    *
    * @param input_array : input image which needs to get padded
    * @return output_array : returns array which got padded edges
    */
    std::vector<std::vector<double>> apply_padding(std::vector<std::vector<double>> input_array);
};

#endif //CNNS_MAX_POOLING_H
