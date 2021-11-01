//
// Created by tim on 01.10.21.
//

#ifndef CNNS_MAX_POOLING_H
#define CNNS_MAX_POOLING_H

#include "vector"

class Max_pooling{
    int dim_Filter;
    int stride;
    int padding;
private:

public:
    /** \brief Constructor of a Filter
    * @param dim_Filter: Dimension of Filter
    * @param padding : Number of added zeros on each edge
    * @param stride : Number of overjumped elements of each Filter operation
    */
    Max_pooling(int dim_Filter, int stride, int padding);
    /**
     *
     * @param input_image : Input image on which we want to max_pool
     * @return output_array : pooled array
     */
    std::vector<std::vector<double>> run_max_pooling(std::vector<std::vector<double>> input_image);

    /**This routine should be called when one wants to max-pool an image with more than one channel
     *
     * @param input_image : a 3D Input image
     * @return output_image : a 3D max-pooled image
     */
    std::vector<std::vector<std::vector<double>>> run_max_pooling_3D(std::vector<std::vector<std::vector<double>>> input_image);

    /**
    *
    * @param input_array : input image which needs to get padded
    * @return output_array : returns array which got padded edges
    */
    std::vector<std::vector<double>> apply_padding(std::vector<std::vector<double>> input_array);
};

#endif //CNNS_MAX_POOLING_H
