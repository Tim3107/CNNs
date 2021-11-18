//
// Created by tim on 11.10.21.
//

#ifndef CNNS_SAVE_LOAD_IMAGE_LABEL_PAIRS_H
#define CNNS_SAVE_LOAD_IMAGE_LABEL_PAIRS_H

#include "string"
#include "sstream"
#include "vector"
#include "fstream"
#include "iterator"
#include "image_label_pair.h"
#include "image_processing.h"


/**Function to save Image as std::vector<std::vector<double>> and label as String in a textfile
 * The way this is done follows the rule that the first line of the .txt file stores the
 * number of rows of the given image. The second line stores the number of columns of the image.
 * The third line stores the label of the given image
 * The following lines after beginning with line 4 are storing the entries of the image pixels.
 *
 * @param filename : Where to save .txt file
 */
void save_image_label_pairs(std::string filename, Image_label_pair* image_label_object);

/**Function to load image label pairs from .txt file. This methods reads from the .txt-file with the name
 * filename.txt. Here one has to specify the whole root url of the .txt file.
 * Then the method extracts number of rows, cols and the label due to the pre-described procedure
 * (see @save_image_label_pairs) for further informations.
 *
 * @param filename : where to load image label pairs
 */
Image_label_pair load_image_label_pairs(std::string filename);

#endif //CNNS_SAVE_LOAD_IMAGE_LABEL_PAIRS_H