//
// Created by tim on 09.10.21.
//

#ifndef CNNS_OUTPUT_LAYER_H
#define CNNS_OUTPUT_LAYER_H
#include "iostream"
#include "vector"
#include "string"
#include "../functions/softmax.h"
#include "../functions/cross_entropy.h"
#include "Matrix_computations.h"
#include "../functions/MSE.h"


class Output_layer{
    int input_dim;
    int output_dim;
    double loss = 0.0;
    std::string loss_function = "cross_entropy_loss";
    std::string classification_function = "softmax";
    std::vector<double> input;
    std::vector<double> output;
    std::vector<std::string> labels;
private:

public:

    /**Default Constructor
     *
     */
    Output_layer();

    /** Constructor of output_layer. This
     *
     * @param input_dim : Dimension of previous layer
     * @param output_dim : Dimension of last layer i.e. number of labels
     * @param loss_function : Kind of loss function used to measure error. Default is "cross_entropy_loss"
     * @param classification_function : Kind of classification function to classify input feature. Default is
     * "softmax" for the softmax function
     */
    Output_layer(int input_dim, int output_dim, std::string loss_function, std::string classification_function);

    /** Implementation of forward step in output layer
     *
     * @param input_vector : Input vector coming from hidden layer
     * @return returns array of likelihoods corresponding to labels
     */
    std::vector<double> forward_step(std::vector<double> input_vector);

    /**This routine computes the loss given the labels and the output of the hidden layers
     * The loss is stored in this class with the specifier "loss".
     */
    double compute_loss(std::vector<double> target_values);

    /**This routine computes the loss given the labels and the output of the hidden layers whereas the output is here given from outside the class
    * The loss is stored in this class with the specifier "loss".
    */
    double compute_loss(std::vector<double> target_values, std::vector<double> given_output);

    /**This routine computes all the relevant components in terms of backpropagation
    *
    * @param label : Label given from data
    * @param output_prediction : from forward pass created output of Neural Net
    * @return
    */
    //std::tuple<std::vector<std::vector<double>>,std::vector<double>>
    std::vector<double> backward_step(std::vector<double> label);
};


#endif //CNNS_OUTPUT_LAYER_H
