//
// Created by tim on 05.10.21.
//

#ifndef CNNS_FCC_H
#define CNNS_FCC_H

#include <vector>
#include <string>
#include "Matrix_computations.h"
#include "functions/sigmoid.h"
#include "functions/ReLu.h"
#include "functions/softmax.h"
#include <iostream>
#include <cctype>


class FCC{
    int input_dim;
    int output_dim;
    double learning_rate = 1;
    std::string activation_function;
    std::vector<std::vector<double>> weight_matrix;
    std::vector<double> bias;
    std::vector<double> input;
    std::vector<double> output;
private:

public:
    /**@brief constructor creates object which consists of weight matrix and bias vector. This type of constructor sets
     * the activation function default to "sigmoid"
     *
     * @param input_dim : Number of components of input vector
     * @param output_dim : Number of components of output vector
     */
    FCC(int input_dim, int output_dim);

    /**@brief constructor creates object which consists of weight matrix and bias vector
     *
     * @param input_dim : Number of components of input vector
     * @param activation_function : String which defines what activation fucntion is used within one FC-Layer
     * @param output_dim : Number of components of output vector
     */
    FCC(int input_dim, int output_dim, std::string activation_function);

    /**@brief constructor creates object which consists of weight matrix and bias vector, this version allows the user
     * to set matrix and bias directly
     *
     * @param input_dim : Number of components of input vector
     * @param activation_function : String which defines what activation fucntion is used within one FC-Layer
     * @param output_dim : Number of components of output vector
     * @param default_matrix : With this parameter one is able to set the matrix directly
     * @param default_bias : With this parameter one is able to set the bias directly
     */
    FCC(int input_dim, int output_dim, std::string activation_function,std::vector<std::vector<double>> default_matrix, std::vector<double> default_bias);

    /** method which computes one forward step of a fully connected layer
     *
     * @param input_vector : input of previous layer
     * @return output_vector : output after applying linear trafo and activation function
     */
    std::vector<double> forward_step(std::vector<double> input_vector);

    /** method which computes one forward step of a fully connected layer when a 3D filtered image is given
 *
 * @param input_vector : input of previous layer in 3D
 * @return output_vector : output after applying linear trafo and activation function
 */
    std::vector<double> forward_step(std::vector<std::vector<std::vector<double>>> input_vector);

    /**This routine computes all the relevant components in terms of backpropagation
     *
     * @param input_successor : Input vector of successor layer in Feed-Forward NN. Important due to back propagating
     * error through net
     * @return
     */
    //std::tuple<std::vector<std::vector<double>>,std::vector<double>>
    std::vector<double> backward_step(std::vector<double> input_successor);

    /** This method returns weight-matrix and bias. Important for final step.
     *
     * @return weight matrix and bias
     */
    std::tuple<std::vector<std::vector<double>>,std::vector<double>> getter();

    /**This method is a setter method, which sets a new learning rate important for gradient descent
     *
     * @param learning_rate : new learning rate for gradient descent
     */
    void setter_learning_rate(double learning_rate);

    /**This routine updates the weights due to the gradient descent scheme
     *
     * @param update_weight_matrix : update matrix
     * @param update_bias_vector : update bias vector
     */
    void update_weights(std::vector<std::vector<double>> update_weight_matrix, std::vector<double> update_bias_vector);
};

#endif //CNNS_FCC_H
