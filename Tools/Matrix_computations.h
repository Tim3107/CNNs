//
// Created by tim on 05.10.21.
//

#ifndef CNNS_MATRIX_COMPUTATIONS_H
#define CNNS_MATRIX_COMPUTATIONS_H
#include "vector"
#include <random>
#include "assert.h"
#include "iostream"
#include "image_processing.h"

/**
 * Routine which computes Matrix*Vector
 * @param A : Matrix
 * @param b : RHS
 * @return A*b
 */
std::vector<double> matrix_Vector_multiplication(std::vector<std::vector<double>> A, std::vector<double> b);

/**
 * Routine which computes Matrix^T*Vector
 * @param A : Matrix
 * @param b : RHS
 * @return A*b
 */
std::vector<double> matrix_Vector_multiplication_transpose(std::vector<std::vector<double>> A, std::vector<double> b);

/**This routine computes the dyadic product of two vectors
 *
 * @param vector_1 : first vector
 * @param vector_2 : second vector
 * @return outer product
 */
std::vector<std::vector<double>> dyadic_product(std::vector<double> vector_1,std::vector<double> vector_2);

/**This routine adds componentwise two matrices
 *
 * @param matrix_2 : first one
 * @param matrix_2 : second one
 * @return : sum of two matrices
 */
std::vector<std::vector<double>> add_matrices(std::vector<std::vector<double>> matrix_1, std::vector<std::vector<double>> matrix_2);

/**This routine multiples a matrix with a scalar
 *
 * @param matrix : Given matrix
 * @param scalar : Given Scalar to multiply onto matrix
 * @return Result of scalar*matrix
 */
std::vector<std::vector<double>> matrix_scalar(std::vector<std::vector<double>> matrix, double scalar);

/**This routine is able to multiply two vectors elementwise
 *
 * @param input_matrix_1 : target onto which computations are apllied
 * @param input_matrix_2 : second matrix
 * @param scalar : scalar to multiply result
 * @return returns input_matrix_1 * input_matrix_2 * scalar
 */
std::vector<std::vector<double>> matrix_multiplication_elementwise(std::vector<std::vector<double>> input_matrix_1, std::vector<std::vector<double>> input_matrix_2, double scalar);

/**This routine is able to multiply two vectors elementwise
 *
 * @param input_vector_1 : target onto which computations are apllied
 * @param input_vector_2 : second vector
 * @param scalar : scalar to multiply result
 * @return returns input_vector_1 * input_vector_2 * scalar
 */
std::vector<double> vector_multiplication_elementwise(std::vector<double> input_vector_1, std::vector<double> input_vector_2, double scalar);

/**This routine is able to add a scalar times a vector onto a given vector
 *
 * @param input_vector_1 : target onto which computations are apllied
 * @param input_vector_2 : this vector times scalar onto input_vector_1
 * @param scalar : scalar to multiply onto vector_2
 * @return returns input_vector_1 + scalar times input_vector_2
 */
std::vector<double> vector_additions(std::vector<double> input_vector_1, std::vector<double> input_vector_2, double scalar);

/**This function returns a vector filled with random values
 *
 * @param rows : number of rows of random vector
 * @param range : random numbers between 0 and @param range
 * @return random output vector
 */
std::vector<double> random_vector(int rows, double range);

/**This function returns a matrix filled with random values
 *
 * @param rows : number of rows of random matrix
 * @param cols : number of columns of random matrix
 * @param range : random numbers between 0 and @param range
 * @return random output matrix
 */
std::vector<std::vector<double>> random_matrix(int rows, int cols, double range);


#endif //CNNS_MATRIX_COMPUTATIONS_H
