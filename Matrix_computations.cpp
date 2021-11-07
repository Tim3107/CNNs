//
// Created by tim on 05.10.21.
//

#include "Matrix_computations.h"


std::vector<double> matrix_Vector_multiplication(std::vector<std::vector<double>> A, std::vector<double> b){
    int rows = A.size();
    int cols = A[0].size();
    int size_b = b.size();
    assert(cols == size_b);
    std::vector<double> result = std::vector<double>(rows,0);

    for (int i = 0;i<rows;i++){
        for (int j = 0;j<cols;j++){
            result[i] += A[i][j]*b[j];
        }
    }
    return result;
}

std::vector<double> matrix_Vector_multiplication_transpose(std::vector<std::vector<double>> A, std::vector<double> b){


    int rows = A.size();
    int cols = A[0].size();
    int size_b = b.size();
    assert(rows == size_b);

    std::vector<double> result = std::vector<double>(cols,0);

    for (int i = 0;i<cols;i++){
        for (int j = 0;j<rows;j++){
            result[i] += A[j][i]*b[j];
        }
    }
    return result;
}

std::vector<std::vector<double>> dyadic_product(std::vector<double> vector_1,std::vector<double> vector_2){
    int rows = vector_1.size();
    int cols = vector_2.size();
    std::vector<std::vector<double>> outer_product(rows,std::vector<double>(cols,0));

    for (int i = 0;i<rows;i++){
        for (int j = 0;j<cols;j++){
            outer_product[i][j] = vector_1[i]*vector_2[j];
        }
    }

    return outer_product;
}

std::vector<std::vector<double>> add_matrices(std::vector<std::vector<double>> matrix_1, std::vector<std::vector<double>> matrix_2){
    int rows = matrix_1.size();
    int cols = matrix_1[0].size();
    int rows_test = matrix_2.size();
    int cols_test = matrix_2[0].size();
    assert(rows = rows_test);
    assert(cols = cols_test);
    std::vector<std::vector<double>> matrix_sum(rows,std::vector<double>(cols,0));

    for (int i = 0;i<rows;i++){
        for (int j = 0;j<cols;j++){
            matrix_sum[i][j] = matrix_1[i][j]+matrix_2[i][j];
        }
    }

    return matrix_sum;
}

std::vector<std::vector<double>> matrix_scalar(std::vector<std::vector<double>> matrix, double scalar){
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<std::vector<double>> matrix_scalar(rows,std::vector<double>(cols,0));

    for (int i = 0;i<rows;i++){
        for (int j = 0;j<cols;j++){
            matrix_scalar[i][j] = matrix[i][j]*scalar;
        }
    }

    return matrix_scalar;

}

std::vector<double> vector_multiplication_elementwise(std::vector<double> input_vector_1, std::vector<double> input_vector_2, double scalar){
    int size_1 = input_vector_1.size();
    int size_2 = input_vector_2.size();
    assert(size_1 == size_2);
    std::vector<double> output(size_1,0);
    for (int i = 0; i<size_1;i++){
        output[i] = input_vector_1[i]*input_vector_2[i]*scalar;
        }
    return output;
}

std::vector<double> vector_additions(std::vector<double> input_vector_1, std::vector<double> input_vector_2, double scalar){
    int rows = input_vector_1.size();
    int rows_test = input_vector_2.size();
    assert(rows == rows_test);
    std::vector<double> result(rows,0);
    for (int i = 0; i<rows;i++){
        result[i] = input_vector_1[i] + scalar*input_vector_2[i];
    }
    return result;
}


std::vector<double> random_vector(int rows, double range){
    std::uniform_real_distribution<double> unif(0,range);
    std::default_random_engine re;
    std::vector<double> output_array(rows,0);
    for (int i = 0;i<rows;i++){
        output_array[i] = unif(re);
    }
    return  output_array;
}


std::vector<std::vector<double>> random_matrix(int rows, int cols, double range){
    std::uniform_real_distribution<double> unif(0,range);
    std::default_random_engine re;
    std::vector<std::vector<double>> output_array(rows,std::vector<double>(cols,0));
    for (int i = 0;i<rows;i++){
        for (int j = 0;j<cols;j++){
            output_array[i][j] = unif(re);
        }
    }

    return output_array;
}
