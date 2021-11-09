cmake_minimum_required(VERSION 3.20)
project(CNNs)

set(CMAKE_CXX_STANDARD 14)

include(CTest)
enable_testing()

FIND_PACKAGE(OpenCV REQUIRED HINTS ${OPENCV4} ${OPENCV4} ../ ../../ $ENV{OPENCV4})


add_subdirectory(functions)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})


INCLUDE_DIRECTORIES(${functions})

add_executable(CNNs main.cpp Filter.cpp Filter.h max_pooling.cpp max_pooling.h extract_images.cpp extract_images.h image_processing.cpp image_processing.h functions/sigmoid.cpp functions/sigmoid.h functions/ReLu.cpp functions/ReLu.h functions/softmax.cpp functions/softmax.h FCC.cpp FCC.h Matrix_computations.cpp Matrix_computations.h output_layer.cpp output_layer.h functions/cross_entropy.cpp functions/cross_entropy.h image_label_pair.cpp image_label_pair.h Filter_ensemble.cpp Filter_ensemble.h Save_load_image_label_pairs.cpp Save_load_image_label_pairs.h Filter_layer.cpp Filter_layer.h functions/MSE.cpp functions/MSE.h Trainer.cpp Trainer.h)
add_executable(Examples_forward examples/Forward_step_testing.cpp Filter.cpp Filter.h max_pooling.cpp max_pooling.h extract_images.cpp extract_images.h image_processing.cpp image_processing.h functions/sigmoid.cpp functions/sigmoid.h functions/ReLu.cpp functions/ReLu.h functions/softmax.cpp functions/softmax.h FCC.cpp FCC.h Matrix_computations.cpp Matrix_computations.h output_layer.cpp output_layer.h functions/cross_entropy.cpp functions/cross_entropy.h image_label_pair.cpp image_label_pair.h Filter_ensemble.cpp Filter_ensemble.h Save_load_image_label_pairs.cpp Save_load_image_label_pairs.h Filter_layer.cpp Filter_layer.h functions/MSE.cpp functions/MSE.h Trainer.cpp Trainer.h)
add_executable(Examples_backward examples/Backward_step_testing.cpp Filter.cpp Filter.h max_pooling.cpp max_pooling.h extract_images.cpp extract_images.h image_processing.cpp image_processing.h functions/sigmoid.cpp functions/sigmoid.h functions/ReLu.cpp functions/ReLu.h functions/softmax.cpp functions/softmax.h FCC.cpp FCC.h Matrix_computations.cpp Matrix_computations.h output_layer.cpp output_layer.h functions/cross_entropy.cpp functions/cross_entropy.h image_label_pair.cpp image_label_pair.h Filter_ensemble.cpp Filter_ensemble.h Save_load_image_label_pairs.cpp Save_load_image_label_pairs.h Filter_layer.cpp Filter_layer.h functions/MSE.cpp functions/MSE.h Trainer.cpp Trainer.h)
add_executable(Xor_Problems examples/xor_Problem.cpp Filter.cpp Filter.h max_pooling.cpp max_pooling.h extract_images.cpp extract_images.h image_processing.cpp image_processing.h functions/sigmoid.cpp functions/sigmoid.h functions/ReLu.cpp functions/ReLu.h functions/softmax.cpp functions/softmax.h FCC.cpp FCC.h Matrix_computations.cpp Matrix_computations.h output_layer.cpp output_layer.h functions/cross_entropy.cpp functions/cross_entropy.h image_label_pair.cpp image_label_pair.h Filter_ensemble.cpp Filter_ensemble.h Save_load_image_label_pairs.cpp Save_load_image_label_pairs.h Filter_layer.cpp Filter_layer.h functions/MSE.cpp functions/MSE.h Trainer.cpp Trainer.h)
add_executable(Simple_Testing examples/simple_test.cpp Filter.cpp Filter.h max_pooling.cpp max_pooling.h extract_images.cpp extract_images.h image_processing.cpp image_processing.h functions/sigmoid.cpp functions/sigmoid.h functions/ReLu.cpp functions/ReLu.h functions/softmax.cpp functions/softmax.h FCC.cpp FCC.h Matrix_computations.cpp Matrix_computations.h output_layer.cpp output_layer.h functions/cross_entropy.cpp functions/cross_entropy.h image_label_pair.cpp image_label_pair.h Filter_ensemble.cpp Filter_ensemble.h Save_load_image_label_pairs.cpp Save_load_image_label_pairs.h Filter_layer.cpp Filter_layer.h functions/MSE.cpp functions/MSE.h Trainer.cpp Trainer.h)
add_executable(Fun_Testing examples/Fun.cpp Filter.cpp Filter.h max_pooling.cpp max_pooling.h extract_images.cpp extract_images.h image_processing.cpp image_processing.h functions/sigmoid.cpp functions/sigmoid.h functions/ReLu.cpp functions/ReLu.h functions/softmax.cpp functions/softmax.h FCC.cpp FCC.h Matrix_computations.cpp Matrix_computations.h output_layer.cpp output_layer.h functions/cross_entropy.cpp functions/cross_entropy.h image_label_pair.cpp image_label_pair.h Filter_ensemble.cpp Filter_ensemble.h Save_load_image_label_pairs.cpp Save_load_image_label_pairs.h Filter_layer.cpp Filter_layer.h functions/MSE.cpp functions/MSE.h Trainer.cpp Trainer.h)


TARGET_LINK_LIBRARIES(CNNs ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(Examples_forward ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(Examples_backward ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(Xor_Problems ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(Simple_Testing ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(Fun_Testing ${OpenCV_LIBS})

