cmake_minimum_required(VERSION 3.20)
project(CNNs)

set(CMAKE_CXX_STANDARD 14)

include(CTest)
enable_testing()

FIND_PACKAGE(OpenCV REQUIRED HINTS ${OPENCV4} ${OPENCV4} ../ ../../ $ENV{OPENCV4})


add_subdirectory(functions)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})


INCLUDE_DIRECTORIES(${functions})

add_executable(CNNs main.cpp Tools/Filter.cpp Tools/Filter.h Tools/max_pooling.cpp Tools/max_pooling.h Tools/extract_images.cpp Tools/extract_images.h Tools/image_processing.cpp Tools/image_processing.h functions/sigmoid.cpp functions/sigmoid.h functions/ReLu.cpp functions/ReLu.h functions/softmax.cpp functions/softmax.h Tools/FCC.cpp Tools/FCC.h Tools/Matrix_computations.cpp Tools/Matrix_computations.h Tools/output_layer.cpp Tools/output_layer.h functions/cross_entropy.cpp functions/cross_entropy.h Tools/image_label_pair.cpp Tools/image_label_pair.h Tools/Filter_ensemble.cpp Tools/Filter_ensemble.h Tools/Save_load_image_label_pairs.cpp Tools/Save_load_image_label_pairs.h Tools/Filter_layer.cpp Tools/Filter_layer.h functions/MSE.cpp functions/MSE.h Tools/Trainer.cpp Tools/Trainer.h Tools/Filter_Operations.cpp Tools/Filter_Operations.h)
add_executable(Examples_forward examples/Forward_step_testing.cpp Tools/Filter.cpp Tools/Filter.h Tools/max_pooling.cpp Tools/max_pooling.h Tools/extract_images.cpp Tools/extract_images.h Tools/image_processing.cpp Tools/image_processing.h functions/sigmoid.cpp functions/sigmoid.h functions/ReLu.cpp functions/ReLu.h functions/softmax.cpp functions/softmax.h Tools/FCC.cpp Tools/FCC.h Tools/Matrix_computations.cpp Tools/Matrix_computations.h Tools/output_layer.cpp Tools/output_layer.h functions/cross_entropy.cpp functions/cross_entropy.h Tools/image_label_pair.cpp Tools/image_label_pair.h Tools/Filter_ensemble.cpp Tools/Filter_ensemble.h Tools/Save_load_image_label_pairs.cpp Tools/Save_load_image_label_pairs.h Tools/Filter_layer.cpp Tools/Filter_layer.h functions/MSE.cpp functions/MSE.h Tools/Trainer.cpp Tools/Trainer.h Tools/Filter_Operations.cpp Tools/Filter_Operations.h)
add_executable(Examples_backward examples/Backward_step_testing.cpp Tools/Filter.cpp Tools/Filter.h Tools/max_pooling.cpp Tools/max_pooling.h Tools/extract_images.cpp Tools/extract_images.h Tools/image_processing.cpp Tools/image_processing.h functions/sigmoid.cpp functions/sigmoid.h functions/ReLu.cpp functions/ReLu.h functions/softmax.cpp functions/softmax.h Tools/FCC.cpp Tools/FCC.h Tools/Matrix_computations.cpp Tools/Matrix_computations.h Tools/output_layer.cpp Tools/output_layer.h functions/cross_entropy.cpp functions/cross_entropy.h Tools/image_label_pair.cpp Tools/image_label_pair.h Tools/Filter_ensemble.cpp Tools/Filter_ensemble.h Tools/Save_load_image_label_pairs.cpp Tools/Save_load_image_label_pairs.h Tools/Filter_layer.cpp Tools/Filter_layer.h functions/MSE.cpp functions/MSE.h Tools/Trainer.cpp Tools/Trainer.h Tools/Filter_Operations.cpp Tools/Filter_Operations.h)
add_executable(Xor_Problems examples/xor_Problem.cpp Tools/Filter.cpp Tools/Filter.h Tools/max_pooling.cpp Tools/max_pooling.h Tools/extract_images.cpp Tools/extract_images.h Tools/image_processing.cpp Tools/image_processing.h functions/sigmoid.cpp functions/sigmoid.h functions/ReLu.cpp functions/ReLu.h functions/softmax.cpp functions/softmax.h Tools/FCC.cpp Tools/FCC.h Tools/Matrix_computations.cpp Tools/Matrix_computations.h Tools/output_layer.cpp Tools/output_layer.h functions/cross_entropy.cpp functions/cross_entropy.h Tools/image_label_pair.cpp Tools/image_label_pair.h Tools/Filter_ensemble.cpp Tools/Filter_ensemble.h Tools/Save_load_image_label_pairs.cpp Tools/Save_load_image_label_pairs.h Tools/Filter_layer.cpp Tools/Filter_layer.h functions/MSE.cpp functions/MSE.h Tools/Trainer.cpp Tools/Trainer.h Tools/Filter_Operations.cpp Tools/Filter_Operations.h)
add_executable(Simple_Testing examples/simple_test.cpp Tools/Filter.cpp Tools/Filter.h Tools/max_pooling.cpp Tools/max_pooling.h Tools/extract_images.cpp Tools/extract_images.h Tools/image_processing.cpp Tools/image_processing.h functions/sigmoid.cpp functions/sigmoid.h functions/ReLu.cpp functions/ReLu.h functions/softmax.cpp functions/softmax.h Tools/FCC.cpp Tools/FCC.h Tools/Matrix_computations.cpp Tools/Matrix_computations.h Tools/output_layer.cpp Tools/output_layer.h functions/cross_entropy.cpp functions/cross_entropy.h Tools/image_label_pair.cpp Tools/image_label_pair.h Tools/Filter_ensemble.cpp Tools/Filter_ensemble.h Tools/Save_load_image_label_pairs.cpp Tools/Save_load_image_label_pairs.h Tools/Filter_layer.cpp Tools/Filter_layer.h functions/MSE.cpp functions/MSE.h Tools/Trainer.cpp Tools/Trainer.h Tools/Filter_Operations.cpp Tools/Filter_Operations.h)
add_executable(Fun_Testing examples/Fun.cpp Tools/Filter.cpp Tools/Filter.h Tools/max_pooling.cpp Tools/max_pooling.h Tools/extract_images.cpp Tools/extract_images.h Tools/image_processing.cpp Tools/image_processing.h functions/sigmoid.cpp functions/sigmoid.h functions/ReLu.cpp functions/ReLu.h functions/softmax.cpp functions/softmax.h Tools/FCC.cpp Tools/FCC.h Tools/Matrix_computations.cpp Tools/Matrix_computations.h Tools/output_layer.cpp Tools/output_layer.h functions/cross_entropy.cpp functions/cross_entropy.h Tools/image_label_pair.cpp Tools/image_label_pair.h Tools/Filter_ensemble.cpp Tools/Filter_ensemble.h Tools/Save_load_image_label_pairs.cpp Tools/Save_load_image_label_pairs.h Tools/Filter_layer.cpp Tools/Filter_layer.h functions/MSE.cpp functions/MSE.h Tools/Trainer.cpp Tools/Trainer.h Tools/Filter_Operations.cpp Tools/Filter_Operations.h)
add_executable(ConvBP examples/Test_Convolution_Backpropagation.cpp Tools/Filter.cpp Tools/Filter.h Tools/max_pooling.cpp Tools/max_pooling.h Tools/extract_images.cpp Tools/extract_images.h Tools/image_processing.cpp Tools/image_processing.h functions/sigmoid.cpp functions/sigmoid.h functions/ReLu.cpp functions/ReLu.h functions/softmax.cpp functions/softmax.h Tools/FCC.cpp Tools/FCC.h Tools/Matrix_computations.cpp Tools/Matrix_computations.h Tools/output_layer.cpp Tools/output_layer.h functions/cross_entropy.cpp functions/cross_entropy.h Tools/image_label_pair.cpp Tools/image_label_pair.h Tools/Filter_ensemble.cpp Tools/Filter_ensemble.h Tools/Save_load_image_label_pairs.cpp Tools/Save_load_image_label_pairs.h Tools/Filter_layer.cpp Tools/Filter_layer.h functions/MSE.cpp functions/MSE.h Tools/Trainer.cpp Tools/Trainer.h Tools/Filter_Operations.cpp Tools/Filter_Operations.h)


TARGET_LINK_LIBRARIES(CNNs ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(Examples_forward ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(Examples_backward ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(Xor_Problems ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(Simple_Testing ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(Fun_Testing ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(ConvBP ${OpenCV_LIBS})


