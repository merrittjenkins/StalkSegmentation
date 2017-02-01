cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(MY_GRAND_PROJECT)

set(OpenCV_DIR "/usr/include/opencv2")

find_package(PCL 1.3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui calib3d)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
#list(APPEND LIBRARIES_LIST ${OpenCV_LIBRARIES} )

#add_executable(loadImage loadImage.cpp)
#target_link_libraries(loadImage ${OpenCV_LIBS})

#add_executable(image_brightness image_brightness.cpp)
#target_link_libraries(image_brightness ${OpenCV_LIBS})

#add_executable(Flann Flann.cpp)
#target_link_libraries(Flann ${OpenCV_LIBS} opencv_nonfree)

#add_executable(BruteForce BruteForce.cpp)
#target_link_libraries(BruteForce ${OpenCV_LIBS} opencv_nonfree)

include_directories(include)
set(SOURCES src/SGBM4.cpp src/write_ply.cpp src/scale_image.cpp src/feature_match.cpp)# ${include}/write_ply.h)

#add_executable(SGBM SGBM.cpp)
#target_link_libraries(SGBM ${OpenCV_LIBS} opencv_nonfree)

#add_executable(SGBM2 SGBM2.cpp)
#target_link_libraries(SGBM2 ${OpenCV_LIBS} opencv_nonfree)

#add_executable(SGBM3 SGBM3.cpp)
#target_link_libraries(SGBM3 ${OpenCV_LIBS} opencv_nonfree)

#add_executable(SGBM4 ${SOURCES})
#target_link_libraries(SGBM4 ${OpenCV_LIBS} opencv_nonfree)

add_executable(concatenateClouds concatenateClouds.cpp)
target_link_libraries(concatenateClouds ${PCL_LIBRARIES})

add_executable(groundPlane groundPlane.cpp)
target_link_libraries(groundPlane ${PCL_LIBRARIES})

add_executable(wtf wtf.cpp)
target_link_libraries(wtf ${PCL_LIBRARIES})

add_executable(pcl_visualizer pcl_visualizer.cpp)
target_link_libraries(pcl_visualizer ${PCL_LIBRARIES})

add_executable(topng topng.cpp)
target_link_libraries(topng ${PCL_LIBRARIES})

add_executable(normalFilter normalFilter.cpp)
target_link_libraries(normalFilter ${PCL_LIBRARIES})

add_executable(createSections createSections.cpp)
target_link_libraries(createSections ${PCL_LIBRARIES})

add_executable(createScatter createScatter.cpp)
target_link_libraries(createScatter ${PCL_LIBRARIES})

add_executable(createScatter_MLS createScatter_MLS.cpp)
target_link_libraries(createScatter_MLS ${PCL_LIBRARIES})

add_executable(learnMap learnMap.cpp)
target_link_libraries(learnMap ${PCL_LIBRARIES})

add_executable(diffofnorm_seg diffofnorm_seg.cpp)
target_link_libraries(diffofnorm_seg ${PCL_LIBRARIES})

add_executable(cond_euclid_cluster cond_euclid_cluster.cpp)
target_link_libraries(cond_euclid_cluster ${PCL_LIBRARIES})

add_executable(combnormdensity combnormdensity.cpp)
target_link_libraries(combnormdensity ${PCL_LIBRARIES})

add_executable(pcl_cluster_tutorial pcl_cluster_tutorial.cpp)
target_link_libraries(pcl_cluster_tutorial ${PCL_LIBRARIES})

add_executable(grow_up grow_up.cpp)
target_link_libraries(grow_up ${PCL_LIBRARIES})

add_executable(grow_up2 grow_up2.cpp)
target_link_libraries(grow_up2 ${PCL_LIBRARIES})

add_executable(grow_up3 grow_up3.cpp)
target_link_libraries(grow_up3 ${PCL_LIBRARIES})

add_executable(grow_up4 grow_up4.cpp)
target_link_libraries(grow_up4 ${PCL_LIBRARIES})

add_executable(grow_up5 src/grow_up5.cpp)
target_link_libraries(grow_up5 ${PCL_LIBRARIES})

add_executable(grow_up6 src/grow_up6.cpp)
target_link_libraries(grow_up6 ${PCL_LIBRARIES})

add_executable(euclid_cluster euclid_cluster.cpp)
target_link_libraries(euclid_cluster ${PCL_LIBRARIES})

set(OTHER_SOURCES src/main3.cpp src/write_ply.cpp src/scale_image.cpp src/feature_match.cpp)
add_executable(main3 ${OTHER_SOURCES})
target_link_libraries(main3 ${OpenCV_LIBS} opencv_nonfree ${PCL_LIBRARIES})

set(OTHER_SOURCES src/main4.cpp src/write_ply.cpp src/scale_image.cpp src/feature_match.cpp)
add_executable(main4 ${OTHER_SOURCES})
target_link_libraries(main4 ${OpenCV_LIBS} opencv_nonfree ${PCL_LIBRARIES})

set(OTHER_SOURCES src/main4_test.cpp src/write_ply.cpp src/scale_image.cpp src/feature_match.cpp)
add_executable(main4_test ${OTHER_SOURCES})
target_link_libraries(main4_test ${OpenCV_LIBS} opencv_nonfree ${PCL_LIBRARIES})

set(OTHER_SOURCES src/incrementalRegister_noVis.cpp src/write_ply.cpp src/scale_image.cpp src/feature_match.cpp)
add_executable(incrementalRegister_noVis ${OTHER_SOURCES})
target_link_libraries(incrementalRegister_noVis ${OpenCV_LIBS} opencv_nonfree ${PCL_LIBRARIES})
