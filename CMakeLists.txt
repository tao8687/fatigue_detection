project(opencv_test)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV QUIET)
find_package(dlib REQUIRED)

if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message("OpenCV not found, so we won't build the webcam_face_pose_ex example.")
endif()

aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} dlib::dlib )