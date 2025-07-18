cmake_minimum_required(VERSION 3.4.1)

project(rknn_deepface_demographics)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")


# GENERAL LIB
# rknn api
set(RKNN_API_PATH ${RKNPU2}/runtime/${CMAKE_SYSTEM_NAME}/librknn_api)
if(LIB_ARCH STREQUAL "armhf-uclibc")
  set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnmrt.so)
else()
  set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
endif()
include_directories( ${RKNN_API_PATH}/include)


# RESIZE FUNC
set(RESIZE_FUNC ${CMAKE_SOURCE_DIR}/libs/resize_func)
set(RESIZE_FUNC_CC ${RESIZE_FUNC}/resize_function.cc)
include_directories( ${RESIZE_FUNC})

# cnpy
set(CNPY ${CMAKE_SOURCE_DIR}/libs/cnpy)
set(CNPY_CPP ${CNPY}/cnpy.cpp)
include_directories( ${CNPY})

# third party - stb
set(RKNPU2_THIRD_PARTY ${RKNPU2}/examples/3rdparty)
set(STB ${RKNPU2_THIRD_PARTY}/stb)
include_directories( ${STB})

# others
include_directories( ${CMAKE_SOURCE_DIR}/libs/utils)
include_directories( ${CMAKE_SOURCE_DIR}/src)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # set pthread
  set(PTHREAD_LIB pthread)
endif()

add_executable(rknn_deepface_demographics
          src/main.cc
          src/rknn_app.cc
          ${RESIZE_FUNC_CC}
          ${CNPY_CPP}
        )

target_link_libraries(rknn_deepface_demographics
      ${RKNN_RT_LIB}
      ${LIBZ_LIB}
      ${PTHREAD_LIB}
      dl
      jsoncpp
  )
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  target_link_libraries(rknn_deepface_demographics
      log
  )
endif()


# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_deepface_demographics)
install(TARGETS rknn_deepface_demographics DESTINATION ./)

install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/../model/facial_attribute.Age.rknn DESTINATION ./)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/../model/facial_attribute.Gender.rknn DESTINATION ./)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/../model/facial_attribute.Race.rknn DESTINATION ./)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/../model/facial_attribute.Emotion.rknn DESTINATION ./)