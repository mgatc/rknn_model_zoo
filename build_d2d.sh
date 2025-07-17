
GCC_COMPILER=/opt/bpi/aarch64-buildroot-linux-gnu_sdk-buildroot/bin/aarch64-linux
RKNPU2=/home/matt/rknn-toolkit2/rknpu2
RK_VERSION=rk3588

# RetinaFace detection
GCC_COMPILER=$GCC_COMPILER \
RKNPU2=$RKNPU2 \
RK_VERSION=$RK_VERSION \
./examples/RetinaFace/build_d2d.sh


# DeepFace demographic inference
GCC_COMPILER=$GCC_COMPILER \
RKNPU2=$RKNPU2 \
RK_VERSION=$RK_VERSION \
./examples/DeepFace/build_d2d.sh


# package
pushd ./install_d2d
tar -czvf ./d2d_rknn_apps.tar.gz ./rknn_apps/rknn_RetinaFace_demo ./rknn_apps/rknn_deepface_demographics
popd