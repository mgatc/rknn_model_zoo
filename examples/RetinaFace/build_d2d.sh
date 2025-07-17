
# Get the rknn toolkit for Dockerfile and rknpu2 library
if [ ! -d ../rknn-toolkit2 ]; then
    git clone -b main --single-branch git@github.com:mgatc/rknn-toolkit2.git ../rknn-toolkit2 
else
    pushd ../rknn-toolkit2
        git pull
    popd
fi

# build docker image
pushd ../rknn-toolkit2/rknn-toolkit2/docker/docker_file/ubuntu_20_04_cp38
    sudo docker build -t rknn-toolkit2:latest \
        -f Dockerfile_ubuntu_20_04_for_cp38 .
popd


# Get the rknn-model-zoo if not present
if [ ! -d ../rknn_model_zoo ]; then
    git clone -b main --single-branch git@github.com:mgatc/rknn_model_zoo.git ../rknn_model_zoo
else
    pushd ../rknn_model_zoo
        git pull
    popd
fi

# Build the app collaterals
pushd ../rknn_model_zoo
    # Put the collaterals in a subdir of rknn_apps.
    # Put the packages in install_d2d.
    mkdir -p ./install_d2d/rknn_apps

    # RetinaFace

    # download and convert model if not present
    if [ ! -f ./examples/RetinaFace/model/RetinaFace_mobile320.onnx ]; then
        pushd ./examples/RetinaFace/model
            chmod +x ./download_model.sh
            ./download_model.sh
        popd
    fi
    # start the docker
    sudo docker run -t -i --privileged \
        -v ./examples:/examples \
        -w /examples/RetinaFace/python \
        rknn-toolkit2:latest \
            python3 ./convert.py ../model/RetinaFace_mobile320.onnx $RK_VERSION

    # build
    [ -e autom4te.cache ] && rm -rf autom4te.cache
    chmod +x ./build-linux.sh
    GCC_COMPILER=$GCC_COMPILER \
        ./build-linux.sh -t $RK_VERSION -a aarch64 -d RetinaFace

    # package
    pushd ./install_d2d
        mv -f ../install/${RK_VERSION}_linux_aarch64/rknn_RetinaFace_demo ./rknn_apps/
        rm -f ./rknn_RetinaFace_demo.tar.gz
        tar -czvf ./rknn_RetinaFace_demo.tar.gz ./rknn_apps/rknn_RetinaFace_demo
    popd
    
popd
