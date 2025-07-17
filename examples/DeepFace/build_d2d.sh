
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

    # DeepFace

    # start the docker and convert the models
    sudo docker run -t -i --privileged \
        -v ./examples:/examples \
        -w /examples/DeepFace/python \
        rknn-toolkit2:latest \
            python3 ./convert.py facial_attribute Age $RK_VERSION

    sudo docker run -t -i --privileged \
        -v ./examples:/examples \
        -w /examples/DeepFace/python \
        rknn-toolkit2:latest \
            python3 ./convert.py facial_attribute Gender $RK_VERSION

    sudo docker run -t -i --privileged \
        -v ./examples:/examples \
        -w /examples/DeepFace/python \
        rknn-toolkit2:latest \
            python3 ./convert.py facial_attribute Race $RK_VERSION

    sudo docker run -t -i --privileged \
        -v ./examples:/examples \
        -w /examples/DeepFace/python \
        rknn-toolkit2:latest \
            python3 ./convert.py facial_attribute Emotion $RK_VERSION

    pushd ./examples/DeepFace/cpp
        # build
        [ -e autom4te.cache ] && rm -rf autom4te.cache
        chmod +x ./build-linux.sh
        GCC_COMPILER=$GCC_COMPILER RKNPU2=$RKNPU2\
            ./build_linux.sh $RK_VERSION
    popd

    # # package
    pushd ./install_d2d
        rm -rf ./rknn_apps/rknn_deepface_demographics
        mv -f ../examples/DeepFace/cpp/install/rknn_deepface_demographics ./rknn_apps
        rm -f ./rknn_deepface_demographics.tar.gz
        tar -czvf ./rknn_deepface_demographics.tar.gz ./rknn_apps/rknn_deepface_demographics
    popd
    
popd
