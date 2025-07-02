import sys
from rknn.api import RKNN

from deepface import DeepFace
from tensorflow import keras
from keras.models import load_model

import argparse
import sys
import os
import tensorflow as tf

import tf2onnx
import onnx

from deepface.modules import modeling, detection, preprocessing
from deepface.models.demography import Gender, Race, Emotion

DATASET_PATH = '../model/dataset.txt'
DEFAULT_RKNN_PATH = '../model/DeepFace.rknn'
DEFAULT_QUANT = True

def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} [task] [model] [platform] [dtype(optional)] [output_rknn_path(optional)]".format(sys.argv[0]));
        print("       platform choose from [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b, rv1109, rv1126, rk1808]")
        print("       dtype choose from    [i8] for [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b]")
        print("       dtype choose from    [u8] for [rv1109, rv1126, rk1808]")
        exit(1)

    task = sys.argv[1]
    model = sys.argv[2]
    platform = sys.argv[3]

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 4:
        model_type = sys.argv[4]
        if model_type not in ['i8', 'u8', 'fp']:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)
        elif model_type in ['i8', 'u8']:
            do_quant = True
        else:
            do_quant = False

    if len(sys.argv) > 5:
        output_path = sys.argv[5]
    else:
        output_path = DEFAULT_RKNN_PATH

    output_filename, _ = os.path.splitext(output_path)
    intermediate_filename = output_filename + '.onnx'

    return model, task, intermediate_filename, platform, do_quant, output_path

if __name__ == '__main__':
    model, task, intermediate_filename, platform, do_quant, output_path = parse_arg()

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(target_platform=platform)
    print('done')

    # Load model
    print('--> Building model')

    if os.path.isdir(intermediate_filename) == False:
        model = DeepFace.build_model(task=task, model_name=model)  # Build the VGG-Face model
    
    print(' model built, summary:')
    model.model.summary()

    img_path='../model/matt.jpeg'

    img_objs = detection.extract_faces(
        img_path=img_path,
        grayscale=False,
    )

    for img_obj in img_objs:
        img_content = img_obj["face"]
        img_region = img_obj["facial_area"]
        img_confidence = img_obj["confidence"]
        if img_content.shape[0] == 0 or img_content.shape[1] == 0:
            continue

        # rgb to bgr
        img_content = img_content[:, :, ::-1]

        # resize input image
        img_content = preprocessing.resize_image(img=img_content, target_size=(224, 224))


    apparent_age = model.predict(img_content)

    print(f'Apparent age {apparent_age}')

    # input_signature = [tf.TensorSpec([None, 224, 224, 3], tf.int8, name='input')]
    # Use from_function for tf functions

    # print('--> Converting from keras to onnx')
    # onnx_model, _ = tf2onnx.convert.from_keras(model=model.model, opset=13)
    
    
    # print('--> Saving model')
    # onnx.save(onnx_model,intermediate_filename)


    # Print the model's input tensors (symbolic)
    # print("Model Input Tensors:")
    # for i in model.model.inputs:
    #     print(f"  Name: {i.name}, Shape: {i.shape}, Dtype: {i.dtype}")
    
    # print('--> Loading model')
    # ret = rknn.load_onnx(intermediate_filename, inputs=['zero_padding2d_input'], input_size_list=[[1, 224,224,3]])

    # if ret != 0:
    #     print('Load model failed!')
    #     exit(ret)
    # print('done')

    # # Build model
    # print('--> Building model')
    # ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)
    # if ret != 0:
    #     print('Build model failed!')
    #     exit(ret)
    # print('done')

    # # Export rknn model
    # print('--> Export rknn model')
    # ret = rknn.export_rknn(output_path)
    # if ret != 0:
    #     print('Export rknn model failed!')
    #     exit(ret)
    # print('done')

    # Release
    rknn.release()