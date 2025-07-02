import sys
import os
from rknn.api import RKNN
from deepface import DeepFace
import tensorflow as tf

DATASET_PATH = '../model/dataset.txt'

if len(sys.argv) < 3:
    print("Usage: python3 {} [task] [model] [dtype] [output_rknn_path(optional)]".format(sys.argv[0]))
    exit(1)

task = sys.argv[1]
model = sys.argv[2]
dtype = sys.argv[3]
platform = 'rk3588'

do_quant = True

if dtype == 'fp':
    do_quant = False

if len(sys.argv) > 4:
    output_path = sys.argv[4]
else:
    output_path = f"../model/{task}.{model}.rknn"

output_filename, _ = os.path.splitext(output_path)
intermediate_filename = output_filename + '.tflite'

# Load model
print('--> Building model')

model = DeepFace.build_model(task=task, model_name=model)  # Build the VGG-Face model

print(' model built, summary:')
model.model.summary()

# Print the model's input tensors (symbolic)
print("Model Input Tensors:")
for i in model.model.inputs:
    print(f"  Name: {i.name}, Shape: {i.shape}, Dtype: {i.dtype}")
print("Model output Tensors:")
for i in model.model.outputs:
    print(f"  Name: {i.name}, Shape: {i.shape}, Dtype: {i.dtype}")

# Convert the model.
print('--> Converting model')

converter = tf.lite.TFLiteConverter.from_keras_model(model.model)
tflite_model = converter.convert()

# Save the model.
with open(intermediate_filename, 'wb') as f:
    f.write(tflite_model)



# Create RKNN object
rknn = RKNN(verbose=False)

# Pre-process config
print('--> Config RKNN')
rknn.config(target_platform=platform, dynamic_input=[[[1,224,224,3]]]) # use [1,48,48,1] for emotion
print('done')


print('--> Loading model from disk')
ret = rknn.load_tflite(intermediate_filename)

if ret != 0:
    print('Load model failed!')
    exit(ret)
print('done')

# Build model
print('--> Building RKNN model')
ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)
if ret != 0:
    print('Build model failed!')
    exit(ret)
print('done')

# Export rknn model
print(f'--> Export rknn model : {output_path}')
ret = rknn.export_rknn(output_path)
if ret != 0:
    print('Export rknn model failed!')
    exit(ret)
print('done')

# Release
rknn.release()