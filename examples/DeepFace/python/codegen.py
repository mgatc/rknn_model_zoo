from rknn.api import RKNN

# Create RKNN object
rknn = RKNN(verbose=False)

# Generate C++ sample runtime app
app_path = '../cpp/rknn_app_demo'
print(f'--> Generating sample runtime app: {app_path}')
ret = rknn.codegen(output_path=app_path, overwrite=True)

# Release
rknn.release()