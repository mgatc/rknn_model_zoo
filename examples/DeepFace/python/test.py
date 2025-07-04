import sys

from deepface import DeepFace

from deepface.modules import modeling, detection, preprocessing
from deepface.models.demography import Gender, Race, Emotion

DATASET_PATH = '../model/dataset.txt'
DEFAULT_IMAGE_PATH = '../model/test.jpg'
DEFAULT_QUANT = True

def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} [task] [model] [platform] [dtype(optional)] [image(optional)]".format(sys.argv[0]));
        print("       platform choose from [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b, rv1109, rv1126, rk1808]")
        print("       dtype choose from    [i8] for [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b]")
        print("       dtype choose from    [u8] for [rv1109, rv1126, rk1808]")
        exit(1)

    task = sys.argv[1]
    model_name = sys.argv[2]
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
        image_path = sys.argv[5]
    else:
        image_path = DEFAULT_IMAGE_PATH


    return model_name, task,  platform, do_quant, image_path

if __name__ == '__main__':
    model_name, task,  platform, do_quant, image_path = parse_arg()

    # Load model
    print('--> Building model')

    model = DeepFace.build_model(task=task, model_name=model_name)  # Build the VGG-Face model

    print(' model built, summary:')
    model.model.summary()

    img_objs = detection.extract_faces(
        img_path=image_path,
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

    print(f'Apparent {model_name} {apparent_age}')

    objs = DeepFace.analyze(
        img_path = image_path, actions = ['age', 'gender', 'race', 'emotion']
    )
    print(f'DeepFace result: {objs}')