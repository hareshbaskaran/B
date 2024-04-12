from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
from io import BytesIO

model = ResNet50(weights="imagenet")


def read_image(file) -> Image.Image:
    pil_image = Image.open(BytesIO(file))
    return pil_image


def transformacao(file: Image.Image):
    img = np.asarray(file.resize((224, 224)))[..., :3]
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print("Predicted:", decode_predictions(preds, top=3)[0])
    result = decode_predictions(model.predict(x), 3)[0]

    txt = "can be "
    for i, res in enumerate(result):
        resp = {}
        resp["class"] = res[1]
        txt = txt + "a " + resp["class"] + " , "
    return txt
