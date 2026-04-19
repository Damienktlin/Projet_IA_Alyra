from PIL import Image
import tensorflow as tf
import numpy as np

IMG_SIZE = (224, 224)


def preprocess_image(path):
    img = Image.open(path).convert("RGB")

    # même logique que dataset
    img.thumbnail(IMG_SIZE)

    new_img = Image.new("RGB", IMG_SIZE, (0, 0, 0))

    x_offset = (IMG_SIZE[0] - img.size[0]) // 2
    y_offset = (IMG_SIZE[1] - img.size[1]) // 2

    new_img.paste(img, (x_offset, y_offset))

    # conversion numpy
    img_array = np.array(new_img)

    # preprocess MobileNet
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # batch
    img_array = np.expand_dims(img_array, axis=0)

    return img_array