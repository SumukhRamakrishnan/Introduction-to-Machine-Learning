import numpy as np
import tensorflow as tf


def main():

    # model
    model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet')
    model.summary()

    # image
    image = tf.keras.preprocessing.image.load_img('picture.jpg', target_size=(299, 299))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.inception_v3.preprocess_input(image_array)

    # prediction
    prediction = model.predict(image_array)
    decoded_prediction = tf.keras.applications.inception_v3.decode_predictions(prediction, top=3)[0]
    print(decoded_prediction)


if __name__ == '__main__':
    main()
