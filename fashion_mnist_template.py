import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
import tensorflow as tf

fashion_class_labels = {0: 'T-shirt/top',
                        1: 'Trouser',
                        2: 'Pullover',
                        3: 'Dress',
                        4: 'Coat',
                        5: 'Sandal',
                        6: 'Shirt',
                        7: 'Sneaker',
                        8: 'Bag',
                        9: 'Ankle boot'}


def main():
    train_images, train_labels, test_images, test_labels = load_fashion_data()
    train_images = np.reshape(train_images, (-1, 28, 28, 1))
    test_images = np.reshape(test_images, (-1, 28, 28, 1))

    model = create_advanced_model()

    train_images = train_images[:600]
    train_labels = train_labels[:600]

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10)
    model.summary()

    model.evaluate(test_images, test_labels)
    predict(model, test_images)


def load_fashion_data():
    # Load original fashion data.
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    # Shuffle data.
    train_images, train_labels = skl.utils.shuffle(train_images, train_labels)
    test_images, test_labels = skl.utils.shuffle(test_images, test_labels)

    print('train_images.shape:', train_images.shape)
    print('test_images.shape:', test_images.shape)
    plt.imshow(test_images[0], cmap='Greys')
    plt.colorbar()
    plt.show()

    # Normalize images.
    train_images = train_images / 255.
    test_images = test_images / 255.

    plt.imshow(test_images[0], cmap='Greys')
    plt.colorbar()
    plt.show()

    print('train_labels:\n', train_labels)
    print('train_labels.shape:\n', train_labels.shape)
    print('test_labels:\n', test_labels)
    print('test_labels.shape:\n', test_labels.shape)

    plt.imshow(test_images[0], cmap='Greys')
    plt.title(fashion_class_labels[test_labels[0]])
    plt.show()

    return train_images, train_labels, test_images, test_labels


def create_simple_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model


def create_advanced_model():
    model = tf.keras.models.Sequential()

    # convolution layer 1
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=[5, 5], activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2]))

    # convolution layer 2
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=[5, 5], activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2]))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model


def predict(model, test_images):
    label_index_predictions = model.predict(test_images)[0]
    print('Label index predictions:', label_index_predictions)
    print('Predicted fashion category:', fashion_class_labels[int(np.argmax(label_index_predictions))])


if __name__ == '__main__':
    main()



























































