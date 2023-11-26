import numpy as np
from keras import layers
from keras import models
from keras.datasets import mnist
from sklearn.cluster import KMeans


def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(8, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(4, (3, 3), padding='same', activation='relu'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2DTranspose(8, (3, 3), padding='same', activation='relu'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2DTranspose(16, (3, 3), padding='same', activation='relu'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy')
    model.summary()
    return model


def train(train_images):
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255
    model = create_model(input_shape=(28, 28, 1))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(train_images, train_images, epochs=5, batch_size=64)
    model.save_weights('./weights.ckpt')


def load():
    model1 = create_model(input_shape=(28, 28, 1))
    status = model1.load_weights('./weights.ckpt')
    status.expect_partial()
    model1.pop()
    model1.pop()
    model1.pop()
    model1.pop()
    model1.summary()
    return model1


def modelKMeans(train_data, y_train):
    model = KMeans(n_clusters=10)
    pred = model.fit_predict(train_data)
    pred_labels = {}
    for p in pred:
        if str(p) in pred_labels.keys():
            pred_labels[str(p)] += 1
        else:
            pred_labels[str(p)] = 1
    train_labels = {}
    for p in y_train:
        if str(p) in train_labels.keys():
            train_labels[str(p)] += 1
        else:
            train_labels[str(p)] = 1
    for p, l in zip(pred_labels.items(), train_labels.items()):
        if l[1] < p[1]:
            print(f'{p[0]}: {int(l[1])/int(p[1])}')
        else:
            print(f'{p[0]}: {int(p[1])/int(l[1])}')


def main():
    (X_train, y_train), (_, _) = mnist.load_data()
    # train(X_train)
    model = load()
    y_pred = model.predict(X_train)

    print(y_pred.shape)
    print('result shape = {}'.format(y_pred.shape))
    a, b, c, d = y_pred.shape
    code = y_pred.reshape(a, b * c * d)
    print('code shape = {}'.format(code.shape))
    modelKMeans(code, y_train)


if __name__ == '__main__':
    main()