import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

animals_classes = [2, 3, 4, 5, 6, 7]
vehicles_classes = [0, 1, 8, 9]

classes_to_keep = animals_classes + vehicles_classes;

train_indices = [i for i, y in enumerate(y_train) if y[0] in classes_to_keep]
test_indices = [i for i, y in enumerate(y_test) if y[0] in classes_to_keep]

train_indices = train_indices[:int(0.3 * len(train_indices))]
test_indices = test_indices[:int(0.7 * len(test_indices))]

x_train = x_train[train_indices]
y_train = y_train[train_indices]
x_test = x_test[test_indices]
y_test = y_test[test_indices]

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

num_classes = 2

y_train = keras.utils.to_categorical(
    [0 if y[0] in animals_classes else 1 for y in y_train],
    num_classes
)
y_test = keras.utils.to_categorical(
    [0 if y[0] in animals_classes else 1 for y in y_test],
    num_classes
)

classifiers = [
    (1, 32),
    (2, 64),
    (3, 128)
]

for num_conv_layers, num_filters in classifiers:
    model = Sequential()

    for _ in range(num_conv_layers):
        model.add(Conv2D(num_filters, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(f"Training classifier with {num_conv_layers} convolutional layer(s) and {num_filters} filters")
    model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1, validation_data=(x_test, y_test))
