import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dropout,
    Dense,
    Flatten,
    Reshape,
    GRU,
    Bidirectional,
    Concatenate,
    Normalization,
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import MobileNetV2


class MER_CNN_Model:
    def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        return self._model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
        )

    def predict(self, X):
        return self._model.predict(X)

    def summary(self):
        return self._model.summary()


class MER_CNN_Simple(MER_CNN_Model):
    def __init__(self):
        self._model = Sequential(
            [
                Input(shape=(128, None, 1)),
                Normalization(),
                Conv2D(32, (3, 3), activation="relu", padding="same"),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation="relu", padding="same"),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation="relu", padding="same"),
                BatchNormalization(),
                GlobalAveragePooling2D(),
                Dropout(0.5),
                Dense(64, activation="relu"),
                Dense(2, activation="linear"),
            ]
        )
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mean_squared_error",
            metrics=["mae"],
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
                ModelCheckpoint("best_model.h5", save_best_only=True),
            ],
        )


class MER_CNN_VGG_Style(MER_CNN_Model):
    def __init__(self, input_shape=(128, 128, 1)):
        inputs = Input(shape=input_shape)

        x = Normalization()(inputs)

        x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 4))(x)  # frequency pooling

        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)

        outputs = Dense(2, activation="tanh")(x)

        self._model = Model(inputs=inputs, outputs=outputs, name="VGG_Style_CNN")
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mean_squared_error",
            metrics=["mae"],
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
                ModelCheckpoint("best_model_vgg.h5", save_best_only=True),
            ],
        )


class MER_CRNN(MER_CNN_Model):
    def __init__(self, input_shape=(128, 128, 1)):
        inputs = Input(shape=input_shape)

        x = Normalization()(inputs)

        # Feature extraction with CNN
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D((2, 4))(x)
        x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D((2, 4))(x)

        shape_to_reshape = x.shape
        X = Reshape(target_shape=(shape_to_reshape[1], shape_to_reshape[2] * shape_to_reshape[3]))(x)

        # Temporal modeling with RNN
        x = Bidirectional(GRU(128, return_sequences=True))(X)
        x = Bidirectional(GRU(64))(x)

        x = Dense(64, activation="relu")(x)
        x = Dropout(0.4)(x)

        outputs = Dense(2, activation="tanh")(x)

        self._model = Model(inputs=inputs, outputs=outputs, name="CRNN_Model")
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mean_squared_error",
            metrics=["mae"],
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
                ModelCheckpoint("best_model_crnn.h5", save_best_only=True),
            ],
        )


class MER_CNN_MobileNet(MER_CNN_Model):
    def __init__(self, input_shape=(128, 128, 1)):
        inputs = Input(shape=input_shape)

        x = Normalization()(inputs)

        x_3channel = Concatenate(axis=-1)([x, x, x])  # Convert to 3 channels
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
        base_model.trainable = False  # Freeze the base model

        x = base_model(x_3channel)
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.5)(x)

        outputs = Dense(2, activation="tanh")(x)

        self._model = Model(inputs=inputs, outputs=outputs, name="MobileNet_CNN")
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mean_squared_error",
            metrics=["mae"],
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
                ModelCheckpoint("best_model_mobilenet.h5", save_best_only=True),
            ],
        )
