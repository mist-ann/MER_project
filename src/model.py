import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    GlobalAveragePooling2D,
    GlobalAveragePooling1D,
    GlobalMaxPooling2D,
    GlobalMaxPooling1D,
    Dropout,
    Dense,
    Flatten,
    Reshape,
    GRU,
    LSTM,
    Bidirectional,
    Concatenate,
    Normalization,
    Permute,
    Add,
    Activation,
    SpatialDropout2D,
    LayerNormalization,
    MultiHeadAttention,
    Resizing,
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import MobileNetV2


class MER_CNN_Model:
    def fit(self, train_generator, validation_generator, epochs=50, callbacks=None):
        return self._model.fit(
            train_generator, validation_data=validation_generator, epochs=epochs, callbacks=callbacks
        )

    def predict(self, X, **kwargs):
        return self._model.predict(X, **kwargs)

    def summary(self):
        return self._model.summary()


class MER_CNN_Simple(MER_CNN_Model):
    def __init__(self, input_shape=(128, 128, 1), learning_rate=0.001):
        self._model = Sequential(
            [
                Input(shape=input_shape),
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
        )


class MER_CNN_VGG_Style(MER_CNN_Model):
    def __init__(self, input_shape=(128, 128, 1), learning_rate=0.001):
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
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mean_squared_error",
            metrics=["mae"],
        )


def _compile_regression_model(model, learning_rate=0.001):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    return model


def _conv_bn_relu(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    return Activation("relu")(x)


def _residual_block(x, filters, pool_size=None, dropout=0.0):
    shortcut = x

    x = _conv_bn_relu(x, filters)
    x = Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding="same", use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)

    if pool_size is not None:
        x = MaxPooling2D(pool_size)(x)
    if dropout > 0:
        x = SpatialDropout2D(dropout)(x)

    return x


class MER_CNN_Residual(MER_CNN_Model):
    """
    Stronger CNN baseline with residual blocks and average+max global pooling.

    This model keeps the same dynamic-window setup as MER_CNN_Simple, but has
    higher capacity and better gradient flow.
    """

    def __init__(self, input_shape=(128, 128, 1), learning_rate=0.001):
        inputs = Input(shape=input_shape)

        x = BatchNormalization()(inputs)
        x = _residual_block(x, 32, pool_size=(2, 1), dropout=0.05)
        x = _residual_block(x, 64, pool_size=(2, 2), dropout=0.10)
        x = _residual_block(x, 128, pool_size=(2, 2), dropout=0.15)

        final_pool = (2, 2) if input_shape[1] >= 64 else (2, 1)
        x = _residual_block(x, 256, pool_size=final_pool, dropout=0.20)

        avg_pool = GlobalAveragePooling2D()(x)
        max_pool = GlobalMaxPooling2D()(x)
        x = Concatenate()([avg_pool, max_pool])

        x = Dense(256, activation="relu")(x)
        x = Dropout(0.40)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.20)(x)

        outputs = Dense(2, activation="tanh")(x)

        self._model = Model(inputs=inputs, outputs=outputs, name="MER_CNN_Residual")
        _compile_regression_model(self._model, learning_rate=learning_rate)


class MER_CRNN_Temporal(MER_CNN_Model):
    """
    CNN + BiGRU model that treats spectrogram columns as the temporal sequence.

    Unlike the older MER_CRNN class, this model preserves the time axis during
    convolutional pooling and runs the recurrent layers over time frames.
    """

    def __init__(self, input_shape=(128, 128, 1), learning_rate=0.001):
        inputs = Input(shape=input_shape)

        x = BatchNormalization()(inputs)

        x = _conv_bn_relu(x, 32, kernel_size=(5, 3))
        x = MaxPooling2D((2, 1))(x)
        x = SpatialDropout2D(0.05)(x)

        x = _conv_bn_relu(x, 64)
        x = MaxPooling2D((2, 1))(x)
        x = SpatialDropout2D(0.10)(x)

        x = _conv_bn_relu(x, 128)
        x = MaxPooling2D((2, 1))(x)
        x = SpatialDropout2D(0.15)(x)

        x = _conv_bn_relu(x, 128)
        x = MaxPooling2D((2, 1))(x)

        x = Permute((2, 1, 3))(x)
        sequence_features = int(x.shape[2]) * int(x.shape[3])
        x = Reshape((input_shape[1], sequence_features))(x)

        x = Bidirectional(GRU(96, return_sequences=True, dropout=0.20))(x)
        x = Bidirectional(GRU(64, return_sequences=False, dropout=0.20))(x)

        x = Dense(128, activation="relu")(x)
        x = Dropout(0.40)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.20)(x)

        outputs = Dense(2, activation="tanh")(x)

        self._model = Model(inputs=inputs, outputs=outputs, name="MER_CRNN_Temporal")
        _compile_regression_model(self._model, learning_rate=learning_rate)


class MER_CNN_Attention(MER_CNN_Model):
    """
    Convolutional front-end followed by self-attention over time frames.
    """

    def __init__(self, input_shape=(128, 128, 1), learning_rate=0.001):
        inputs = Input(shape=input_shape)

        x = BatchNormalization()(inputs)
        x = _conv_bn_relu(x, 32)
        x = MaxPooling2D((2, 1))(x)
        x = _conv_bn_relu(x, 64)
        x = MaxPooling2D((2, 1))(x)
        x = _conv_bn_relu(x, 128)
        x = MaxPooling2D((2, 1))(x)

        x = Permute((2, 1, 3))(x)
        sequence_features = int(x.shape[2]) * int(x.shape[3])
        x = Reshape((input_shape[1], sequence_features))(x)
        x = Dense(128, activation="relu")(x)

        attention = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.10)(x, x)
        x = Add()([x, attention])
        x = LayerNormalization()(x)

        feed_forward = Dense(256, activation="relu")(x)
        feed_forward = Dropout(0.20)(feed_forward)
        feed_forward = Dense(128)(feed_forward)
        x = Add()([x, feed_forward])
        x = LayerNormalization()(x)

        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = Concatenate()([avg_pool, max_pool])

        x = Dense(128, activation="relu")(x)
        x = Dropout(0.40)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.20)(x)

        outputs = Dense(2, activation="tanh")(x)

        self._model = Model(inputs=inputs, outputs=outputs, name="MER_CNN_Attention")
        _compile_regression_model(self._model, learning_rate=learning_rate)


class MER_CNN_LSTM(MER_CNN_Model):
    """
    CNN feature extractor followed by BiLSTM over spectrogram time frames.
    """

    def __init__(self, input_shape=(128, 128, 1), learning_rate=0.001):
        inputs = Input(shape=input_shape)

        x = BatchNormalization()(inputs)

        x = _conv_bn_relu(x, 32, kernel_size=(5, 3))
        x = MaxPooling2D((2, 1))(x)
        x = SpatialDropout2D(0.05)(x)

        x = _conv_bn_relu(x, 64)
        x = MaxPooling2D((2, 1))(x)
        x = SpatialDropout2D(0.10)(x)

        x = _conv_bn_relu(x, 128)
        x = MaxPooling2D((2, 1))(x)
        x = SpatialDropout2D(0.15)(x)

        x = Permute((2, 1, 3))(x)
        sequence_features = int(x.shape[2]) * int(x.shape[3])
        x = Reshape((input_shape[1], sequence_features))(x)

        x = Bidirectional(LSTM(96, return_sequences=True, dropout=0.20))(x)
        x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.20))(x)

        x = Dense(128, activation="relu")(x)
        x = Dropout(0.40)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.20)(x)

        outputs = Dense(2, activation="tanh")(x)

        self._model = Model(inputs=inputs, outputs=outputs, name="MER_CNN_LSTM")
        _compile_regression_model(self._model, learning_rate=learning_rate)


class MER_CRNN(MER_CNN_Model):
    def __init__(self, input_shape=(128, 128, 1), learning_rate=0.001):
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
        )


class MER_CNN_MobileNet(MER_CNN_Model):
    def __init__(
        self,
        input_shape=(128, 128, 1),
        learning_rate=0.001,
        resize_shape=(96, 96),
        alpha=0.35,
        weights=None,
    ):
        inputs = Input(shape=input_shape)

        x = BatchNormalization()(inputs)
        x = Resizing(resize_shape[0], resize_shape[1])(x)

        x_3channel = Concatenate(axis=-1)([x, x, x])
        base_model = MobileNetV2(
            weights=weights,
            include_top=False,
            input_shape=(resize_shape[0], resize_shape[1], 3),
            alpha=alpha,
        )
        base_model.trainable = weights is None

        x = base_model(x_3channel)
        avg_pool = GlobalAveragePooling2D()(x)
        max_pool = GlobalMaxPooling2D()(x)
        x = Concatenate()([avg_pool, max_pool])
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.40)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.20)(x)

        outputs = Dense(2, activation="tanh")(x)

        self._model = Model(inputs=inputs, outputs=outputs, name="MobileNet_CNN")
        _compile_regression_model(self._model, learning_rate=learning_rate)


class MER_LSTM(MER_CNN_Model):
    """
    Pure LSTM model for dynamic MER regression.

    Expected input shape from existing DEAMSegmentGenerator:
        (128, 128, 1)

    Interpretation:
        128 mel bins x 128 time frames x 1 channel

    The model converts the input into a sequence:
        128 time frames x 128 mel features

    Output:
        [valence, arousal]
    """

    def __init__(self, input_shape=(128, 128, 1), learning_rate=0.001):
        inputs = Input(shape=input_shape)

        x = Normalization()(inputs)

        # (mel_bins, time_frames, 1) -> (mel_bins, time_frames)
        x = Reshape((input_shape[0], input_shape[1]))(x)

        # (mel_bins, time_frames) -> (time_frames, mel_bins)
        x = Permute((2, 1))(x)

        x = LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.0)(x)
        x = LSTM(64, return_sequences=False, dropout=0.25, recurrent_dropout=0.0)(x)

        x = Dense(64, activation="relu")(x)
        x = Dropout(0.4)(x)

        outputs = Dense(2, activation="tanh")(x)

        self._model = Model(inputs=inputs, outputs=outputs, name="MER_LSTM")
        _compile_regression_model(self._model, learning_rate=learning_rate)
