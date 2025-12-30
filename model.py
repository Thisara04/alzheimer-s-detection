import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

IMG_SIZE = 224
NUM_CLASSES = 4


def build_model():
    """
    Rebuilds the EfficientNetB2-based Alzheimer MRI classifier architecture.
    Must EXACTLY match the training-time architecture.
    """

    base_model = EfficientNetB2(
        weights=None,            
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

  
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)

    outputs = Dense(
        NUM_CLASSES,
        activation="softmax",
        dtype="float32"         
    )(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    return model

