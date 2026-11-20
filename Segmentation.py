import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2


def residual_block(x, filters):

    shortcut = x

    x = layers.Conv2D(
        filters, 3,
        padding='same',
        activation='relu'
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters, 3,
        padding='same'
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Add()([shortcut, x])

    x = layers.Activation('relu')(x)

    return x

# ============================================================
# 5. RESIDUAL AUTOENCODER
# ============================================================

def residual_autoencoder(input_shape, filters):

    inputs = layers.Input(shape=input_shape)

    # ---------------- ENCODER ----------------

    x = layers.Conv2D(
        filters, 3,
        strides=2,
        padding='same',
        activation='relu'
    )(inputs)

    x = residual_block(x, filters)

    x = layers.Conv2D(
        filters * 2, 3,
        strides=2,
        padding='same',
        activation='relu'
    )(x)

    x = residual_block(x, filters * 2)

    # ---------------- BOTTLENECK ----------------

    encoded = layers.Conv2D(
        filters * 4, 3,
        padding='same',
        activation='relu',
        name='encoded_features'
    )(x)

    # ---------------- DECODER ----------------

    x = layers.Conv2DTranspose(
        filters * 2, 3,
        strides=2,
        padding='same',
        activation='relu'
    )(encoded)

    x = residual_block(x, filters * 2)

    x = layers.Conv2DTranspose(
        filters, 3,
        strides=2,
        padding='same',
        activation='relu'
    )(x)

    x = residual_block(x, filters)

    # ---------------- OUTPUT ----------------

    outputs = layers.Conv2D(
        1,
        3,
        padding='same',
        activation='sigmoid',
        name='segmentation_output'
    )(x)

    model = models.Model(
        inputs,
        outputs,
        name='SBOA_Residual_Autoencoder'
    )

    return model

# --------------------- Improved K-Means on Features ---------------------
def improved_kmeans(features, n_clusters=4):
    h, w, c = features.shape
    flat_features = features.reshape(-1, c)

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=42)
    labels = kmeans.fit_predict(flat_features)
    return labels.reshape(h, w)

# --------------------- Combined Model Pipeline ---------------------
def segment_with_residual_autoencoder(image, autoencoder_model, n_clusters=4):
    # Normalize and reshape input
    input_img = image.astype('float32') / 255.0
    if len(input_img.shape) == 2:
        input_img = np.expand_dims(input_img, axis=-1)
    input_img = np.expand_dims(input_img, axis=0)  # (1, H, W, C)

    # Extract bottleneck features
    feature_extractor = models.Model(
        inputs=autoencoder_model.input,
        outputs=autoencoder_model.get_layer('encoded_features').output
    )
    bottleneck_features = feature_extractor.predict(input_img)[0]  # remove batch

    # K-Means on encoded features
    segmentation_mask = improved_kmeans(bottleneck_features, n_clusters=n_clusters)
    return segmentation_mask

