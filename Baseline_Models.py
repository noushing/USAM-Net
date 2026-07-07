# ============================================================
# Baseline Models for U-SAMNet Comparison
# Manuscript ID: applsci-4356152
# Journal: Applied Sciences (MDPI)
#
# Authors: Prosenjit Roy, Kijoon Lee, Mohsen Taheri Andani,
#          Toan Truong, Haojun You, Noushin Ghaffari
#
# Description: Implementation of all 9 baseline models used
# for comparison with U-SAMNet in the paper.
#
# Baselines:
#   Standard Segmentation:
#     1. U-Net
#     2. VGG16-UNet
#     3. ResNet50-UNet
#     4. HRNet
#     5. DeepLabV3+
#   Uncertainty-Aware and Attention-Based:
#     6. MC-Dropout U-Net
#     7. Attention U-Net
#     8. SE-U-Net
#     9. CBAM-U-Net
#
# Requirements:
#   - Python 3.9
#   - TensorFlow 2.10
#   - scikit-learn
#   - numpy
#   - CUDA 11.2, cuDNN 8.1
#
# Hardware: Bridges-2 Supercomputing System
#           Pittsburgh Supercomputing Center
#           NVIDIA Tesla V100 32GB GPU
#
# Training Protocol (identical for all models):
#   - Optimizer: Adam
#   - Epochs: 50
#   - Batch size: 16
#   - LR scheduler: ReduceLROnPlateau
#     (factor=0.5, patience=5, min_lr=1e-6, mode=min)
#   - Training set: 7,110 images (1,110 real + 6,000 GAN)
#   - Test set: 139 held-out real images
#   - Threshold: 0.4
# ============================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Conv2DTranspose,
    concatenate, Lambda, Dropout, BatchNormalization,
    Activation, UpSampling2D, Concatenate, Add,
    AveragePooling2D, GlobalAveragePooling2D,
    GlobalMaxPooling2D, Dense, Multiply, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import (
    load_img, img_to_array)
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix)

# ============================================================
# DATA LOADING
# ============================================================
def load_data(folder, size=(256, 256)):
    """
    Load images and masks from folder.
    
    Args:
        folder: Path containing images/ and masks/ subfolders
        size: Target image size (H, W)
    
    Returns:
        images: Normalized array [N, H, W, 1]
        masks: Normalized array [N, H, W, 1]
    """
    images, masks = [], []
    img_path  = os.path.join(folder, 'images')
    mask_path = os.path.join(folder, 'masks')

    for file in sorted(os.listdir(img_path)):
        img = img_to_array(load_img(
            os.path.join(img_path, file),
            color_mode="grayscale",
            target_size=size)) / 255.0
        mask = img_to_array(load_img(
            os.path.join(mask_path, file),
            color_mode="grayscale",
            target_size=size)) / 255.0
        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

# ============================================================
# DATASET PATHS
# ============================================================
combined_dataset = '/content/drive/MyDrive/ML/Unetcomb'
train_folder = os.path.join(combined_dataset, 'train')
test_folder  = os.path.join(combined_dataset, 'test')

X_train, Y_train = load_data(train_folder)
X_test,  Y_test  = load_data(test_folder)

print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")

# ============================================================
# SHARED EVALUATION FUNCTION
# ============================================================
def evaluate_model(model, X_test, Y_test, 
                   model_name, threshold=0.4):
    """
    Evaluate model and print all metrics reported in paper.
    
    Args:
        model: Trained Keras model
        X_test: Test images
        Y_test: Test masks
        model_name: Name for printing
        threshold: Binarization threshold (0.4 as in paper)
    """
    Y_pred     = model.predict(X_test)
    Y_pred_bin = (Y_pred > threshold).astype(np.uint8)
    Y_test_bin = (Y_test > threshold).astype(np.uint8)
    y_t        = Y_test_bin.flatten()
    y_p        = Y_pred_bin.flatten()

    cm             = confusion_matrix(y_t, y_p)
    tn, fp, fn, tp = cm.ravel()
    iou            = tp / (tp + fp + fn)

    acc  = accuracy_score(y_t, y_p)
    prec = precision_score(y_t, y_p, zero_division=0)
    rec  = recall_score(y_t, y_p, zero_division=0)
    f1   = f1_score(y_t, y_p, zero_division=0)

    print(f"\n{'='*50}")
    print(f"{model_name} Results:")
    print(f"{'='*50}")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print(f"IoU:       {iou*100:.2f}%")

    return {
        'accuracy':  acc,
        'precision': prec,
        'recall':    rec,
        'f1':        f1,
        'iou':       iou
    }

# ============================================================
# SHARED TRAINING FUNCTION
# ============================================================
def train_baseline(model, model_name, 
                   X_train, Y_train, X_test, Y_test):
    """
    Train baseline model with identical protocol as U-SAMNet.
    
    Protocol:
        - Adam optimizer
        - Binary crossentropy loss
        - 50 epochs, batch size 16
        - ReduceLROnPlateau: factor=0.5, patience=5,
          min_lr=1e-6, mode=min
    """
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        mode='min')

    model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=50,
        batch_size=16,
        callbacks=[reduce_lr],
        verbose=1)

    save_path = f'/content/{model_name.lower().replace(" ", "_")}.h5'
    model.save(save_path)
    print(f"Model saved: {save_path}")

    return evaluate_model(model, X_test, Y_test, model_name)

# ============================================================
# MODEL 1: U-Net
# Standard encoder-decoder baseline for pixel-level
# segmentation. Selected as the foundational baseline.
# Parameters: 7.69M
# ============================================================
def unet_model(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D(2)(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D(2)(c2)

    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D(2)(c3)

    # Bottleneck
    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)

    # Decoder
    u5 = Conv2DTranspose(256, 2, strides=2, padding='same')(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, 3, activation='relu', padding='same')(u5)
    c5 = Conv2D(256, 3, activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(128, 2, strides=2, padding='same')(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(128, 3, activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, 2, strides=2, padding='same')(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(64, 3, activation='relu', padding='same')(c7)

    outputs = Conv2D(1, 1, activation='sigmoid')(c7)
    return Model(inputs, outputs)

# ============================================================
# MODEL 2: VGG16-UNet
# Pre-trained VGG16 encoder with U-Net decoder.
# Selected to evaluate benefit of ImageNet pre-training.
# Parameters: 38.78M
# ============================================================
def vgg16_unet_model(input_size=(256, 256, 1)):
    inputs = Input(shape=input_size)

    # Convert grayscale to RGB for VGG16 compatibility
    x = Lambda(lambda x: tf.repeat(x, 3, axis=-1))(inputs)

    base_model = VGG16(
        input_shape=(input_size[0], input_size[1], 3),
        include_top=False,
        weights='imagenet')

    for layer in base_model.layers:
        layer.trainable = True

    # Skip connections from VGG16 blocks
    block1 = base_model.get_layer('block1_conv2').output
    block2 = base_model.get_layer('block2_conv2').output
    block3 = base_model.get_layer('block3_conv3').output
    block4 = base_model.get_layer('block4_conv3').output
    block5 = base_model.get_layer('block5_conv3').output

    encoder = Model(
        inputs=base_model.input,
        outputs=[block1, block2, block3, block4, block5])
    f1, f2, f3, f4, f5 = encoder(x)

    # Decoder
    u6 = Conv2DTranspose(512, 2, strides=2, padding='same')(f5)
    u6 = concatenate([u6, f4])
    c6 = Conv2D(512, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(512, 3, activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
    u7 = concatenate([u7, f3])
    c7 = Conv2D(256, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(256, 3, activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
    u8 = concatenate([u8, f2])
    c8 = Conv2D(128, 3, activation='relu', padding='same')(u8)
    c8 = Conv2D(128, 3, activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
    u9 = concatenate([u9, f1])
    c9 = Conv2D(64, 3, activation='relu', padding='same')(u9)
    c9 = Conv2D(64, 3, activation='relu', padding='same')(c9)

    outputs = Conv2D(1, 1, activation='sigmoid')(c9)
    return Model(inputs=inputs, outputs=outputs)

# ============================================================
# MODEL 3: ResNet50-UNet
# Pre-trained ResNet50 encoder with U-Net decoder.
# Selected to evaluate residual learning with deep features.
# Parameters: 45.76M
# ============================================================
def resnet50_unet_model(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    # Convert grayscale to RGB for ResNet50 compatibility
    x = Lambda(
        lambda x: tf.repeat(x, 3, axis=-1),
        output_shape=(input_size[0], input_size[1], 3))(inputs)

    base_model = ResNet50(
        input_shape=(input_size[0], input_size[1], 3),
        include_top=False,
        weights='imagenet')

    layer_names = [
        'conv1_relu',
        'conv2_block3_out',
        'conv3_block4_out',
        'conv4_block6_out',
        'conv5_block3_out',
    ]
    skip_outputs = [
        base_model.get_layer(n).output for n in layer_names]
    encoder = Model(
        inputs=base_model.input, outputs=skip_outputs)
    skips = encoder(x)

    bridge = skips[-1]

    u5 = Conv2DTranspose(256, 2, strides=2, padding='same')(bridge)
    u5 = concatenate([u5, skips[3]])
    c5 = Conv2D(256, 3, activation='relu', padding='same')(u5)
    c5 = Conv2D(256, 3, activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(128, 2, strides=2, padding='same')(c5)
    u6 = concatenate([u6, skips[2]])
    c6 = Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(128, 3, activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, 2, strides=2, padding='same')(c6)
    u7 = concatenate([u7, skips[1]])
    c7 = Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(64, 3, activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(32, 2, strides=2, padding='same')(c7)
    u8 = concatenate([u8, skips[0]])
    c8 = Conv2D(32, 3, activation='relu', padding='same')(u8)
    c8 = Conv2D(32, 3, activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(16, 2, strides=2, padding='same')(c8)
    c9 = Conv2D(16, 3, activation='relu', padding='same')(u9)
    c9 = Conv2D(16, 3, activation='relu', padding='same')(c9)

    outputs = Conv2D(1, 1, activation='sigmoid')(c9)
    return Model(inputs=inputs, outputs=outputs)

# ============================================================
# MODEL 4: HRNet
# Multi-resolution network with parallel streams.
# Selected to evaluate multi-scale feature representation
# relevant for detecting pores of varying sizes.
# Parameters: 54.00M
# ============================================================
def hrnet_model(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    # Stem
    x    = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x    = BatchNormalization()(x)
    x    = Conv2D(64, 3, activation='relu', padding='same')(x)
    stem = BatchNormalization()(x)

    # Stage 1
    x        = Conv2D(64, 3, activation='relu', padding='same')(stem)
    x        = BatchNormalization()(x)
    x        = Conv2D(64, 3, activation='relu', padding='same')(x)
    high_1   = BatchNormalization()(x)

    # Stage 2 — high and medium streams
    high_x  = Conv2D(32, 3, activation='relu', padding='same')(high_1)
    high_x  = BatchNormalization()(high_x)
    high_x  = Conv2D(32, 3, activation='relu', padding='same')(high_x)
    high_x  = BatchNormalization()(high_x)

    med_x   = MaxPooling2D(2)(high_1)
    med_x   = Conv2D(64, 3, activation='relu', padding='same')(med_x)
    med_x   = BatchNormalization()(med_x)
    med_x   = Conv2D(64, 3, activation='relu', padding='same')(med_x)
    med_x   = BatchNormalization()(med_x)

    m2h     = UpSampling2D(2)(med_x)
    m2h     = Conv2D(32, 1, activation='relu', padding='same')(m2h)
    h2m     = MaxPooling2D(2)(high_x)
    h2m     = Conv2D(64, 1, activation='relu', padding='same')(h2m)

    high_2  = Add()([high_x, m2h])
    med_2   = Add()([med_x, h2m])

    # Stage 3 — high, medium, low streams
    high_x  = Conv2D(32, 3, activation='relu', padding='same')(high_2)
    high_x  = BatchNormalization()(high_x)
    high_x  = Conv2D(32, 3, activation='relu', padding='same')(high_x)
    high_x  = BatchNormalization()(high_x)

    med_x   = Conv2D(64, 3, activation='relu', padding='same')(med_2)
    med_x   = BatchNormalization()(med_x)
    med_x   = Conv2D(64, 3, activation='relu', padding='same')(med_x)
    med_x   = BatchNormalization()(med_x)

    low_x   = MaxPooling2D(2)(med_2)
    low_x   = Conv2D(128, 3, activation='relu', padding='same')(low_x)
    low_x   = BatchNormalization()(low_x)
    low_x   = Conv2D(128, 3, activation='relu', padding='same')(low_x)
    low_x   = BatchNormalization()(low_x)

    l2m     = UpSampling2D(2)(low_x)
    l2m     = Conv2D(64, 1, activation='relu', padding='same')(l2m)
    l2h     = UpSampling2D(2)(low_x)
    l2h     = Conv2D(64, 1, activation='relu', padding='same')(l2h)
    l2h     = UpSampling2D(2)(l2h)
    l2h     = Conv2D(32, 1, activation='relu', padding='same')(l2h)
    m2h     = UpSampling2D(2)(med_x)
    m2h     = Conv2D(32, 1, activation='relu', padding='same')(m2h)
    m2l     = MaxPooling2D(2)(med_x)
    m2l     = Conv2D(128, 1, activation='relu', padding='same')(m2l)
    h2m     = MaxPooling2D(2)(high_x)
    h2m     = Conv2D(64, 1, activation='relu', padding='same')(h2m)
    h2l     = MaxPooling2D(2)(high_x)
    h2l     = Conv2D(64, 1, activation='relu', padding='same')(h2l)
    h2l     = MaxPooling2D(2)(h2l)
    h2l     = Conv2D(128, 1, activation='relu', padding='same')(h2l)

    high_3  = Add()([high_x, m2h, l2h])
    med_3   = Add()([med_x, h2m, l2m])
    low_3   = Add()([low_x, h2l, m2l])

    # Final fusion
    f_high  = high_3
    f_med   = UpSampling2D(2)(med_3)
    f_low   = UpSampling2D(4)(low_3)
    fusion  = Concatenate()([f_high, f_med, f_low])

    x       = Conv2D(64, 3, activation='relu', padding='same')(fusion)
    x       = Conv2D(64, 3, activation='relu', padding='same')(x)
    outputs = Conv2D(1, 1, activation='sigmoid')(x)
    return Model(inputs, outputs)

# ============================================================
# MODEL 5: DeepLabV3+
# ASPP-based semantic segmentation.
# Selected to evaluate dilated convolution approach.
# Parameters: 13.31M
# ============================================================
def conv_bn_relu(x, filters, kernel_size=3, 
                 dilation_rate=1, padding='same'):
    x = Conv2D(filters, kernel_size,
               dilation_rate=dilation_rate,
               padding=padding, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def aspp_block(x, filters=256):
    shape  = tf.keras.backend.int_shape(x)
    b1     = conv_bn_relu(x, filters, kernel_size=1)
    b2     = conv_bn_relu(x, filters, kernel_size=3, 
                          dilation_rate=6)
    b3     = conv_bn_relu(x, filters, kernel_size=3, 
                          dilation_rate=12)
    pool   = AveragePooling2D(
        pool_size=(shape[1], shape[2]))(x)
    pool   = conv_bn_relu(pool, filters, kernel_size=1)
    pool   = UpSampling2D(
        size=(shape[1], shape[2]), 
        interpolation='bilinear')(pool)
    x      = Concatenate()([b1, b2, b3, pool])
    x      = conv_bn_relu(x, filters, kernel_size=1)
    return x

def deeplabv3plus_model(input_size=(256, 256, 1)):
    inputs = Input(shape=input_size)

    x      = conv_bn_relu(inputs, 64)
    x      = conv_bn_relu(x, 64)
    skip1  = x
    x      = MaxPooling2D(2)(x)

    x      = conv_bn_relu(x, 128)
    x      = conv_bn_relu(x, 128)
    skip2  = x
    x      = MaxPooling2D(2)(x)

    x      = conv_bn_relu(x, 256)
    x      = conv_bn_relu(x, 256)
    low_f  = x
    x      = MaxPooling2D(2)(x)

    x      = conv_bn_relu(x, 512)
    x      = conv_bn_relu(x, 512)
    x      = MaxPooling2D(2)(x)

    x      = aspp_block(x, filters=256)
    x      = UpSampling2D(size=(4, 4), 
                          interpolation='bilinear')(x)

    low_p  = conv_bn_relu(low_f, 48, kernel_size=1)
    x      = Concatenate()([x, low_p])
    x      = conv_bn_relu(x, 256)
    x      = conv_bn_relu(x, 256)
    x      = UpSampling2D(size=(4, 4), 
                          interpolation='bilinear')(x)

    outputs = Conv2D(1, 1, padding='same', 
                     activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=outputs)

# ============================================================
# MODEL 6: MC-Dropout U-Net
# U-Net with MC-dropout for uncertainty estimation.
# Selected to isolate the effect of MC-dropout uncertainty
# without uncertainty-guided attention suppression.
# ============================================================
def mc_dropout_unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    x      = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x      = BatchNormalization()(x)
    x      = Conv2D(64, 3, activation='relu', padding='same')(x)
    x      = BatchNormalization()(x)
    x      = Dropout(0.5)(x, training=True)
    skip1  = x

    x      = MaxPooling2D()(x)
    x      = Conv2D(128, 3, activation='relu', padding='same')(x)
    x      = BatchNormalization()(x)
    x      = Conv2D(128, 3, activation='relu', padding='same')(x)
    x      = BatchNormalization()(x)
    x      = Dropout(0.5)(x, training=True)
    skip2  = x

    x      = MaxPooling2D()(x)
    x      = Conv2D(256, 3, activation='relu', padding='same')(x)
    x      = BatchNormalization()(x)
    x      = Dropout(0.5)(x, training=True)

    x      = UpSampling2D()(x)
    x      = Concatenate()([x, skip2])
    x      = Conv2D(128, 3, activation='relu', padding='same')(x)
    x      = BatchNormalization()(x)

    x      = UpSampling2D()(x)
    x      = Concatenate()([x, skip1])
    x      = Conv2D(64, 3, activation='relu', padding='same')(x)
    x      = BatchNormalization()(x)

    output = Conv2D(1, 1, activation='sigmoid')(x)
    return Model(inputs, output)

# ============================================================
# MODEL 7: Attention U-Net
# U-Net with attention gates at skip connections.
# Selected to isolate the effect of spatial attention
# without uncertainty guidance.
# ============================================================
def attention_gate(x, g, inter_channels):
    theta = Conv2D(inter_channels, 1, padding='same')(x)
    phi   = Conv2D(inter_channels, 1, padding='same')(g)
    phi   = UpSampling2D(size=(
        theta.shape[1] // phi.shape[1],
        theta.shape[2] // phi.shape[2]))(phi)
    add   = Activation('relu')(Add()([theta, phi]))
    psi   = Conv2D(1, 1, activation='sigmoid', 
                   padding='same')(add)
    return Multiply()([x, psi])

def attention_unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    x      = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x      = BatchNormalization()(x)
    x      = Conv2D(64, 3, activation='relu', padding='same')(x)
    skip1  = BatchNormalization()(x)

    x      = MaxPooling2D()(skip1)
    x      = Conv2D(128, 3, activation='relu', padding='same')(x)
    x      = BatchNormalization()(x)
    x      = Conv2D(128, 3, activation='relu', padding='same')(x)
    skip2  = BatchNormalization()(x)

    x      = MaxPooling2D()(skip2)
    x      = Conv2D(256, 3, activation='relu', padding='same')(x)
    btn    = BatchNormalization()(x)

    x      = UpSampling2D()(btn)
    s2_att = attention_gate(skip2, btn, 64)
    x      = Concatenate()([x, s2_att])
    x      = Conv2D(128, 3, activation='relu', padding='same')(x)
    x      = BatchNormalization()(x)

    x      = UpSampling2D()(x)
    s1_att = attention_gate(skip1, x, 32)
    x      = Concatenate()([x, s1_att])
    x      = Conv2D(64, 3, activation='relu', padding='same')(x)
    x      = BatchNormalization()(x)

    output = Conv2D(1, 1, activation='sigmoid')(x)
    return Model(inputs, output)

# ============================================================
# MODEL 8: SE-U-Net
# U-Net with Squeeze-and-Excitation blocks.
# Selected to isolate the effect of channel attention
# without uncertainty guidance.
# ============================================================
def se_block(x, ratio=8):
    channels = x.shape[-1]
    gap      = GlobalAveragePooling2D()(x)
    gap      = Dense(channels // ratio, activation='relu')(gap)
    gap      = Dense(channels, activation='sigmoid')(gap)
    gap      = Reshape((1, 1, channels))(gap)
    return Multiply()([x, gap])

def se_unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    x      = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x      = BatchNormalization()(x)
    x      = Conv2D(64, 3, activation='relu', padding='same')(x)
    x      = se_block(x)
    skip1  = BatchNormalization()(x)

    x      = MaxPooling2D()(skip1)
    x      = Conv2D(128, 3, activation='relu', padding='same')(x)
    x      = BatchNormalization()(x)
    x      = Conv2D(128, 3, activation='relu', padding='same')(x)
    x      = se_block(x)
    skip2  = BatchNormalization()(x)

    x      = MaxPooling2D()(skip2)
    x      = Conv2D(256, 3, activation='relu', padding='same')(x)
    x      = se_block(x)
    x      = BatchNormalization()(x)

    x      = UpSampling2D()(x)
    x      = Concatenate()([x, skip2])
    x      = Conv2D(128, 3, activation='relu', padding='same')(x)
    x      = BatchNormalization()(x)

    x      = UpSampling2D()(x)
    x      = Concatenate()([x, skip1])
    x      = Conv2D(64, 3, activation='relu', padding='same')(x)
    x      = BatchNormalization()(x)

    output = Conv2D(1, 1, activation='sigmoid')(x)
    return Model(inputs, output)

# ============================================================
# MODEL 9: CBAM-U-Net
# U-Net with Convolutional Block Attention Module.
# Selected to isolate the effect of combined spatial and
# channel attention without uncertainty guidance.
# ============================================================
def cbam_block(x, ratio=8):
    channels    = x.shape[-1]
    gap         = GlobalAveragePooling2D()(x)
    gmp         = GlobalMaxPooling2D()(x)
    gap         = Dense(channels // ratio, activation='relu')(gap)
    gap         = Dense(channels)(gap)
    gmp         = Dense(channels // ratio, activation='relu')(gmp)
    gmp         = Dense(channels)(gmp)
    channel_att = Activation('sigmoid')(Add()([gap, gmp]))
    channel_att = Reshape((1, 1, channels))(channel_att)
    return Multiply()([x, channel_att])

def cbam_unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    x      = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x      = BatchNormalization()(x)
    x      = Conv2D(64, 3, activation='relu', padding='same')(x)
    x      = cbam_block(x)
    skip1  = BatchNormalization()(x)

    x      = MaxPooling2D()(skip1)
    x      = Conv2D(128, 3, activation='relu', padding='same')(x)
    x      = BatchNormalization()(x)
    x      = Conv2D(128, 3, activation='relu', padding='same')(x)
    x      = cbam_block(x)
    skip2  = BatchNormalization()(x)

    x      = MaxPooling2D()(skip2)
    x      = Conv2D(256, 3, activation='relu', padding='same')(x)
    x      = cbam_block(x)
    x      = BatchNormalization()(x)

    x      = UpSampling2D()(x)
    x      = Concatenate()([x, skip2])
    x      = Conv2D(128, 3, activation='relu', padding='same')(x)
    x      = BatchNormalization()(x)

    x      = UpSampling2D()(x)
    x      = Concatenate()([x, skip1])
    x      = Conv2D(64, 3, activation='relu', padding='same')(x)
    x      = BatchNormalization()(x)

    output = Conv2D(1, 1, activation='sigmoid')(x)
    return Model(inputs, output)

# ============================================================
# TRAIN AND EVALUATE ALL 9 BASELINES
# ============================================================
results = {}

print("\n" + "="*50)
print("Training Model 1: U-Net")
print("="*50)
results['U-Net'] = train_baseline(
    unet_model(), 'U-Net', X_train, Y_train, X_test, Y_test)

print("\n" + "="*50)
print("Training Model 2: VGG16-UNet")
print("="*50)
results['VGG16-UNet'] = train_baseline(
    vgg16_unet_model(), 'VGG16-UNet',
    X_train, Y_train, X_test, Y_test)

print("\n" + "="*50)
print("Training Model 3: ResNet50-UNet")
print("="*50)
results['ResNet50-UNet'] = train_baseline(
    resnet50_unet_model(), 'ResNet50-UNet',
    X_train, Y_train, X_test, Y_test)

print("\n" + "="*50)
print("Training Model 4: HRNet")
print("="*50)
results['HRNet'] = train_baseline(
    hrnet_model(), 'HRNet', X_train, Y_train, X_test, Y_test)

print("\n" + "="*50)
print("Training Model 5: DeepLabV3+")
print("="*50)
results['DeepLabV3+'] = train_baseline(
    deeplabv3plus_model(), 'DeepLabV3+',
    X_train, Y_train, X_test, Y_test)

print("\n" + "="*50)
print("Training Model 6: MC-Dropout U-Net")
print("="*50)
results['MC-Dropout U-Net'] = train_baseline(
    mc_dropout_unet(), 'MC-Dropout U-Net',
    X_train, Y_train, X_test, Y_test)

print("\n" + "="*50)
print("Training Model 7: Attention U-Net")
print("="*50)
results['Attention U-Net'] = train_baseline(
    attention_unet(), 'Attention U-Net',
    X_train, Y_train, X_test, Y_test)

print("\n" + "="*50)
print("Training Model 8: SE-U-Net")
print("="*50)
results['SE-U-Net'] = train_baseline(
    se_unet(), 'SE-U-Net', X_train, Y_train, X_test, Y_test)

print("\n" + "="*50)
print("Training Model 9: CBAM-U-Net")
print("="*50)
results['CBAM-U-Net'] = train_baseline(
    cbam_unet(), 'CBAM-U-Net', X_train, Y_train, X_test, Y_test)

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "="*75)
print("SUMMARY OF ALL BASELINE RESULTS")
print("="*75)
print(f"{'Model':<20} {'Acc':>8} {'Prec':>8} "
      f"{'Rec':>8} {'F1':>8} {'IoU':>8}")
print("-"*75)
for name, res in results.items():
    print(f"{name:<20} "
          f"{res['accuracy']*100:>7.2f}% "
          f"{res['precision']*100:>7.2f}% "
          f"{res['recall']*100:>7.2f}% "
          f"{res['f1']*100:>7.2f}% "
          f"{res['iou']*100:>7.2f}%")
print("="*75)
