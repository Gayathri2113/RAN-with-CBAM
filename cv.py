import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# CBAM (Convolutional Block Attention Module)
def cbam_block(input_tensor, reduction_ratio=16):
    # Channel Attention
    channel_axis = -1
    channels = input_tensor.shape[channel_axis]

    # Average Pooling
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    avg_pool = tf.keras.layers.Reshape((1, 1, channels))(avg_pool)

    # Max Pooling
    max_pool = tf.keras.layers.GlobalMaxPooling2D()(input_tensor)
    max_pool = tf.keras.layers.Reshape((1, 1, channels))(max_pool)

    # Shared MLP
    shared_mlp = tf.keras.Sequential([
        layers.Dense(channels // reduction_ratio, activation='relu', kernel_initializer='he_normal'),
        layers.Dense(channels, activation='sigmoid', kernel_initializer='he_normal')
    ])

    # Channel Attention
    avg_channel_att = shared_mlp(avg_pool)
    max_channel_att = shared_mlp(max_pool)
    channel_attention = layers.Add()([avg_channel_att, max_channel_att])
    channel_attention = layers.Activation('sigmoid')(channel_attention)

    # Apply Channel Attention
    scaled_input = layers.Multiply()([input_tensor, channel_attention])

    # Spatial Attention
    avg_feat = tf.reduce_mean(scaled_input, axis=-1, keepdims=True)
    max_feat = tf.reduce_max(scaled_input, axis=-1, keepdims=True)
    spatial_feat = tf.concat([avg_feat, max_feat], axis=-1)

    spatial_attention = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid', kernel_initializer='he_normal')(spatial_feat)

    # Apply Spatial Attention
    output = layers.Multiply()([scaled_input, spatial_attention])

    return output

# Residual CBAM Block
def residual_cbam_block(input_tensor, filters, stride=1):
    # Residual Path
    if stride != 1 or input_tensor.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='valid')(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor

    # Convolution Path
    x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # CBAM Module
    x = cbam_block(x)

    # Combine Residual and Convolution Paths
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)

    return x

# Residual Attention Network
def create_residual_attention_network(input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # Initial Convolution
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # Layer 1
    x = residual_cbam_block(x, 64)
    for _ in range(2):
        x = residual_cbam_block(x, 64)

    # Layer 2
    x = residual_cbam_block(x, 128, stride=2)
    for _ in range(3):
        x = residual_cbam_block(x, 128)

    # Layer 3
    x = residual_cbam_block(x, 256, stride=2)
    for _ in range(5):
        x = residual_cbam_block(x, 256)

    # Layer 4
    x = residual_cbam_block(x, 512, stride=2)
    for _ in range(2):
        x = residual_cbam_block(x, 512)

    # Global Average Pooling and Classifier
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=x)
    return model

# Data Preparation and Training Function
def train_cifar10(num_samples=10000, epochs=50, batch_size=64, learning_rate=1e-3):
    # Load and Preprocess CIFAR-10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Randomly select 10,000 samples
    np.random.seed(42)
    train_indices = np.random.choice(len(x_train), num_samples, replace=False)
    x_train_subset = x_train[train_indices]
    y_train_subset = y_train[train_indices]

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(x_train_subset)

    # Create Model
    model = create_residual_attention_network()

    # Compile Model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Learning Rate Scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-5
    )

    # Model Checkpoint
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    )

    # Training
    history = model.fit(
        datagen.flow(x_train_subset, y_train_subset, batch_size=batch_size),
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[lr_scheduler, checkpoint]
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc * 100:.2f}%')

    return model, history

# Run the training
if _name_ == '_main_':
    trained_model, training_history = train_cifar10()