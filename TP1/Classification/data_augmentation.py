import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

# Data Augmentation with Keras
def data_augmentation(X_train, y_train, num_augmented_images=60000, batch_size=32):
    # Reshape the data to have a single channel (grayscale)
    X_train = np.reshape(X_train, (*X_train.shape, 1))

    # Define the data augmentation transformations
    datagen = ImageDataGenerator(
        rotation_range=20,  # Random rotation by 20 degrees
        width_shift_range=0.1,  # Random horizontal shift
        height_shift_range=0.1,  # Random vertical shift
        shear_range=0.2,  # Random shear transformation
        zoom_range=0.2,  # Random zoom
        horizontal_flip=True  # Random horizontal flipping
    )

    # Fit the data augmentation generator on the training data
    datagen.fit(X_train)

    # Generate augmented data
    num_augmented_images = num_augmented_images  # Number of augmented images to generate
    batch_size = batch_size  # Batch size for generating augmented images
    augmented_images = np.zeros((num_augmented_images, *X_train.shape[1:]))
    augmented_labels = np.zeros(num_augmented_images)

    augmented_data_generator = datagen.flow(X_train, y_train, batch_size=batch_size)

    num_batches = num_augmented_images // batch_size
    for i in range(num_batches):
        batch_images, batch_labels = next(augmented_data_generator)
        augmented_images[i * batch_size: (i + 1) * batch_size] = batch_images
        augmented_labels[i * batch_size: (i + 1) * batch_size] = batch_labels

    # Concatenate the original training data with augmented data
    X_train_augmented = np.concatenate([X_train, augmented_images])
    y_train_augmented = np.concatenate([y_train, augmented_labels])

    # Shuffle the augmented dataset
    random_indices = np.random.permutation(X_train_augmented.shape[0])
    X_train_augmented = X_train_augmented[random_indices]
    y_train_augmented = y_train_augmented[random_indices].astype('int32')

    return X_train_augmented[:,:,:,0], y_train_augmented