import numpy as np
import keras
import os
import cv2


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, dataframe, dataset_path, augmentations, batch_size=8, dim=(256, 256), n_channels=3,
                 n_classes=4, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.filenames_list = dataframe['image_name'].values
        self.dataset_path = dataset_path
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augm = augmentations
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filenames_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        filenames_list = [self.filenames_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(filenames_list)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.filenames_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, filenames_list):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((len(filenames_list), self.dim[0], self.dim[1], self.n_channels))
        y = np.zeros((len(filenames_list), self.dim[0], self.dim[1]), dtype=int)

        # Generate data
        for i, filename in enumerate(filenames_list):
            # Store sample
            image = cv2.imread(os.path.join(self.dataset_path, 'images', filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = np.load(os.path.join(self.dataset_path, 'masks', "{0}.npy".format(filename.split('.')[0])))

            augmented = self.augm(image=image, mask=mask)
            X[i] = augmented['image'] / 255
            y[i] = augmented['mask']

        return X, np.expand_dims(y, axis=3)
        # return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
