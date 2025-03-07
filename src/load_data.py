import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, img_size=(150, 150), batch_size=32, validation_split=0.2):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)

    train_generator = datagen.flow_from_directory(
        data_dir, target_size=img_size, batch_size=batch_size,
        class_mode='binary', subset='training')

    val_generator = datagen.flow_from_directory(
        data_dir, target_size=img_size, batch_size=batch_size,
        class_mode='binary', subset='validation')

    return train_generator, val_generator
