import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

# Load model
model = tf.keras.models.load_model("../saved_model/cats_vs_dogs_model.h5")


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)[0][0]
    label = "Dog" if prediction > 0.5 else "Cat"

    return label, prediction


if __name__ == "__main__":
    img_path = sys.argv[1]  # Get image path from command line
    label, confidence = predict_image(img_path)
    print(f"Prediction: {label} (Confidence: {confidence:.2f})")
