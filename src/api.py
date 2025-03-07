from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image

# Load model
model = tf.keras.models.load_model("../saved_model/cats_vs_dogs_model.h5")

app = FastAPI()


def preprocess_image(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read()))
    img_array = preprocess_image(img)

    prediction = model.predict(img_array)[0][0]
    label = "Dog" if prediction > 0.5 else "Cat"

    return {"prediction": label, "confidence": float(prediction)}

