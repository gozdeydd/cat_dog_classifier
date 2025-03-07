from load_data import load_data
from model import create_model
import tensorflow as tf

# Load data
train_gen, val_gen = load_data("../dataset/")

# Create model
model = create_model()

# Train model
history = model.fit(train_gen, validation_data=val_gen, epochs=10)

# Save trained model
model.save("../saved_model/cats_vs_dogs_model.h5")
print("Model saved successfully!")
