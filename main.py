import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the forgery detection model (replace 'model_path' with your actual path)
model = tf.keras.models.load_model('forgery_discriminator_model.h5')

# Load and preprocess the image
image_path = "D:/IMAGE_FORGERY_DATASET/Dataset/Forged/8674.tif"
img = Image.open(image_path).resize((128, 128))  # Resize to match model input size
img_array = np.array(img) / 255.0  # Normalize pixel values

# Add batch dimension and predict
img_array = np.expand_dims(img_array, axis=0)
predictions = model.predict(img_array)
print(predictions)

# Thresholding (adjust threshold as needed)
threshold = 0.5
prediction_label = "Real" if predictions[0, 0] > threshold else "Fake"

print(f"Prediction: {prediction_label} (Probability: {predictions[0, 0]:.4f})")

# Display the image using matplotlib.pyplot
plt.imshow(img)
plt.title(f"Prediction: {prediction_label} (Probability: {predictions[0, 0]:.4f})")
plt.axis('off')
plt.show()
