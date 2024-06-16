import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet')

# Example image (dummy data)
img_path = 'path_to_your_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Predict class probabilities
preds = base_model.predict(x)
# Decode predictions
print('Predicted:', decode_predictions(preds, top=3)[0])
