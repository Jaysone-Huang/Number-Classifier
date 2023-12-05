import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
# load up our model
model = tf.keras.models.load_model('Number_Model.keras')

# Function to make predictions using the MNIST model
def classify_image(img):
    if img is not None:
        img = img.reshape((1, 28, 28, 1)).astype('float32') / 255
        prediction = model.predict(img)
        predicted_number = np.argmax(prediction)
        prob = round(100 * np.max(prediction), 2)
        return f"Predicted Number: {predicted_number}, Probability: {prob}%"
    return ''
# Create Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(shape=(28,28), image_mode='L', invert_colors=True, source='canvas'),
    outputs=gr.Label(num_top_classes=4),
    live=True,
)

# Launch the interface
iface.launch()