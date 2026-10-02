import json
import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr

# Load your class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Load your downloaded TFLite model
interpreter = tf.lite.Interpreter(model_path="dermascope_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_skin_disease(image):
    if image is None:
        return "Please upload an image."

    # Preprocessing: Resize to 224x224 and normalize pixels (x / 255)
    image = image.resize((224, 224)).convert("RGB")
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run model inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    preds = output_data[0]

    # Format the results into a readable dictionary for the user
    results = {class_names[i]: float(preds[i]) * 100 for i in range(len(class_names))}
    return results

# Create the web app interface
demo = gr.Interface(
    fn=predict_skin_disease,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="DermaScope AI Model",
    description="Upload a skin lesion image to get automated disease predictions."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
