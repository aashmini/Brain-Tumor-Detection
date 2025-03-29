import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tkinter import Tk, Button, Label, filedialog, messagebox, Frame
from PIL import Image, ImageTk

# Define paths
model_path = os.path.join('models', 'brain_tumor_detection_model2.h5')

# Load the saved model
model = load_model(model_path)

# Function to load and preprocess images
def load_image(img_path, target_size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize pixel values
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Function to handle image selection and prediction
def predict_image():
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    if not file_path:
        return

    # Clear previous image and result
    image_label.config(image=None)
    result_label.config(text="Prediction: ")

    # Load and preprocess the selected image
    try:
        image = load_image(file_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image: {e}")
        return

    # Run the model
    try:
        prediction = model.predict(image)
        predicted_label = "Tumor" if prediction > 0.5 else "No Tumor"
    except Exception as e:
        messagebox.showerror("Error", f"Failed to make prediction: {e}")
        return

    # Display the image and prediction result
    img = Image.open(file_path)
    img = img.resize((300, 300), Image.Resampling.LANCZOS)  # Resize for better display
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img  # Keep a reference to avoid garbage collection
    result_label.config(text=f"Prediction: {predicted_label}", fg="green" if predicted_label == "No Tumor" else "red")

# Function to clear the current image and result
def clear_image():
    image_label.config(image=None)
    result_label.config(text="Prediction: ", fg="black")

# Create the GUI
root = Tk()
root.title("Brain Tumor Detection")
root.geometry("500x450")  # Adjusted window size
root.configure(bg="#f0f0f0")  # Light gray background

# Custom fonts
button_font = ("Arial", 12, "bold")
label_font = ("Arial", 14, "bold")
result_font = ("Arial", 16, "bold")

# Create a frame for buttons
button_frame = Frame(root, bg="#f0f0f0")
button_frame.pack(pady=10)

# Add a button to select an image
select_button = Button(
    button_frame,
    text="Select Image",
    command=predict_image,
    font=button_font,
    bg="#4CAF50",  # Green background
    fg="white",    # White text
    padx=10,
    pady=5,
    borderwidth=0,
    relief="flat"
)
select_button.pack(side="left", padx=10)

# Add a button to clear the current image and result
clear_button = Button(
    button_frame,
    text="Clear",
    command=clear_image,
    font=button_font,
    bg="#f44336",  # Red background
    fg="white",    # White text
    padx=10,
    pady=5,
    borderwidth=0,
    relief="flat"
)
clear_button.pack(side="left", padx=10)

# Add a label to display the selected image
image_label = Label(root, bg="#f0f0f0")
image_label.pack(pady=10)

# Add a label to display the prediction result
result_label = Label(
    root,
    text="Prediction: ",
    font=result_font,
    bg="#f0f0f0",
    fg="black"  # Default text color
)
result_label.pack(pady=10)

# Run the GUI
root.mainloop()