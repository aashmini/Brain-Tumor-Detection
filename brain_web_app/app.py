from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ImageOps
from flask_login import LoginManager, login_required, current_user, UserMixin
from functools import wraps
import numpy as np
import os
import io
import time
import joblib
import pickle
import pandas as pd

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key')

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_PATH'] = 'models/brain_tumor_detection_model2.h5'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Flask-Login Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User Class & Loader (Mock)
class User(UserMixin):
    def __init__(self, id, role='patient'):
        self.id = id
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Admin-only decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Load the model
try:
    model = load_model(app.config['MODEL_PATH'])
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
text_model = joblib.load('brain_tumor_model_text.pkl')
# Helpers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def is_valid_mri(file_stream):
    try:
        img = Image.open(io.BytesIO(file_stream.read()))
        if img.mode not in ('L', 'RGB'):
            return False, "Image must be grayscale or RGB"
        if img.mode == 'RGB':
            gray_img = ImageOps.grayscale(img)
            if np.array(gray_img).var() < 100:
                return False, "Image doesn't appear to be a valid MRI scan"
        file_stream.seek(0)
        return True, "Valid MRI"
    except Exception as e:
        return False, f"Invalid image: {str(e)}"

# --- HOME PAGE + UPLOAD ---
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if not model:
            flash('System error: Model not loaded', 'danger')
            return redirect(request.url)

        file = request.files.get('file')
        if not file or file.filename == '':
            flash('No file selected', 'warning')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Only JPG, JPEG, PNG files allowed', 'warning')
            return redirect(request.url)

        is_valid, msg = is_valid_mri(file.stream)
        if not is_valid:
            flash(msg, 'danger')
            return redirect(request.url)

        try:
            filename = f"{int(time.time())}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = load_img(filepath, target_size=(224, 224), color_mode='rgb')
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0][0]
            confidence = prediction if prediction > 0.5 else 1 - prediction

            if confidence < 0.75:
                result = "Inconclusive - Poor Quality MRI"
                status = "warning"
            elif prediction > 0.5:
                result = "Tumor Detected"
                status = "danger"
            else:
                result = "No Tumor Detected"
                status = "success"

            return render_template('index.html',
                                   result=result,
                                   status=status,
                                   confidence=round(confidence * 100, 2),
                                   image_url=url_for('uploaded_file', filename=filename))

        except Exception as e:
            flash(f'Processing error: {str(e)}', 'danger')
            return redirect(request.url)

    return render_template('index.html')
@app.route('/home')
def home():
    return render_template('home.html')  
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
# Text Detection Route (using pkl model)
@app.route('/text_detect', methods=['GET', 'POST'])
def text_detect():
    try:
        age = int(request.form['age'])
        gender = request.form['gender']
        family_history = request.form['family_history']
        symptoms = request.form['symptoms']
        medical_history = request.form['medical_history']  # You are collecting it but not using it
        neurological_exam = "Generic exam result"  # Optional fixed input or form field

        # Combine text for model input
        combined_text = symptoms + " " + neurological_exam

        input_df = pd.DataFrame([{
            'age': age,
            'gender': gender,
            'family_history': family_history,
            'combined_text': combined_text
        }])

        prediction = text_model.predict(input_df)[0]
        if prediction == 1:
            result = "⚠️ Tumor Detected"
            status = "positive"
        else:
            result = "✅ No Tumor Detected"
            status = "negative"
        return render_template('text_detect.html', prediction=result, status=status)


    
    except Exception as e:
        return render_template('text_detect.html', prediction=f"Error during prediction: {str(e)}")



# --- GENERAL ---
@app.route('/faq')
def faq():
    return render_template('support/faq.html')

@app.route('/contact')
def contact():
    return render_template('support/contact.html')

@app.route('/settings')
def settings():
    return render_template('support/settings.html')
@app.route('/about')
def about():
    return render_template('support/about.html')
@app.route("/treatment")
def treatment():
    return render_template("treatment.html")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
