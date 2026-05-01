import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import io
import base64
from werkzeug.security import generate_password_hash, check_password_hash
from database import init_db, get_db_connection
import io
import nltk
from nltk.stem.lancaster import LancasterStemmer
import json
import random
import numpy as np
import tensorflow as tf
from functools import wraps
# Initialize stemmer
stemmer = LancasterStemmer()

# Global variables for chatbot
chat_model = None
chat_words = None
chat_labels = None
chat_data = None

def init_chatbot():
    """Initialize chatbot model and components"""
    global chat_model, chat_words, chat_labels, chat_data
    
    try:
        print("Initializing chatbot...")
        
        # Load intents
        with open("sample.json", encoding="utf-8") as f:
            chat_data = json.load(f)
        print("✓ Loaded intents data")
        
        # Load processed data
        with open("assets/input_data.pickle", "rb") as f:
            chat_words, chat_labels, chat_training, chat_output = pickle.load(f)
        print("✓ Loaded processed data")
        
        # Load model
        chat_model = tf.keras.models.load_model("assets/chatbot_model.keras")
        print("✓ Loaded chatbot model")
        
        print("Chatbot initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        # Create a simple fallback dataset
        create_fallback_chatbot()
        return False

def create_fallback_chatbot():
    """Create a simple fallback chatbot if ML model fails"""
    global chat_data, chat_words, chat_labels
    
    print("Creating fallback chatbot...")
    
    # Simple intents for fallback
    chat_data = {
        "intents": [
            {
                "tag": "greeting",
                "patterns": ["hi", "hello", "hey", "good morning", "good afternoon"],
                "responses": ["Hello! I'm your health assistant. How can I help with heart disease prediction today?"]
            },
            {
                "tag": "prediction",
                "patterns": ["predict", "prediction", "how does it work", "model"],
                "responses": ["I analyze health parameters like age, BP, cholesterol to predict heart disease risk using AI."]
            },
            {
                "tag": "features", 
                "patterns": ["what data", "parameters", "features", "what do I need"],
                "responses": ["Need: age, gender, BP, cholesterol, smoking status, diabetes, BMI, and heart rate."]
            },
            {
                "tag": "health",
                "patterns": ["health tips", "healthy heart", "prevent", "exercise", "diet"],
                "responses": ["Exercise regularly, eat balanced diet, avoid smoking, manage stress, monitor BP & cholesterol."]
            },
            {
                "tag": "thanks",
                "patterns": ["thank you", "thanks", "appreciate"],
                "responses": ["You're welcome! Stay heart healthy! ❤️"]
            },
            {
                "tag": "fallback",
                "patterns": [""],
                "responses": ["I can help with heart disease predictions, health tips, and explain how our AI system works!"]
            }
        ]
    }
    
    # Create simple word list
    chat_words = ['health', 'heart', 'predict', 'help', 'hello', 'hi', 'thanks']
    chat_labels = ['greeting', 'prediction', 'features', 'health', 'thanks', 'fallback']
    
    print("✓ Fallback chatbot created")

def simple_bag_of_words(s, words):
    """Simple bag of words implementation"""
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s.lower())
    
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
                
    return np.array(bag)

def get_simple_response(inp):
    """Simple rule-based response matching"""
    inp_lower = inp.lower()
    
    # Check each intent
    for intent in chat_data["intents"]:
        for pattern in intent["patterns"]:
            if pattern in inp_lower:
                return random.choice(intent["responses"])
    
    # Fallback response
    return random.choice([tg["responses"] for tg in chat_data["intents"] if tg["tag"] == "fallback"][0])

def get_chat_response(inp):
    """Get chatbot response - tries ML model first, falls back to simple matching"""
    if not inp or len(inp.strip()) < 2:
        return "Please type a longer message (at least 2 characters)."
    
    # Clean input
    inp = inp.strip()
    
    try:
        # Try ML model first if available
        if chat_model is not None and chat_words is not None:
            bow = simple_bag_of_words(inp, chat_words)
            results = chat_model.predict(np.array([bow]), verbose=0)[0]
            results_index = np.argmax(results)
            tag = chat_labels[results_index]

            if results[results_index] > 0.5:  # Lower confidence threshold for fallback
                for tg in chat_data["intents"]:
                    if tg["tag"] == tag:
                        return random.choice(tg["responses"])
        
        # Fall back to simple matching
        return get_simple_response(inp)
        
    except Exception as e:
        print(f"Chatbot error: {e}")
        # Final fallback
        return get_simple_response(inp)

# Initialize chatbot when app starts
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Load the trained model
with open('pcos_best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Selected features from the model
SELECTED_FEATURES = [
    'Age (yrs)', 'Weight (Kg)', 'BMI', 'Cycle(R/I)', 'Hip(inch)', 'Waist(inch)',
    'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)',
    'Pimples(Y/N)', 'Fast food (Y/N)', 'Follicle No. (L)', 'Follicle No. (R)',
    'Avg. F size (L) (mm)'
]

# Initialize database
init_db()

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        existing_user = conn.execute(
            'SELECT id FROM users WHERE username = ? OR email = ?', 
            (username, email)
        ).fetchone()
        
        if existing_user:
            flash('Username or email already exists!', 'error')
            return render_template('register.html')
        
        hashed_password = generate_password_hash(password)
        conn.execute(
            'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
            (username, email, hashed_password)
        )
        conn.commit()
        conn.close()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute(
            'SELECT * FROM users WHERE username = ?', (username,)
        ).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials!', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    predictions = conn.execute(
        'SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 5',
        (session['user_id'],)
    ).fetchall()
    conn.close()
    
    return render_template('dashboard.html', predictions=predictions)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            # Get form data
            input_data = {
                'Age (yrs)': float(request.form['age']),
                'Weight (Kg)': float(request.form['weight']),
                'BMI': float(request.form['bmi']),
                'Cycle(R/I)': 1 if request.form['cycle_pattern'] == 'Regular' else 0,
                'Hip(inch)': float(request.form['hip']),
                'Waist(inch)': float(request.form['waist']),
                'Weight gain(Y/N)': 1 if request.form.get('weight_gain') else 0,
                'hair growth(Y/N)': 1 if request.form.get('hair_growth') else 0,
                'Skin darkening (Y/N)': 1 if request.form.get('skin_darkening') else 0,
                'Hair loss(Y/N)': 1 if request.form.get('hair_loss') else 0,
                'Pimples(Y/N)': 1 if request.form.get('pimples') else 0,
                'Fast food (Y/N)': 1 if request.form.get('fast_food') else 0,
                'Follicle No. (L)': float(request.form['follicle_left']),
                'Follicle No. (R)': float(request.form['follicle_right']),
                'Avg. F size (L) (mm)': float(request.form['follicle_avg_size'])
            }
            
            # Convert to numpy array for prediction
            features_array = np.array([list(input_data.values())])
            
            # Make prediction
            prediction = model.predict(features_array)[0]
            probability = model.predict_proba(features_array)[0]
            confidence = probability[prediction]
            
            # Generate SHAP explanation
            shap_values = generate_shap_plot(features_array, input_data.keys())
            
            # Generate recommendations
            recommendations = generate_recommendations(
                prediction, 
                input_data['Age (yrs)'], 
                input_data['Weight (Kg)'],
                input_data['BMI']
            )
            
            # Save prediction to database and get the ID
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions 
                (user_id, age, weight, bmi, cycle_pattern, hip, waist, weight_gain, 
                 hair_growth, skin_darkening, hair_loss, pimples, fast_food, 
                 follicle_left, follicle_right, follicle_avg_size, prediction_result, 
                 prediction_confidence, shap_values)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session['user_id'], input_data['Age (yrs)'], input_data['Weight (Kg)'],
                input_data['BMI'], request.form['cycle_pattern'], input_data['Hip(inch)'],
                input_data['Waist(inch)'], input_data['Weight gain(Y/N)'],
                input_data['hair growth(Y/N)'], input_data['Skin darkening (Y/N)'],
                input_data['Hair loss(Y/N)'], input_data['Pimples(Y/N)'],
                input_data['Fast food (Y/N)'], input_data['Follicle No. (L)'],
                input_data['Follicle No. (R)'], input_data['Avg. F size (L) (mm)'],
                int(prediction), float(confidence), shap_values
            ))
            
            # Get the prediction ID before committing
            prediction_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return render_template('prediction_result.html',
                                prediction=prediction,
                                confidence=confidence,
                                recommendations=recommendations,
                                shap_plot=generate_shap_plot_base64(features_array),
                                prediction_id=prediction_id)
            
        except Exception as e:
            flash(f'Error in prediction: {str(e)}', 'error')
            return redirect(url_for('predict'))
    
    return render_template('predict.html')

def generate_shap_plot(features_array, feature_names):
    """Generate SHAP values for the prediction"""
    try:
        # Create a SHAP explainer
        explainer = shap.TreeExplainer(model.named_estimators_['rf'])
        shap_values = explainer.shap_values(features_array)
        
        # Convert to list for storage
        return str(shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values)
    except:
        return None

def generate_shap_plot_base64(features_array):
    """Generate SHAP plot as base64 image"""
    try:
        explainer = shap.TreeExplainer(model.named_estimators_['rf'])
        shap_values = explainer.shap_values(features_array)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, features_array, feature_names=SELECTED_FEATURES, show=False)
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=300)
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    except Exception as e:
        print(f"SHAP plot error: {e}")
        return None

def generate_recommendations(prediction, age, weight, bmi):
    """Generate personalized recommendations based on prediction, age, weight, and BMI"""
    recommendations = {}
    
    # Calculate ideal weight range based on height (using BMI formula)
    height_m = (weight / bmi) ** 0.5  # Estimate height from weight and BMI
    ideal_weight_min = 18.5 * (height_m ** 2)
    ideal_weight_max = 24.9 * (height_m ** 2)
    weight_to_lose = weight - ideal_weight_max if weight > ideal_weight_max else 0
    weight_to_gain = ideal_weight_min - weight if weight < ideal_weight_min else 0
    
    # Age categories
    age_group = "teen" if age < 20 else "young_adult" if age < 35 else "adult"
    
    if prediction == 1:  # PCOS detected
        recommendations['diagnosis'] = 'PCOS Detected'
        recommendations['overview'] = f'Based on the analysis, PCOS (Polycystic Ovary Syndrome) has been detected. As a {age}-year-old with BMI {bmi:.1f}, here is your personalized management plan:'
        
        # Enhanced Diet recommendations based on BMI, age, and weight
        if bmi >= 30:  # Obese
            calorie_target = 1200 if age < 25 else 1400
            recommendations['diet'] = [
                f'Weight loss target: {weight_to_lose:.1f} kg ({weight_to_lose*2.2:.1f} lbs) to reach healthy BMI',
                f'Calorie restriction: {calorie_target}-1600 calories/day based on your age',
                'Low glycemic index diet: whole grains, legumes, non-starchy vegetables',
                'High protein intake (30-35% of calories): lean meats, fish, eggs, legumes',
                'Healthy fats (30% of calories): avocados, nuts, olive oil, fatty fish',
                'Strictly avoid: sugary drinks, processed foods, refined carbohydrates',
                'Intermittent fasting (16:8 method) under medical supervision',
                'Focus on fiber-rich foods: 25-30g daily for satiety',
                'Meal timing: 3 main meals, no late-night eating'
            ]
        elif bmi >= 25:  # Overweight
            calorie_target = 1400 if age < 25 else 1600
            recommendations['diet'] = [
                f'Weight loss target: {weight_to_lose:.1f} kg to reach optimal BMI',
                f'Moderate calorie restriction: {calorie_target}-1800 calories/day',
                'Balanced macronutrients: 40% carbs, 30% protein, 30% fats',
                'Emphasize anti-inflammatory foods: berries, leafy greens, turmeric',
                'Include insulin-sensitizing foods: cinnamon, apple cider vinegar',
                'Limit dairy consumption, especially high-fat dairy',
                'Regular meal timing every 4-5 hours',
                'Hydration: 2-3 liters water daily, herbal teas'
            ]
        else:  # Normal or underweight
            if bmi < 18.5:
                recommendations['diet'] = [
                    f'Weight gain target: {weight_to_gain:.1f} kg to reach healthy range',
                    'Calorie surplus: 200-300 extra calories daily',
                    'Focus on nutrient-dense foods rather than empty calories',
                    'Healthy weight gain strategies: nuts, seeds, avocados, whole grains',
                    'Frequent small meals: 5-6 meals per day',
                    'Protein-rich snacks between meals',
                    'Monitor for eating disorders or malabsorption issues'
                ]
            else:
                recommendations['diet'] = [
                    'Weight maintenance with focus on body composition',
                    'Balanced diet emphasizing complex carbohydrates',
                    'Regular meal timing to maintain insulin sensitivity',
                    'Adequate protein (1.2-1.6g per kg body weight)',
                    'Limit dairy intake if sensitive, consider alternatives',
                    'Anti-inflammatory Mediterranean-style diet',
                    'Include fermented foods for gut health'
                ]
        
        # Age-specific medication and treatment
        if age < 20:
            recommendations['medication'] = [
                'Lifestyle modification as first-line treatment',
                'Metformin: 500-1000mg daily if insulin resistant',
                'Nutritional counseling focused on healthy eating habits',
                'Calcium and Vitamin D for bone health',
                'Psychological support for body image concerns',
                'Regular monitoring of growth and development'
            ]
        elif age >= 20 and age < 35:
            recommendations['medication'] = [
                'Metformin: 1000-2000mg daily to improve insulin sensitivity',
                'Oral contraceptives for cycle regulation (if not trying to conceive)',
                'Spironolactone 50-100mg daily for androgen-related symptoms',
                'Myo-inositol supplements: 2000-4000mg daily',
                'Vitamin D: 2000-4000 IU daily based on levels',
                'Omega-3 supplements: 1000-2000mg EPA/DHA daily',
                'Consider berberine as alternative to metformin'
            ]
        else:  # 35 and above
            recommendations['medication'] = [
                'Metformin: 1500-2000mg daily for metabolic health',
                'Regular cardiovascular risk assessment',
                'Bone density monitoring',
                'Lipid profile management',
                'Blood pressure monitoring',
                'Comprehensive metabolic panel annually',
                'Individualized hormone therapy if needed'
            ]
        
        # Enhanced Yoga and exercise based on weight and age
        if bmi >= 30:
            recommendations['yoga'] = [
                'Start with gentle yoga: 20-30 minutes daily',
                'Chair yoga for joint protection if needed',
                'Focus on breathing exercises (Pranayama)',
                'Gradual progression to standing poses',
                'Surya Namaskar (Sun Salutation): Start with 5 rounds, build to 12',
                'Baddha Konasana (Butterfly Pose) - 3-5 minutes daily',
                'Supported Sarvangasana (Shoulder Stand) with props',
                'Walking: 30 minutes daily, gradually increase intensity'
            ]
        elif bmi >= 25:
            recommendations['yoga'] = [
                'Surya Namaskar (Sun Salutation) - 12 rounds daily',
                'Baddha Konasana (Butterfly Pose) - improves pelvic circulation',
                'Bhujangasana (Cobra Pose) - stimulates reproductive organs',
                'Paschimottanasana (Seated Forward Bend) - calms nervous system',
                'Dhanurasana (Bow Pose) - strengthens back and core',
                'Pranayama: Kapalbhati, Anulom Vilom - 10 minutes daily',
                'Cardio: 30 minutes, 5 days/week + strength training 2 days/week'
            ]
        else:
            recommendations['yoga'] = [
                'Dynamic yoga flow: 45-60 minutes daily',
                'Power yoga for strength building',
                'Inversion poses: Sirsasana, Sarvangasana',
                'Twisting poses for detoxification',
                'Advanced pranayama techniques',
                'High-intensity interval training 3 days/week',
                'Strength training: 3 days/week focusing on major muscle groups'
            ]
        
        # Enhanced Lifestyle modifications
        lifestyle_base = [
            f'Exercise: {150 if bmi < 30 else 200} minutes moderate intensity weekly',
            'Stress management: meditation, yoga nidra, mindfulness',
            'Sleep: 7-9 hours nightly, consistent schedule',
            'Limit alcohol: maximum 1-2 drinks weekly',
            'Caffeine: limit to 200mg daily (2 cups coffee)',
            'Smoking cessation if applicable',
            'Regular monitoring with endocrinologist/gynecologist'
        ]
        
        if age < 25:
            lifestyle_base.extend([
                'Focus on establishing healthy lifelong habits',
                'Academic/work-life balance management',
                'Social support system development',
                'Regular health education about PCOS'
            ])
        elif age >= 25 and age < 35:
            lifestyle_base.extend([
                'Fertility awareness and planning if desired',
                'Career stress management',
                'Regular pelvic ultrasounds and hormone testing',
                'Preconception counseling if planning pregnancy'
            ])
        else:
            lifestyle_base.extend([
                'Menopausal transition preparation',
                'Cardiovascular health focus',
                'Bone health maintenance',
                'Regular cancer screenings'
            ])
        
        recommendations['lifestyle'] = lifestyle_base
        
        # Additional specialized sections
        recommendations['monitoring'] = [
            f'Weight tracking: Weekly, target BMI 18.5-24.9 (currently {bmi:.1f})',
            'Menstrual cycle tracking: Use app or calendar',
            'Blood tests: HbA1c, fasting insulin, lipids every 6 months',
            'Pelvic ultrasound: Annually',
            'Blood pressure: Monthly at home',
            'Symptom diary: Track improvements/changes'
        ]
        
        recommendations['special_considerations'] = [
            f'Given your age ({age}) and BMI ({bmi:.1f}), focus on { "weight loss and metabolic health" if bmi >= 25 else "hormone balance and maintenance"}',
            'Consider working with registered dietitian specializing in PCOS',
            'Join PCOS support group for emotional well-being',
            'Be patient - improvements take 3-6 months of consistent effort'
        ]
        
    else:  # Normal - no PCOS detected
        recommendations['diagnosis'] = 'Normal (No PCOS Detected)'
        recommendations['overview'] = f'Based on the analysis, no signs of PCOS were detected. As a {age}-year-old with BMI {bmi:.1f}, maintain your health with these personalized recommendations:'
        
        # Enhanced preventive recommendations
        if bmi >= 25:
            recommendations['diet'] = [
                f'Weight management: Target {weight_to_lose:.1f} kg loss for optimal health',
                'Preventive nutrition: Focus on maintaining insulin sensitivity',
                'Balanced Mediterranean-style diet',
                'Portion control and mindful eating',
                'Limit processed foods and added sugars',
                'Regular meal timing to prevent metabolic issues',
                'Adequate fiber: 25-30g daily from whole foods'
            ]
        elif bmi < 18.5:
            recommendations['diet'] = [
                f'Healthy weight gain: Target {weight_to_gain:.1f} kg to reach normal range',
                'Nutrient-dense calorie sources',
                'Regular balanced meals and snacks',
                'Strength training to build muscle mass',
                'Monitor for underlying health conditions'
            ]
        else:
            recommendations['diet'] = [
                'Maintenance of healthy weight and body composition',
                'Varied, colorful plant-based diet',
                'Lean proteins and healthy fats',
                'Regular hydration with water',
                'Occasional treats in moderation'
            ]
        
        # Age-specific preventive care
        if age < 25:
            recommendations['medication'] = [
                'No specific medication required',
                'Calcium + Vitamin D for bone health (1000-1200mg/600-800IU)',
                'Iron supplementation if heavy periods',
                'Regular multivitamin for nutritional insurance'
            ]
        elif age >= 25 and age < 40:
            recommendations['medication'] = [
                'No routine medication needed',
                'Consider Vitamin D: 1000-2000 IU daily',
                'Omega-3 for inflammation prevention',
                'Prenatal vitamins if planning pregnancy'
            ]
        else:
            recommendations['medication'] = [
                'Regular health screenings based on age',
                'Vitamin D: 2000-4000 IU daily',
                'Calcium for bone health',
                'Consider low-dose aspirin if cardiovascular risk factors'
            ]
        
        # Physical activity based on age and weight
        if bmi >= 25:
            recommendations['yoga'] = [
                'General fitness yoga: 30 minutes daily',
                'Focus on cardiovascular health',
                'Strength-building poses',
                'Regular walking or swimming',
                'Gradual intensity progression'
            ]
        else:
            recommendations['yoga'] = [
                'Maintenance yoga practice: 20-30 minutes daily',
                'Variety of styles for overall fitness',
                'Stress reduction techniques',
                'Regular physical activity enjoyment'
            ]
        
        # Enhanced Lifestyle for prevention
        recommendations['lifestyle'] = [
            f'Exercise: {150 if bmi < 25 else 200} minutes weekly moderate activity',
            'Stress management appropriate for life stage',
            f'Sleep: 7-{9 if age < 65 else 8} hours quality sleep',
            'Alcohol moderation: 1 drink or less daily',
            'No smoking or tobacco use',
            'Regular health check-ups and screenings',
            'Sun protection and skin care'
        ]
        
        # PCOS prevention specific advice
        recommendations['prevention'] = [
            'Maintain healthy weight throughout life',
            'Regular exercise to prevent insulin resistance',
            'Balanced diet low in processed foods',
            'Regular menstrual cycle monitoring',
            'Awareness of PCOS symptoms for early detection',
            'Family history awareness of metabolic disorders'
        ]
        
        recommendations['monitoring'] = [
            f'Annual health check-up including BMI (current: {bmi:.1f})',
            'Regular menstrual cycle tracking',
            'Blood pressure monitoring',
            'Dental and vision check-ups',
            'Age-appropriate cancer screenings'
        ]
    
    return recommendations

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    predictions = conn.execute(
        'SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC',
        (session['user_id'],)
    ).fetchall()
    conn.close()
    
    return render_template('history.html', predictions=predictions)
# Chatbot route
@app.route('/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return jsonify({'response': '🔒 Please login first to chat with the health assistant.'})
    
    user_message = request.json.get('message', '').strip()
    
    if not user_message:
        return jsonify({'response': 'Please enter a message.'})
    
    if len(user_message) < 2:
        return jsonify({'response': 'Please type a longer message (at least 2 characters).'})
    
    # Get response from chatbot
    response = get_chat_response(user_message)
    return jsonify({'response': response})
# Admin login required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_id' not in session:
            flash('Admin access required!', 'error')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

# Admin login
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        admin = conn.execute(
            'SELECT * FROM admin_users WHERE username = ?', (username,)
        ).fetchone()
        conn.close()
        
        if admin and check_password_hash(admin['password'], password):
            session['admin_id'] = admin['id']
            session['admin_username'] = admin['username']
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials!', 'error')
    
    return render_template('admin_login.html')

# Admin logout
@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_id', None)
    session.pop('admin_username', None)
    flash('Admin logged out successfully!', 'success')
    return redirect(url_for('admin_login'))

def generate_daily_chart_data(daily_predictions):
    """Generate daily prediction trends data for charts"""
    dates = []
    pcos_counts = []
    normal_counts = []
    total_counts = []
    
    for day in daily_predictions:
        dates.append(day['date'])
        pcos_counts.append(day['pcos_count'])
        normal_counts.append(day['normal_count'])
        total_counts.append(day['total_predictions'])
    
    return {
        'dates': dates,
        'pcos_counts': pcos_counts,
        'normal_counts': normal_counts,
        'total_counts': total_counts
    }

def generate_weekly_chart_data(weekly_trends):
    """Generate weekly prediction trends data for charts"""
    weeks = []
    pcos_counts = []
    normal_counts = []
    
    for week in weekly_trends:
        # Format week number for display
        week_num = week['week'].split('-')[1]
        weeks.append(f"Week {week_num}")
        pcos_counts.append(week['pcos_count'])
        normal_counts.append(week['normal_count'])
    
    return {
        'weeks': weeks,
        'pcos_counts': pcos_counts,
        'normal_counts': normal_counts
    }

def generate_user_chart_data(user_registrations):
    """Generate user registration trends data for charts"""
    dates = []
    new_users = []
    
    for reg in user_registrations:
        dates.append(reg['date'])
        new_users.append(reg['new_users'])
    
    return {
        'dates': dates,
        'new_users': new_users
    }
def generate_timeline_chart_data(user_predictions_timeline):
    """Generate individual user prediction timeline data for line graph"""
    # Group predictions by user
    user_data = {}
    
    for prediction in user_predictions_timeline:
        username = prediction['username']
        if username not in user_data:
            user_data[username] = {
                'timestamps': [],
                'results': [],
                'confidences': []
            }
        
        # Format timestamp for display
        timestamp = prediction['created_at'][:16]  # YYYY-MM-DD HH:MM
        user_data[username]['timestamps'].append(timestamp)
        user_data[username]['results'].append(prediction['prediction_result'])
        user_data[username]['confidences'].append(float(prediction['prediction_confidence']) * 100)
    
    return user_data
# Admin Dashboard

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    conn = get_db_connection()
    
    # Get statistics
    total_users = conn.execute('SELECT COUNT(*) as count FROM users').fetchone()['count']
    total_predictions = conn.execute('SELECT COUNT(*) as count FROM predictions').fetchone()['count']
    total_feedback = conn.execute('SELECT COUNT(*) as count FROM feedback').fetchone()['count']
    avg_rating = conn.execute('SELECT AVG(rating) as avg FROM feedback WHERE rating IS NOT NULL').fetchone()['avg'] or 0
    
    # Recent predictions with user info
    recent_predictions = conn.execute('''
        SELECT p.*, u.username 
        FROM predictions p 
        JOIN users u ON p.user_id = u.id 
        ORDER BY p.created_at DESC 
        LIMIT 10
    ''').fetchall()
    
    # Feedback with user and prediction info
    recent_feedback = conn.execute('''
        SELECT f.*, u.username, p.prediction_result
        FROM feedback f
        JOIN users u ON f.user_id = u.id
        JOIN predictions p ON f.prediction_id = p.id
        ORDER BY f.created_at DESC 
        LIMIT 10
    ''').fetchall()
    
    # Prediction statistics
    prediction_stats = conn.execute('''
        SELECT 
            prediction_result,
            COUNT(*) as count,
            AVG(prediction_confidence) as avg_confidence
        FROM predictions 
        GROUP BY prediction_result
    ''').fetchall()
    
    # Rating distribution
    rating_distribution = conn.execute('''
        SELECT rating, COUNT(*) as count
        FROM feedback 
        WHERE rating IS NOT NULL
        GROUP BY rating 
        ORDER BY rating
    ''').fetchall()
    
    # Daily predictions data for line graphs
    daily_predictions = conn.execute('''
        SELECT 
            DATE(created_at) as date,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN prediction_result = 1 THEN 1 ELSE 0 END) as pcos_count,
            SUM(CASE WHEN prediction_result = 0 THEN 1 ELSE 0 END) as normal_count
        FROM predictions 
        WHERE created_at >= date('now', '-30 days')
        GROUP BY DATE(created_at)
        ORDER BY date
    ''').fetchall()
    
    # Weekly trends
    weekly_trends = conn.execute('''
        SELECT 
            strftime('%Y-%W', created_at) as week,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN prediction_result = 1 THEN 1 ELSE 0 END) as pcos_count,
            SUM(CASE WHEN prediction_result = 0 THEN 1 ELSE 0 END) as normal_count
        FROM predictions 
        WHERE created_at >= date('now', '-90 days')
        GROUP BY strftime('%Y-%W', created_at)
        ORDER BY week
    ''').fetchall()
    
    # User registration trends
    user_registrations = conn.execute('''
        SELECT 
            DATE(created_at) as date,
            COUNT(*) as new_users
        FROM users 
        WHERE created_at >= date('now', '-30 days')
        GROUP BY DATE(created_at)
        ORDER BY date
    ''').fetchall()

    user_predictions_timeline = conn.execute('''
        SELECT 
            u.username,
            p.prediction_result,
            p.prediction_confidence,
            p.created_at
        FROM predictions p
        JOIN users u ON p.user_id = u.id
        WHERE p.created_at >= date('now', '-30 days')
        ORDER BY p.created_at
    ''').fetchall()
    
    conn.close()
    
    # Generate chart data
    daily_chart_data = generate_daily_chart_data(daily_predictions)
    weekly_chart_data = generate_weekly_chart_data(weekly_trends)
    user_chart_data = generate_user_chart_data(user_registrations)
    timeline_chart_data = generate_timeline_chart_data(user_predictions_timeline)
    
    return render_template('admin_dashboard.html',
                         total_users=total_users,
                         total_predictions=total_predictions,
                         total_feedback=total_feedback,
                         avg_rating=round(avg_rating, 2),
                         recent_predictions=recent_predictions,
                         recent_feedback=recent_feedback,
                         prediction_stats=prediction_stats,
                         rating_distribution=rating_distribution,
                         daily_chart_data=daily_chart_data,
                         weekly_chart_data=weekly_chart_data,
                         user_chart_data=user_chart_data,
                         timeline_chart_data=timeline_chart_data) 

# User feedback route
@app.route('/feedback', methods=['POST'])
def submit_feedback():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first!'})
    
    try:
        prediction_id = request.json.get('prediction_id')
        rating = request.json.get('rating')
        comments = request.json.get('comments', '')
        
        conn = get_db_connection()
        
        # Check if prediction belongs to user
        prediction = conn.execute(
            'SELECT id FROM predictions WHERE id = ? AND user_id = ?',
            (prediction_id, session['user_id'])
        ).fetchone()
        
        if not prediction:
            conn.close()
            return jsonify({'success': False, 'message': 'Prediction not found!'})
        
        # Insert feedback
        conn.execute('''
            INSERT INTO feedback (prediction_id, user_id, rating, comments)
            VALUES (?, ?, ?, ?)
        ''', (prediction_id, session['user_id'], rating, comments))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Feedback submitted successfully!'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

# Update prediction result template to include feedback
@app.route('/prediction/<int:prediction_id>')
def prediction_detail(prediction_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    
    # Get prediction with user check
    prediction = conn.execute('''
        SELECT p.*, f.rating, f.comments, f.created_at as feedback_date
        FROM predictions p
        LEFT JOIN feedback f ON p.id = f.prediction_id AND f.user_id = ?
        WHERE p.id = ? AND p.user_id = ?
    ''', (session['user_id'], prediction_id, session['user_id'])).fetchone()
    
    conn.close()
    
    if not prediction:
        flash('Prediction not found!', 'error')
        return redirect(url_for('history'))
    
    # Generate recommendations (you might want to store these in database)
    recommendations = generate_recommendations(
        prediction['prediction_result'],
        prediction['age'],
        prediction['weight'],
        prediction['bmi']
    )
    
    return render_template('prediction_detail.html',
                         prediction=prediction,
                         recommendations=recommendations)
if __name__ == '__main__':
    init_chatbot()
    app.run(debug=True)