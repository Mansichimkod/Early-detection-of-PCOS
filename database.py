import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash

def init_db():
    conn = sqlite3.connect('pcos_detection.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            age INTEGER,
            weight REAL,
            bmi REAL,
            cycle_pattern TEXT,
            hip REAL,
            waist REAL,
            weight_gain INTEGER,
            hair_growth INTEGER,
            skin_darkening INTEGER,
            hair_loss INTEGER,
            pimples INTEGER,
            fast_food INTEGER,
            follicle_left INTEGER,
            follicle_right INTEGER,
            follicle_avg_size REAL,
            prediction_result INTEGER,
            prediction_confidence REAL,
            shap_values BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Feedback table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER,
            user_id INTEGER,
            rating INTEGER CHECK(rating >= 1 AND rating <= 5),
            comments TEXT,
            helpful INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (prediction_id) REFERENCES predictions (id),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Admin users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admin_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert default admin user if not exists
    default_admin_password = generate_password_hash('admin123')
    cursor.execute('''
        INSERT OR IGNORE INTO admin_users (username, email, password) 
        VALUES (?, ?, ?)
    ''', ('admin', 'admin@pcos.com', default_admin_password))
    
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect('pcos_detection.db')
    conn.row_factory = sqlite3.Row
    return conn