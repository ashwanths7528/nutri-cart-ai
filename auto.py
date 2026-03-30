import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 0 = all logs, 3 = only errors

# Optional (removes oneDNN message)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import layers, models

print("🧠 Starting DL Training Process (Autoencoder)...")

# Database path
DB_PATH = r"D:\Learning Curve\autoenc\smart_trolley.db"
conn = sqlite3.connect(DB_PATH)

# Load data
df = pd.read_sql_query("""
SELECT product_id, product_name, category, sub_category, calories, sugar, fat, fiber, protein
FROM products
""", conn)
conn.close()

# Features
feature_cols = ["calories", "sugar", "fat", "fiber", "protein"]

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[feature_cols])

# =========================
# 🔥 AUTOENCODER MODEL
# =========================

input_dim = scaled_features.shape[1]  # 5 features
latent_dim = 4  # compressed representation

# Encoder
input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(8, activation="relu")(input_layer)
latent = layers.Dense(4, activation="relu")(encoded)

# Decoder
decoded = layers.Dense(4, activation="relu")(latent)
output_layer = layers.Dense(input_dim, activation="linear")(decoded)

# Model
autoencoder = models.Model(input_layer, output_layer)
encoder = models.Model(input_layer, latent)  # for inference

# Compile
autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

# Train
print("📊 Training Autoencoder...")
autoencoder.fit(
    scaled_features,
    scaled_features,
    epochs=200,
    batch_size=8,
    verbose=1
)

# =========================
# 🔥 GENERATE LATENT FEATURES
# =========================

latent_vectors = encoder.predict(scaled_features)

# Save everything
joblib.dump(scaler, "scaler.pkl")
joblib.dump(latent_vectors, "latent_vectors.pkl")
df.to_pickle("product_features.pkl")
autoencoder.save("autoencoder_model.h5")
encoder.save("encoder_model.h5")

print("✅ Autoencoder trained successfully!")