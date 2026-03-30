import numpy as np
import pandas as pd
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 0 = all logs, 3 = only errors

# Optional (removes oneDNN message)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import load_model

print("🔍 Testing Autoencoder Recommender...")

# Load everything
encoder = load_model("encoder_model.h5")
latent_vectors = joblib.load("latent_vectors.pkl")
df = pd.read_pickle("product_features.pkl")
scaler = joblib.load("scaler.pkl")

def health_score(row):
    return (
        row["fiber"] * 6 +          # 🔥 strongest signal
        row["protein"] * 2 -
        row["sugar"] * 3 -
        row["fat"] * 2 -
        row["calories"] * 0.03
    )
def is_healthier(candidate, base):
    return health_score(candidate) > health_score(base)

def recommend_best(product_name):
    matches = df[df["product_name"].str.lower() == product_name.lower()]
    
    if matches.empty:
        print("❌ Product not found")
        return
    
    idx = matches.index[0]
    base = df.loc[idx]

    same_cat_df = df[
        (df["category"] == base["category"]) &
        (df["sub_category"] == base["sub_category"])
    ]
    same_cat_idx = same_cat_df.index

    target_vec = latent_vectors[idx]
    filtered_vectors = latent_vectors[same_cat_idx]

    distances = np.linalg.norm(filtered_vectors - target_vec, axis=1)

    best_item = None
    best_score = float("-inf")

    base_score = health_score(base)

    for i, real_idx in enumerate(same_cat_idx):
        if real_idx == idx:
            continue

        candidate = df.loc[real_idx]

        if not is_healthier(candidate, base):
            continue

        # 🔥 Combined scoring
        score = (
            health_score(candidate) - base_score
            - 0.5 * distances[i]   # penalty for dissimilarity
        )

        if score > best_score:
            best_score = score
            best_item = candidate

    print(f"\n🛒 Selected: {base['product_name']}")
    print("💡 Best Alternative:\n")

    if best_item is None:
        print("⚠️ No better alternative found")
    else:
        print(f"- {best_item['product_name']}")
recommend_best("Nutri Choice")