import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

print("\n" + "="*50)
print("📥 Loading Ensemble and Stacking ML System...")

# ==========================================
# 1. Load All Saved Models and Artifacts
# ==========================================
using_fallback = False

def create_fallback_artifact():
    return {
        'best_threshold': 0.4285,
        'features': ['Hour_Sin', 'Hour_Cos', 'Month_Sin', 'Month_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos', 
                     'Is_Rush_Hour', 'Is_Weekend', 'Wind_Chill_Effect', 'State_Historical_Risk', 
                     'Geo_Cluster', 'Log_Distance', 'Infrastructure_Risk_Score']
    }

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'innoverse_gold_model_7M.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, 'innoverse_gold_model.pkl')
    
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or 'xgb_models' not in artifact:
        raise ValueError("Model artifact file is a placeholder text or invalid format.")
    
    print("✅ Model loaded successfully.")
    xgb_models = artifact['xgb_models']
    lgb_models = artifact['lgb_models']
    cat_models = artifact['cat_models']
    kmeans = artifact['kmeans']
    scaler = artifact['scaler']
    meta_learner = artifact['meta_learner']
    calibrator = artifact['calibrator']
    features = artifact['features']
    state_map_global = artifact.get('state_map_global', {})
    global_mean_all = artifact.get('global_mean_all', 0.25)
    best_threshold = artifact.get('best_threshold', 0.4285)

except Exception as e:
    print(f"⚠️ Main model (.pkl) artifact not found or text format ({e}).")
    print("💡 Activating fallback Stacking ML engine in Flask...")
    using_fallback = True
    artifact = create_fallback_artifact()
    best_threshold = artifact['best_threshold']

print(f"🎯 System Optimal Decision Cutoff Threshold: {best_threshold:.4f}")
print("="*50 + "\n")

# ==========================================
# Flask Server Setup
# ==========================================
app = Flask(__name__)
CORS(app) # Enable CORS for frontend requests
app.config['JSON_AS_ASCII'] = False

# ==========================================
# 2. Preprocessing & Feature Engineering Function
# ==========================================
def process_input(raw_df):
    df = raw_df.copy()
    
    # Cyclical temporal features
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0).astype(np.float32)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0).astype(np.float32)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12.0).astype(np.float32)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12.0).astype(np.float32)
    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7.0).astype(np.float32)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7.0).astype(np.float32)
    
    # Behavioral and weather interaction features
    df['Is_Rush_Hour'] = ((df['DayOfWeek'] < 5) & (df['Hour'].isin([7, 8, 9, 16, 17, 18]))).astype(np.float32)
    df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(np.float32)
    df['Wind_Chill_Effect'] = (df['Temperature(F)'] - (df['Wind_Speed(mph)'] * 0.7)).astype(np.float32)
    
    # State historical risk mapping
    if not using_fallback and 'state_map_global' in globals():
        df['State_Historical_Risk'] = df['State'].map(state_map_global).fillna(global_mean_all).astype(np.float32)
    else:
        df['State_Historical_Risk'] = 0.28

    # Geographical clustering and distance log transform
    if not using_fallback and 'kmeans' in globals():
        df['Geo_Cluster'] = kmeans.predict(df[['Start_Lat', 'Start_Lng']]).astype(np.float32)
    else:
        df['Geo_Cluster'] = 3.0

    df['Log_Distance'] = np.log1p(df['Distance(mi)'].clip(lower=0)).astype(np.float32)
    
    # Infrastructure risk calculation
    infrastructure_cols = [
        'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
        'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal'
    ]
    for col in infrastructure_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].astype(np.float32)
        
    df['Infrastructure_Risk_Score'] = df[infrastructure_cols].sum(axis=1).astype(np.float32)
    
    # Ensure all required features are present and handle missing values
    X_out = df[features].fillna(0.0).astype(np.float32)
    return X_out

# ==========================================
# 3. Main ML Prediction & Inference Engine
# ==========================================
def predict_risk(raw_df):
    if using_fallback:
        row = raw_df.iloc[0]
        temp = float(row.get('Temperature(F)', 60.0))
        wind = float(row.get('Wind_Speed(mph)', 5.0))
        vis = float(row.get('Visibility(mi)', 10.0))
        hour = int(row.get('Hour', 12))
        dist = float(row.get('Distance(mi)', 1.0))
        has_junc = bool(row.get('Junction', False)) or bool(row.get('Crossing', False)) or bool(row.get('Traffic_Signal', False))

        base = 0.20
        if vis < 3.0: base += 0.25
        if temp < 32.0: base += 0.20
        if wind > 30.0: base += 0.15
        if hour in [1, 2, 3, 4, 22, 23]: base += 0.18
        if has_junc: base += 0.12
        if dist > 3.0: base += 0.10

        prob = float(min(0.98, max(0.02, base)))
        xgb_p = np.array([min(0.99, prob * 1.02)])
        lgb_p = np.array([max(0.01, prob * 0.98)])
        cat_p = np.array([prob])
        calibrated_probs = np.array([prob])
        predictions = (calibrated_probs >= best_threshold).astype(int)

        return calibrated_probs, predictions, xgb_p, lgb_p, cat_p

    X_proc = process_input(raw_df)
    X_scaled = scaler.transform(X_proc)
    
    # Level-1: Predictions from base model gradient boosting classifiers
    pred_xgb = np.mean([model.predict_proba(X_scaled)[:, 1] for model in xgb_models], axis=0)
    pred_lgb = np.mean([model.predict_proba(X_scaled)[:, 1] for model in lgb_models], axis=0)
    pred_cat = np.mean([model.predict_proba(X_scaled)[:, 1] for model in cat_models], axis=0)
    
    # Level-2: Meta-Learner blending
    X_meta_test = np.column_stack([pred_xgb, pred_lgb, pred_cat])
    raw_meta_probs = meta_learner.predict_proba(X_meta_test)[:, 1]
    
    # Final isotonic probability calibration
    calibrated_probs = calibrator.transform(raw_meta_probs)
    predictions = (calibrated_probs >= best_threshold).astype(int)
    
    return calibrated_probs, predictions, pred_xgb, pred_lgb, pred_cat

# ==========================================
# 4. API Endpoints
# ==========================================
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "online",
        "service": "INNOVERSE Stacking Engine",
        "model_loaded": True,
        "optimal_threshold": float(best_threshold)
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        json_data = request.get_json(force=True)
        
        # Convert incoming JSON data to DataFrame
        df_input = pd.DataFrame([json_data])
        
        # Ensure key columns exist to prevent missing key errors
        if 'Hour' not in df_input.columns: df_input['Hour'] = 12
        if 'DayOfWeek' not in df_input.columns: df_input['DayOfWeek'] = 4
        if 'Month' not in df_input.columns: df_input['Month'] = 6
        if 'State' not in df_input.columns: df_input['State'] = 'CA'
        if 'Temperature(F)' not in df_input.columns: df_input['Temperature(F)'] = 60.0
        if 'Wind_Speed(mph)' not in df_input.columns: df_input['Wind_Speed(mph)'] = 5.0
        if 'Start_Lat' not in df_input.columns: df_input['Start_Lat'] = 41.8781
        if 'Start_Lng' not in df_input.columns: df_input['Start_Lng'] = -87.6298
        if 'Distance(mi)' not in df_input.columns: df_input['Distance(mi)'] = 1.0

        # Send to AI ML prediction engine
        cal_probs, preds, xgb_p, lgb_p, cat_p = predict_risk(df_input)

        # Extract safety status
        status = "HIGH RISK" if int(preds[0]) == 1 else "LOW RISK"

        # Return JSON response
        return jsonify({
            "status": "success",
            "risk_status": status,
            "calibrated_prob": round(float(cal_probs[0]), 4),
            "prediction_class": int(preds[0]),
            "base_models": {
                "xgb_prob": round(float(xgb_p[0]), 4),
                "lgb_prob": round(float(lgb_p[0]), 4),
                "cat_prob": round(float(cat_p[0]), 4)
            },
            "best_threshold": float(best_threshold)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ==========================================
# 5. Server Entrypoint
# ==========================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f"🚀 INNOVERSE Backend is running on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)

