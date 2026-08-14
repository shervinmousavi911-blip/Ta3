import joblib
import numpy as np
import pandas as pd

print("📥 Loading Ensemble and Stacking system...")

# 1. Load all saved models and artifacts
try:
    artifact = joblib.load('innoverse_gold_model.pkl')
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ File not found error! Run train.py first to generate innoverse_gold_model.pkl.")
    exit()

xgb_models = artifact['xgb_models']
lgb_models = artifact['lgb_models']
cat_models = artifact['cat_models']
kmeans = artifact['kmeans']
scaler = artifact['scaler']
meta_learner = artifact['meta_learner']
calibrator = artifact['calibrator']
features = artifact['features']
state_map_global = artifact['state_map_global']
global_mean_all = artifact['global_mean_all']
best_threshold = artifact['best_threshold']

# 2. Preprocessing and Feature Engineering Function
def process_input(raw_df):
    df = raw_df.copy()
    
    # Cyclical Time Features
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0).astype(np.float32)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0).astype(np.float32)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12.0).astype(np.float32)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12.0).astype(np.float32)
    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7.0).astype(np.float32)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7.0).astype(np.float32)
    
    # Behavioral and Weather Features
    df['Is_Rush_Hour'] = ((df['DayOfWeek'] < 5) & (df['Hour'].isin([7, 8, 9, 16, 17, 18]))).astype(np.float32)
    df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(np.float32)
    df['Wind_Chill_Effect'] = (df['Temperature(F)'] - (df['Wind_Speed(mph)'] * 0.7)).astype(np.float32)
    
    # State Risk Mapping
    df['State_Historical_Risk'] = df['State'].map(state_map_global).fillna(global_mean_all).astype(np.float32)
    
    # Geographical Clustering and Distance Log
    df['Geo_Cluster'] = kmeans.predict(df[['Start_Lat', 'Start_Lng']]).astype(np.float32)
    df['Log_Distance'] = np.log1p(df['Distance(mi)'].clip(lower=0)).astype(np.float32)
    
    # Urban Infrastructure
    infrastructure_cols = [
        'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
        'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal'
    ]
    for col in infrastructure_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].astype(np.float32)
        
    df['Infrastructure_Risk_Score'] = df[infrastructure_cols].sum(axis=1).astype(np.float32)
    
    # Ensure all features exist and fill missing values
    X_out = df[features].fillna(df[features].median()).astype(np.float32)
    return X_out

# 3. Main Evaluation and Inference Function
def predict_risk(raw_df):
    X_proc = process_input(raw_df)
    X_scaled = scaler.transform(X_proc)
    
    # First Stage (Level-1): Get predictions from all 15 models
    pred_xgb = np.mean([model.predict_proba(X_scaled)[:, 1] for model in xgb_models], axis=0)
    pred_lgb = np.mean([model.predict_proba(X_scaled)[:, 1] for model in lgb_models], axis=0)
    pred_cat = np.mean([model.predict_proba(X_scaled)[:, 1] for model in cat_models], axis=0)
    
    # Second Stage (Level-2): Smart combination with Meta-Learner
    X_meta_test = np.column_stack([pred_xgb, pred_lgb, pred_cat])
    raw_meta_probs = meta_learner.predict_proba(X_meta_test)[:, 1]
    
    # Final Isotonic Calibration
    calibrated_probs = calibrator.transform(raw_meta_probs)
    predictions = (calibrated_probs >= best_threshold).astype(int)
    
    return calibrated_probs, predictions, pred_xgb, pred_lgb, pred_cat

# ==========================================
# 4. Testing on Sample Scenarios
# ==========================================
test_cases = pd.DataFrame([
    { # Scenario 1: Highly Dangerous (Severe night storm at a busy Illinois intersection)
        'Start_Lat': 41.8781, 'Start_Lng': -87.6298, 'State': 'IL', 'Distance(mi)': 12.5,
        'Hour': 2, 'DayOfWeek': 4, 'Month': 1, 'Temperature(F)': -10.0, 'Humidity(%)': 98.0,
        'Pressure(in)': 28.10, 'Visibility(mi)': 0.1, 'Wind_Speed(mph)': 50.0,
        'Amenity': 0.0, 'Bump': 0.0, 'Crossing': 1.0, 'Give_Way': 0.0, 'Junction': 1.0,
        'No_Exit': 0.0, 'Railway': 0.0, 'Roundabout': 0.0, 'Station': 0.0, 'Stop': 0.0,
        'Traffic_Calming': 0.0, 'Traffic_Signal': 1.0
    },
    { # Scenario 2: Very Safe (Ideal sunny day in California)
        'Start_Lat': 34.0522, 'Start_Lng': -118.2437, 'State': 'CA', 'Distance(mi)': 0.1,
        'Hour': 3, 'DayOfWeek': 1, 'Month': 6, 'Temperature(F)': 75.0, 'Humidity(%)': 20.0,
        'Pressure(in)': 29.95, 'Visibility(mi)': 10.0, 'Wind_Speed(mph)': 2.0,
        'Amenity': 0.0, 'Bump': 0.0, 'Crossing': 0.0, 'Give_Way': 0.0, 'Junction': 0.0,
        'No_Exit': 0.0, 'Railway': 0.0, 'Roundabout': 0.0, 'Station': 0.0, 'Stop': 0.0,
        'Traffic_Calming': 0.0, 'Traffic_Signal': 0.0
    }
    # Expected output = 13% 
])

cal_probs, preds, xgb_p, lgb_p, cat_p = predict_risk(test_cases)

print("\n" + "=" * 75)
print("🏅 INNOVERSE STACKING CHAMPIONSHIP REPORT")
print(f"🎯 Dynamic Optimal Threshold: {best_threshold:.4f}")
print("=" * 75)

titles = [
    "Scenario 1: Critical Condition (Storm, Night, Intersection)",
    "Scenario 2: Normal Condition (Sunny Day, Standard Route)"
]

for i in range(len(test_cases)):
    status = "🚨 HIGH RISK (Class 1)" if preds[i] == 1 else "✅ LOW RISK (Class 0)"
    print(f"\n📌 {titles[i]}")
    print(f" ├─ Final System Evaluation: {status}")
    print(f" ├─ Calibrated Risk Probability: {cal_probs[i]*100:.2f}%")
    print(f" └─ Base Models Output: XGB ({xgb_p[i]*100:.1f}%) | LGB ({lgb_p[i]*100:.1f}%) | CAT ({cat_p[i]*100:.1f}%)")

print("\n" + "=" * 75)

# ==========================================
# 5. Usage Guide for New CSV Files
# ==========================================
# If you want to evaluate a new CSV file, uncomment the following code:
"""
new_data = pd.read_csv('your_test_file.csv')
probs, predictions, _, _, _ = predict_risk(new_data)
new_data['Risk_Probability'] = probs
new_data['Predicted_Risk_Class'] = predictions
new_data.to_csv('predictions_output.csv', index=False)
print("✅ Predictions saved to predictions_output.csv file.")
"""