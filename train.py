import os
import warnings
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings('ignore')

print("🚀 [INNOVERSE GOLD CHAMPIONSHIP] Starting Multi-stage Architecture Training (Level-2 Stacking)...")

# 1. Optimized Data Loading
needed_cols = [
    'Severity', 'Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng',
    'Distance(mi)', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
    'Visibility(mi)', 'Wind_Speed(mph)', 'Amenity', 'Bump', 'Crossing',
    'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
    'Stop', 'Traffic_Calming', 'Traffic_Signal', 'State'
]

df = pd.read_csv('US_Accidents_March23.csv', usecols=needed_cols)
print(f"📦 Total initial samples: {len(df):,}")

# 2. Cleaning and Time Processing
df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed', errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], format='mixed', errors='coerce')
df = df.dropna(subset=['Start_Time', 'Start_Lat', 'Start_Lng', 'State']).reset_index(drop=True)

df['Hour'] = df['Start_Time'].dt.hour.astype(np.float32)
df['DayOfWeek'] = df['Start_Time'].dt.dayofweek.astype(np.float32)
df['Month'] = df['Start_Time'].dt.month.astype(np.float32)
df['Duration_Minutes'] = ((df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60.0).astype(np.float32)

df = df[(df['Duration_Minutes'] >= 0) & (df['Duration_Minutes'] <= 1440)].reset_index(drop=True)

# 3. Composite Target Variable (with np.int32 type to prevent Overflow)
weather_hazard_temp = (
    (df['Humidity(%)'] / (df['Visibility(mi)'].replace(0, 0.01) + 0.1)) *
    (100 / (df['Temperature(F)'] + 60).replace(0, 0.01))
).astype(np.float32)

df['Risk_Class'] = (
    (df['Severity'] >= 3) | 
    ((weather_hazard_temp > 25.0) & (df['Severity'] >= 2)) | 
    (df['Duration_Minutes'] > 240.0)
).astype(np.int32)

del weather_hazard_temp

# 4. Advanced Feature Engineering
df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0).astype(np.float32)
df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0).astype(np.float32)
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12.0).astype(np.float32)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12.0).astype(np.float32)
df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7.0).astype(np.float32)
df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7.0).astype(np.float32)

df['Is_Rush_Hour'] = ((df['DayOfWeek'] < 5) & (df['Hour'].isin([7, 8, 9, 16, 17, 18]))).astype(np.float32)
df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(np.float32)
df['Wind_Chill_Effect'] = (df['Temperature(F)'] - (df['Wind_Speed(mph)'] * 0.7)).astype(np.float32)
df['Log_Distance'] = np.log1p(df['Distance(mi)'].clip(lower=0)).astype(np.float32)

infrastructure_cols = [
    'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
    'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal'
]
for col in infrastructure_cols:
    df[col] = df[col].astype(np.float32)

df['Infrastructure_Risk_Score'] = df[infrastructure_cols].sum(axis=1).astype(np.float32)

# Geographical Clustering
print("🗺️ Performing geographical clustering...")
kmeans = MiniBatchKMeans(n_clusters=50, random_state=42, batch_size=8192, n_init='auto')
df['Geo_Cluster'] = kmeans.fit_predict(df[['Start_Lat', 'Start_Lng']]).astype(np.float32)

global_mean_all = float(df['Risk_Class'].mean())
state_map_global = df.groupby('State')['Risk_Class'].mean().to_dict()

features = [
    'Start_Lat', 'Start_Lng', 'Geo_Cluster', 'State_Historical_Risk', 'Log_Distance',
    'Hour_Sin', 'Hour_Cos', 'Month_Sin', 'Month_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos',
    'Is_Rush_Hour', 'Is_Weekend', 'Wind_Chill_Effect',
    'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
    'Infrastructure_Risk_Score'
] + infrastructure_cols

# 5. Cross-Validation and Training
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

df['State_Historical_Risk'] = 0.0
oof_xgb = np.zeros(len(df), dtype=np.float32)
oof_lgb = np.zeros(len(df), dtype=np.float32)
oof_cat = np.zeros(len(df), dtype=np.float32)

xgb_models, lgb_models, cat_models = [], [], []
y = df['Risk_Class'].values

print(f"\n⚡ Starting 5-fold training on {len(df):,} samples...")

for fold, (train_idx, val_idx) in enumerate(skf.split(df, y)):
    print(f"\n--- [Fold {fold + 1}/{N_SPLITS}] ---")
    
    fold_train_df = df.iloc[train_idx]
    state_map_fold = fold_train_df.groupby('State')['Risk_Class'].mean().to_dict()
    fold_global_mean = fold_train_df['Risk_Class'].mean()
    
    df.loc[train_idx, 'State_Historical_Risk'] = df.iloc[train_idx]['State'].map(state_map_fold).fillna(fold_global_mean).values
    df.loc[val_idx, 'State_Historical_Risk'] = df.iloc[val_idx]['State'].map(state_map_fold).fillna(fold_global_mean).values
    
    X_fold = df[features].fillna(df[features].median()).astype(np.float32)
    
    scaler_fold = RobustScaler()
    X_tr = scaler_fold.fit_transform(X_fold.iloc[train_idx])
    X_va = scaler_fold.transform(X_fold.iloc[val_idx])
    
    y_tr, y_va = y[train_idx], y[val_idx]
    
    # Safe calculation without Overflow
    num_pos = int(np.sum(y_tr, dtype=np.int64))
    pos_weight = float((len(y_tr) - num_pos) / num_pos)
    
    # Model 1: XGBoost
    xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight, random_state=42, n_jobs=-1, tree_method='hist')
    xgb.fit(X_tr, y_tr)
    oof_xgb[val_idx] = xgb.predict_proba(X_va)[:, 1]
    xgb_models.append(xgb)
    
    # Model 2: LightGBM
    lgb = LGBMClassifier(n_estimators=400, max_depth=6, num_leaves=31, learning_rate=0.04, subsample=0.8, scale_pos_weight=pos_weight, random_state=42, n_jobs=-1, verbose=-1)
    lgb.fit(X_tr, y_tr)
    oof_lgb[val_idx] = lgb.predict_proba(X_va)[:, 1]
    lgb_models.append(lgb)
    
    # Model 3: CatBoost
    cat = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.04, scale_pos_weight=pos_weight, random_seed=42, verbose=0)
    cat.fit(X_tr, y_tr)
    oof_cat[val_idx] = cat.predict_proba(X_va)[:, 1]
    cat_models.append(cat)
    
    print(f"   ├─ XGB AUC: {roc_auc_score(y_va, oof_xgb[val_idx]):.4f} | LGB AUC: {roc_auc_score(y_va, oof_lgb[val_idx]):.4f} | CAT AUC: {roc_auc_score(y_va, oof_cat[val_idx]):.4f}")

# 6. Meta-Learner Training (Stacking)
print("\n🧠 Training level-two model (Meta-Learner Stacking)...")
X_meta_train = np.column_stack([oof_xgb, oof_lgb, oof_cat])
meta_learner = LogisticRegression(C=1.0, max_iter=1000)
meta_learner.fit(X_meta_train, y)

oof_meta_preds = meta_learner.predict_proba(X_meta_train)[:, 1]
meta_auc = roc_auc_score(y, oof_meta_preds)
print(f"🏆 [Total Stacking OOF ROC-AUC Score]: {meta_auc:.5f}")

# 7. Calibration and Saving
print("🎯 Applying Isotonic Calibration...")
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrated_oof = calibrator.fit_transform(oof_meta_preds, y)

precisions, recalls, thresholds = precision_recall_curve(y, calibrated_oof)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold = float(thresholds[np.argmax(f1_scores)])

final_scaler = RobustScaler()
final_X = df[features].fillna(df[features].median()).astype(np.float32)
final_scaler.fit(final_X)

gold_artifact = {
    'xgb_models': xgb_models,
    'lgb_models': lgb_models,
    'cat_models': cat_models,
    'kmeans': kmeans,
    'scaler': final_scaler,
    'meta_learner': meta_learner,
    'calibrator': calibrator,
    'features': features,
    'state_map_global': state_map_global,
    'global_mean_all': global_mean_all,
    'best_threshold': best_threshold
}

joblib.dump(gold_artifact, 'innoverse_gold_model.pkl')
print(f"\n✅ [Total Success] All 15 models and Stacking system saved without errors! Optimal absolute threshold: {best_threshold:.4f}")