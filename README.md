# 🚦 INNOVERSE Road Accident Risk Prediction System

> **The 5th Edition of the INNOVERSE Invention & Innovation Expo – Online Edition**

An AI-powered road accident risk assessment platform designed to identify **high-risk traffic conditions and geographical areas from historical accident data**.

The project combines **XGBoost, LightGBM, CatBoost, geographical clustering, feature engineering, stacking, probability calibration, and a Flask REST API** into a multi-stage machine-learning system.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Goals](#-project-goals)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Dataset](#-dataset)
- [Target Definition](#-target-definition)
- [Feature Engineering](#-feature-engineering)
- [Ensemble and Stacking](#-ensemble-and-stacking)
- [Probability Calibration](#-probability-calibration)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Training the Model](#-training-the-model)
- [Running Predictions](#-running-predictions)
- [Running the Flask API](#-running-the-flask-api)
- [API Documentation](#-api-documentation)
- [Example API Request](#-example-api-request)
- [Example API Response](#-example-api-response)
- [Testing Scenarios](#-testing-scenarios)
- [Input Features](#-input-features)
- [Output Interpretation](#-output-interpretation)
- [Model Artifact](#-model-artifact)
- [Fallback Engine](#-fallback-engine)
- [Important Scientific Notes](#-important-scientific-notes)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Competition Context](#-competition-context)
- [License](#-license)
- [Disclaimer](#-disclaimer)

---

## 🧠 Overview

The **INNOVERSE Road Accident Risk Prediction System** is a machine-learning project for estimating the risk class associated with a road-accident scenario using historical accident records.

The system considers several categories of information:

- Geographic location
- State
- Distance
- Time of day
- Day of week
- Month
- Temperature
- Humidity
- Atmospheric pressure
- Visibility
- Wind speed
- Road infrastructure
- Intersections and junctions
- Traffic signals
- Railway and station proximity
- Rush-hour conditions
- Weekend conditions

The final prediction engine is not based on a single classifier. Instead, it uses multiple gradient-boosting models and a second-level stacking model.

### Main technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Joblib
- Flask
- Flask-CORS

---

# 🎯 Project Goals

The main objectives are:

1. Analyze historical road-accident data.
2. Extract temporal, geographical, weather, and infrastructure patterns.
3. Identify conditions associated with higher accident risk.
4. Build a robust ensemble machine-learning system.
5. Combine predictions from multiple models.
6. Calibrate the final probability.
7. Expose the trained model through a REST API.
8. Provide a foundation for an interactive road-safety application.

The project is intended as a **decision-support and research system**, not as a replacement for professional traffic-safety analysis.

---

# ✨ Key Features

## 1. Multi-model ensemble

The training system uses three independent gradient-boosting algorithms:

- **XGBoost**
- **LightGBM**
- **CatBoost**

Each model learns the same engineered feature space using a different learning strategy.

---

## 2. 5-Fold Cross-Validation

The training pipeline uses:

```text
StratifiedKFold
n_splits = 5
shuffle = True
random_state = 42
```

This produces five validation folds and generates out-of-fold predictions for the stacking stage.

---

## 3. Level-2 Stacking

The predictions of XGBoost, LightGBM, and CatBoost are combined into a second-level model:

```text
XGBoost probability ─┐
                     ├──> Logistic Regression ──> Calibration ──> Final Risk
LightGBM probability ┤
                     │
CatBoost probability ┘
```

The meta-learner is:

```python
LogisticRegression(C=1.0, max_iter=1000)
```

---

## 4. Probability Calibration

The final meta-model probability is calibrated using:

```python
IsotonicRegression(out_of_bounds='clip')
```

This is intended to make the reported probability better aligned with the observed target frequency than an uncalibrated classifier score.

---

## 5. Geographical Clustering

The system uses:

```python
MiniBatchKMeans(
    n_clusters=50,
    random_state=42,
    batch_size=8192
)
```

on:

```text
Start_Lat
Start_Lng
```

This converts raw coordinates into a geographical cluster feature.

The resulting feature is:

```text
Geo_Cluster
```

---

## 6. Historical State Risk

A historical risk statistic is calculated for each US state:

```text
State_Historical_Risk
```

During cross-validation, the state statistic is calculated from the training portion of each fold before being applied to the validation portion.

This reduces direct target leakage during the cross-validation feature-generation stage.

---

# 🏗️ System Architecture

The complete system can be represented as:

```text
                 US_Accidents_March23.csv
                           │
                           ▼
                  Data Loading & Cleaning
                           │
                           ▼
                    Time Extraction
                           │
                           ▼
                   Target Construction
                           │
                           ▼
                  Feature Engineering
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          XGBoost       LightGBM      CatBoost
              │            │            │
              └────────────┼────────────┘
                           ▼
                    OOF Predictions
                           │
                           ▼
                  Logistic Regression
                    Meta-Learner
                           │
                           ▼
                  Isotonic Calibration
                           │
                           ▼
                 Optimal F1 Threshold
                           │
                           ▼
                  Saved Model Artifact
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
          test.py                  Flask app.py
              │                         │
              ▼                         ▼
       Batch Prediction             REST API
```

---

# 🔬 Machine Learning Pipeline

## Step 1 — Data Loading

The training script reads:

```text
US_Accidents_March23.csv
```

Only the columns required by the model are loaded.

This reduces unnecessary memory usage compared with loading the entire dataset.

The training code selects fields such as:

- `Severity`
- `Start_Time`
- `End_Time`
- `Start_Lat`
- `Start_Lng`
- `Distance(mi)`
- `Temperature(F)`
- `Humidity(%)`
- `Pressure(in)`
- `Visibility(mi)`
- `Wind_Speed(mph)`
- infrastructure indicators
- `State`

---

## Step 2 — Data Cleaning

The timestamps are converted using:

```python
pd.to_datetime(..., format='mixed', errors='coerce')
```

Rows without essential values such as:

- `Start_Time`
- `Start_Lat`
- `Start_Lng`
- `State`

are removed.

Accident duration is calculated in minutes:

```text
Duration_Minutes =
(End_Time - Start_Time) / 60
```

Records with negative durations or durations greater than 1440 minutes are removed.

---

# 🎯 Target Definition

The current training implementation creates a binary target called:

```text
Risk_Class
```

The target is constructed from three conditions:

```text
Severity >= 3
OR
(weather hazard score > 25 AND Severity >= 2)
OR
Duration_Minutes > 240
```

In simplified form:

```python
Risk_Class =
    (Severity >= 3)
    OR
    ((WeatherHazard > 25) & (Severity >= 2))
    OR
    (Duration_Minutes > 240)
```

The result is converted into:

```text
0 → lower-risk class
1 → higher-risk class
```

### ⚠️ Important

This means the current system is primarily predicting a **derived accident-risk class from accident records**, rather than predicting whether an accident will occur at a location where no accident has yet happened.

Therefore, the phrase **"risk prediction"** should be interpreted according to this target definition.

A future version intended to predict *accident occurrence before an accident happens* should use a target and dataset design that explicitly includes non-accident observations.

---

# 🧩 Feature Engineering

The system creates several engineered features.

## Cyclical time features

Time is periodic, so the project represents time using sine/cosine transformations.

### Hour

```text
Hour_Sin
Hour_Cos
```

### Month

```text
Month_Sin
Month_Cos
```

### Day of week

```text
DayOfWeek_Sin
DayOfWeek_Cos
```

This prevents the model from treating values such as hour 23 and hour 0 as extremely far apart.

---

## Rush Hour

The system defines rush hour using:

```text
Monday–Friday:
07:00
08:00
09:00
16:00
17:00
18:00
```

Feature:

```text
Is_Rush_Hour
```

---

## Weekend

```text
Is_Weekend
```

is set when:

```text
DayOfWeek >= 5
```

---

## Wind Chill Effect

The engineered feature is:

```text
Wind_Chill_Effect =
Temperature(F) - 0.7 × Wind_Speed(mph)
```

---

## Log Distance

Because accident distance can have a skewed distribution:

```text
Log_Distance = log(1 + Distance)
```

is used.

---

## Infrastructure Risk Score

The system aggregates road-infrastructure indicators:

```text
Amenity
Bump
Crossing
Give_Way
Junction
No_Exit
Railway
Roundabout
Station
Stop
Traffic_Calming
Traffic_Signal
```

into:

```text
Infrastructure_Risk_Score
```

The score is the sum of these binary infrastructure indicators.

---

## Geographic Cluster

Latitude and longitude are grouped into 50 geographical clusters.

```text
Start_Lat
Start_Lng
       │
       ▼
MiniBatchKMeans
       │
       ▼
Geo_Cluster
```

---

# 🤖 Base Models

The project uses three gradient-boosting classifiers.

## XGBoost

Main configuration:

```python
XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=pos_weight,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)
```

---

## LightGBM

Main configuration:

```python
LGBMClassifier(
    n_estimators=400,
    max_depth=6,
    num_leaves=31,
    learning_rate=0.04,
    subsample=0.8,
    scale_pos_weight=pos_weight,
    random_state=42,
    n_jobs=-1
)
```

---

## CatBoost

Main configuration:

```python
CatBoostClassifier(
    iterations=400,
    depth=6,
    learning_rate=0.04,
    scale_pos_weight=pos_weight,
    random_seed=42,
    verbose=0
)
```

---

# ⚖️ Class Imbalance Handling

The training script calculates:

```text
positive samples
negative samples
```

and derives:

```text
pos_weight =
negative_samples / positive_samples
```

This value is passed to the three classifiers through:

```text
scale_pos_weight
```

The purpose is to reduce the effect of class imbalance during model training.

---

# 🧠 Level-2 Stacking

After the five folds are trained, the project has out-of-fold predictions:

```text
OOF XGBoost
OOF LightGBM
OOF CatBoost
```

These are combined into:

```python
X_meta_train = np.column_stack([
    oof_xgb,
    oof_lgb,
    oof_cat
])
```

The meta-learner then learns how to combine the three predictions.

---

# 🎯 Probability Calibration

The meta-model produces a raw probability.

The project then applies:

```python
IsotonicRegression(out_of_bounds='clip')
```

The resulting value is the final calibrated probability:

```text
Calibrated Risk Probability
```

---

# 📏 Decision Threshold

Instead of always using:

```text
0.50
```

as the classification threshold, the training pipeline searches for a threshold that maximizes the F1 score on the calibrated OOF predictions.

The threshold is saved in:

```text
best_threshold
```

The inference rule is:

```python
prediction = calibrated_probability >= best_threshold
```

---

# 📁 Project Structure

A recommended project structure is:

```text
INNOVERSE-Road-Risk/
│
├── train.py
├── test.py
├── app.py
│
├── US_Accidents_March23.csv
│
├── innoverse_gold_model.pkl
│
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
│
└── frontend/
    └── ...
```

### Main files

| File | Purpose |
|---|---|
| `train.py` | Trains the complete multi-stage ML system |
| `test.py` | Runs prediction tests and sample scenarios |
| `app.py` | Flask REST API |
| `US_Accidents_March23.csv` | Historical accident dataset |
| `innoverse_gold_model.pkl` | Saved trained model artifact |
| `.env.example` | Example environment configuration |
| `.gitignore` | Git ignore rules |
| `README.md` | Project documentation |

---

# 📦 Requirements

Recommended Python version:

```text
Python 3.10+
```

Install the main dependencies:

```bash
pip install numpy pandas scikit-learn joblib
pip install xgboost lightgbm catboost
pip install flask flask-cors
```

Or create a `requirements.txt`:

```text
numpy
pandas
scikit-learn
joblib
xgboost
lightgbm
catboost
flask
flask-cors
```

Then install:

```bash
pip install -r requirements.txt
```

---

# 🛠️ Installation

## 1. Clone the project

```bash
git clone <YOUR_REPOSITORY_URL>
cd INNOVERSE-Road-Risk
```

Replace `<YOUR_REPOSITORY_URL>` with the URL of your GitHub repository.

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📊 Dataset

The training script expects:

```text
US_Accidents_March23.csv
```

The project uses historical accident records containing information about:

- Accident severity
- Location
- Time
- Weather
- Road conditions
- Infrastructure
- Duration

The dataset is read directly by `train.py`.

Example:

```python
df = pd.read_csv(
    'US_Accidents_March23.csv',
    usecols=needed_cols
)
```

---

# 🚀 Training the Model

Place the dataset in the project root:

```text
US_Accidents_March23.csv
```

Then run:

```bash
python train.py
```

The training process performs:

```text
1. Load dataset
2. Clean data
3. Extract time features
4. Build Risk_Class
5. Engineer features
6. Create geographic clusters
7. Perform 5-fold cross-validation
8. Train XGBoost
9. Train LightGBM
10. Train CatBoost
11. Generate OOF predictions
12. Train Logistic Regression meta-learner
13. Calculate stacking ROC-AUC
14. Apply isotonic calibration
15. Find F1-optimal threshold
16. Fit final scaler
17. Save model artifact
```

The final artifact is:

```text
innoverse_gold_model.pkl
```

The training script stores:

```python
{
    'xgb_models': ...,
    'lgb_models': ...,
    'cat_models': ...,
    'kmeans': ...,
    'scaler': ...,
    'meta_learner': ...,
    'calibrator': ...,
    'features': ...,
    'state_map_global': ...,
    'global_mean_all': ...,
    'best_threshold': ...
}
```

---

# 🧪 Running Predictions

After training:

```bash
python test.py
```

The testing script loads the artifact:

```python
artifact = joblib.load('innoverse_gold_model.pkl')
```

and restores the ensemble components.

It then:

1. Processes input data.
2. Applies feature engineering.
3. Scales the features.
4. Generates XGBoost probabilities.
5. Generates LightGBM probabilities.
6. Generates CatBoost probabilities.
7. Passes them to the meta-learner.
8. Calibrates the probability.
9. Applies the optimal threshold.
10. Returns the final risk class.

---

# 🌐 Running the Flask API

Start the backend:

```bash
python app.py
```

The application listens by default on:

```text
http://localhost:8000
```

The application exposes:

```text
GET  /api/health
POST /api/predict
```

---

# ❤️ Health Check API

## Endpoint

```http
GET /api/health
```

Example:

```bash
curl http://localhost:8000/api/health
```

Example response:

```json
{
  "status": "online",
  "service": "INNOVERSE Stacking Engine",
  "model_loaded": true,
  "optimal_threshold": 0.4285
}
```

The exact threshold is loaded from the trained artifact when available.

---

# 🔮 Prediction API

## Endpoint

```http
POST /api/predict
```

Content type:

```text
application/json
```

The API converts the incoming JSON object into a Pandas DataFrame and sends it through the same inference pipeline used by the test system.

---

# 📤 Example API Request

```json
{
  "Start_Lat": 41.8781,
  "Start_Lng": -87.6298,
  "State": "IL",
  "Distance(mi)": 12.5,
  "Hour": 2,
  "DayOfWeek": 4,
  "Month": 1,
  "Temperature(F)": -10.0,
  "Humidity(%)": 98.0,
  "Pressure(in)": 28.10,
  "Visibility(mi)": 0.1,
  "Wind_Speed(mph)": 50.0,
  "Amenity": 0,
  "Bump": 0,
  "Crossing": 1,
  "Give_Way": 0,
  "Junction": 1,
  "No_Exit": 0,
  "Railway": 0,
  "Roundabout": 0,
  "Station": 0,
  "Stop": 0,
  "Traffic_Calming": 0,
  "Traffic_Signal": 1
}
```

---

# 📥 Example API Response

```json
{
  "status": "success",
  "risk_status": "HIGH RISK",
  "calibrated_prob": 0.82,
  "prediction_class": 1,
  "base_models": {
    "xgb_prob": 0.85,
    "lgb_prob": 0.79,
    "cat_prob": 0.81
  },
  "best_threshold": 0.4285
}
```

> The values above are an illustrative response format. Actual probabilities are produced by the trained model and depend on the input and saved artifact.

---

# 🧾 API Output Fields

| Field | Description |
|---|---|
| `status` | API request status |
| `risk_status` | `HIGH RISK` or `LOW RISK` |
| `calibrated_prob` | Final calibrated risk probability |
| `prediction_class` | Binary prediction: `0` or `1` |
| `base_models.xgb_prob` | XGBoost ensemble probability |
| `base_models.lgb_prob` | LightGBM ensemble probability |
| `base_models.cat_prob` | CatBoost ensemble probability |
| `best_threshold` | Learned classification threshold |

---

# 🧪 Testing Scenarios

The included `test.py` contains sample scenarios.

## Scenario 1 — Critical Conditions

The test includes conditions such as:

- Night-time
- Very low visibility
- High wind speed
- Very high humidity
- Low temperature
- Junction
- Traffic signal
- Crossing

These features are intentionally designed to represent a difficult road environment.

---

## Scenario 2 — Lower-risk Conditions

The second scenario represents a more favorable environment with:

- Good visibility
- Mild temperature
- Low humidity
- Low wind speed
- Minimal infrastructure indicators

The actual output should always be taken from the trained model rather than assumed from the scenario description.

---

# 🧩 Input Features

The core model features include:

### Geographic

```text
Start_Lat
Start_Lng
Geo_Cluster
State_Historical_Risk
```

### Distance

```text
Distance(mi)
Log_Distance
```

### Time

```text
Hour
DayOfWeek
Month
Hour_Sin
Hour_Cos
DayOfWeek_Sin
DayOfWeek_Cos
Month_Sin
Month_Cos
Is_Rush_Hour
Is_Weekend
```

### Weather

```text
Temperature(F)
Humidity(%)
Pressure(in)
Visibility(mi)
Wind_Speed(mph)
Wind_Chill_Effect
```

### Road Infrastructure

```text
Amenity
Bump
Crossing
Give_Way
Junction
No_Exit
Railway
Roundabout
Station
Stop
Traffic_Calming
Traffic_Signal
```

### Infrastructure aggregate

```text
Infrastructure_Risk_Score
```

---

# 🔄 Inference Pipeline

For every new prediction:

```text
Raw JSON / DataFrame
        │
        ▼
Feature Engineering
        │
        ├── Cyclical Time Features
        ├── Rush Hour
        ├── Weekend
        ├── Wind Effect
        ├── State Risk
        ├── Geo Cluster
        ├── Log Distance
        └── Infrastructure Score
        │
        ▼
RobustScaler
        │
        ▼
XGBoost Ensemble
        │
        ▼
LightGBM Ensemble
        │
        ▼
CatBoost Ensemble
        │
        ▼
Logistic Regression Meta-Learner
        │
        ▼
Isotonic Calibration
        │
        ▼
Optimal Threshold
        │
        ▼
Risk Class + Probability
```

---

# 💾 Model Artifact

The trained model is stored using Joblib:

```text
innoverse_gold_model.pkl
```

The artifact contains the complete inference state.

This is important because the application does not only need the classifiers. It also needs:

- Geographic clustering model
- Feature scaler
- Meta-learner
- Probability calibrator
- Feature list
- Historical state mapping
- Global risk mean
- Optimal threshold

Therefore, **do not delete or replace individual components of the artifact unless the inference code is updated accordingly.**

---

# 🛡️ Robust Scaling

The project uses:

```python
RobustScaler()
```

This scaling method is less sensitive to extreme values than standard mean/variance scaling.

The final scaler is fitted on the engineered feature matrix and stored inside the artifact.

---

# 🧯 Fallback Engine

The Flask application contains a fallback mechanism.

If the trained `.pkl` artifact cannot be loaded, the application can activate a lightweight fallback prediction engine.

The fallback estimates a probability using manually defined rules involving:

- Visibility
- Temperature
- Wind speed
- Hour
- Junction/intersection indicators
- Distance

This fallback is useful for keeping the API operational during development.

### ⚠️ The fallback is NOT the trained machine-learning model.

For competition demonstrations and scientific evaluation, the real trained artifact should be used.

---

# 🔐 Environment Variables

The application supports a `PORT` environment variable.

Example:

```text
PORT=8000
```

Windows PowerShell:

```powershell
$env:PORT=8000
python app.py
```

Linux/macOS:

```bash
export PORT=8000
python app.py
```

---

# 🧹 Git and Security

Do not commit large datasets, model artifacts, or secret configuration files unless required.

Recommended `.gitignore` entries include:

```text
__pycache__/
*.pyc
.venv/
.env
*.pkl
*.csv
```

If the dataset or trained model is intentionally distributed with the repository, remove the corresponding entries from `.gitignore`.

Never commit real API keys, passwords, tokens, or private credentials.

---

# 📈 Evaluation

The training system reports ROC-AUC for the three base models on each validation fold.

It also calculates a final out-of-fold stacking ROC-AUC:

```text
Total Stacking OOF ROC-AUC Score
```

The decision threshold is selected using the F1 score.

Relevant evaluation concepts are:

- ROC-AUC
- Precision
- Recall
- F1-score
- Probability calibration

### Why ROC-AUC?

ROC-AUC evaluates ranking performance across different thresholds.

### Why F1?

F1 balances precision and recall and is useful when the positive class is important and class distributions are not perfectly balanced.

### Why calibration?

A classifier's raw probability is not necessarily a well-calibrated probability. Isotonic calibration is used to transform the meta-model output into a calibrated score.

---

# ⚠️ Important Scientific Notes

This section is especially important for interpreting the project correctly.

## 1. Risk is a derived target

The current target is constructed using:

```text
Severity
Weather hazard
Accident duration
```

Therefore, the model is learning a definition of risk created from historical accident records.

It is not a pure causal model of why accidents happen.

---

## 2. This is not causal inference

A high predicted probability does not prove that a particular feature caused an accident.

For example:

```text
Traffic_Signal = 1
```

does not mean that traffic signals cause accidents.

A feature may simply be correlated with locations where more accidents are recorded.

---

## 3. It is not a guaranteed prediction

The model produces an estimated probability/class based on learned historical patterns.

It cannot guarantee that an accident will or will not happen.

---

## 4. Geographic risk requires careful interpretation

The geographic cluster represents patterns found in the training data.

It should not automatically be interpreted as a permanent safety classification of a real-world area.

Road design, traffic volume, construction, weather, population, and infrastructure can change over time.

---

# ⚠️ Limitations

Current limitations include:

- The target is derived from accident-record attributes.
- The dataset represents historical observations.
- Traffic volume is not directly included in the current feature set.
- Driver behavior is not directly observed.
- Real-time traffic information is not included.
- Real-time road closures are not included.
- The model does not establish causality.
- Geographic clustering is based on the available historical coordinates.
- Prediction quality depends strongly on the training data distribution.
- A probability should not be interpreted as a guarantee.
- The fallback engine is rule-based and should not be used as a substitute for the trained model.

---

# 🚀 Future Improvements

Potential future versions can improve the system with:

## Real-time traffic

Integrate:

- Traffic density
- Average speed
- Congestion
- Road closures
- Traffic incidents

---

## Real-time weather

Integrate:

- Rain
- Snow
- Fog
- Ice
- Storm conditions
- Weather radar

---

## Better geographic modeling

Possible approaches:

- H3 spatial indexing
- Geospatial neural networks
- Spatial-temporal models
- Road-network graph models

---

## Temporal forecasting

Instead of only classifying historical accident records, a future version could model:

```text
Location + Time + Weather + Traffic
                     │
                     ▼
           Probability of Accident
```

This would require a carefully constructed dataset containing both accident and non-accident observations.

---

## Explainable AI

Future versions could integrate:

```text
SHAP
LIME
Permutation Importance
```

to explain why a particular prediction is high or low.

Example:

```text
Risk Score: 81%

Main contributing factors:
1. Very low visibility
2. High wind speed
3. Night-time
4. Junction
5. Severe weather conditions
```

---

## Web Dashboard

A complete frontend can provide:

- Interactive map
- Risk heatmap
- Location search
- Weather information
- Risk percentage
- Model explanation
- Historical trends
- High-risk region visualization

---

# 🏆 Competition Context

This project was developed for:

**The 5th Edition of the INNOVERSE Invention & Innovation Expo – Online Edition**

The project focuses on applying artificial intelligence and machine learning to road safety and historical accident-risk analysis.

The core innovation is the integration of:

```text
Historical Accident Data
        +
Feature Engineering
        +
Geographical Clustering
        +
XGBoost
        +
LightGBM
        +
CatBoost
        +
Level-2 Stacking
        +
Probability Calibration
        +
REST API
```

This creates a modular AI pipeline that can be connected to a web or mobile interface.

---

# 🧪 Quick Start

The shortest path to run the complete system is:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Put the dataset in the project directory
# US_Accidents_March23.csv

# 3. Train
python train.py

# 4. Test
python test.py

# 5. Start API
python app.py
```

Then open:

```text
http://localhost:8000/api/health
```

---

# 🔌 Example Frontend Integration

A JavaScript frontend can call the Flask API using:

```javascript
async function predictRisk(data) {
    const response = await fetch("http://localhost:8000/api/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    });

    return await response.json();
}
```

Example:

```javascript
const result = await predictRisk({
    Start_Lat: 41.8781,
    Start_Lng: -87.6298,
    State: "IL",
    Distance: 2.5,
    Hour: 18,
    DayOfWeek: 4,
    Month: 1,
    "Temperature(F)": 20,
    "Humidity(%)": 85,
    "Pressure(in)": 29.1,
    "Visibility(mi)": 2,
    "Wind_Speed(mph)": 25,
    Junction: 1,
    Crossing: 1,
    "Traffic_Signal": 1
});

console.log(result);
```

---

# 📌 Recommended Repository Description

You can use this short description for GitHub:

> AI-powered road accident risk assessment system using XGBoost, LightGBM, CatBoost, geographical clustering, stacking, probability calibration, and a Flask REST API, developed for the 5th Edition of the INNOVERSE Invention & Innovation Expo.

---

# 🔑 Keywords

```text
Artificial Intelligence
Machine Learning
Road Safety
Accident Prediction
Risk Prediction
Traffic Safety
Road Accident Analysis
XGBoost
LightGBM
CatBoost
Ensemble Learning
Stacking
Probability Calibration
Isotonic Regression
Geospatial Machine Learning
KMeans
MiniBatchKMeans
Python
Flask
REST API
Data Science
INNOVERSE
```

---

# 📜 License

Add the license appropriate for your project.

For example:

```text
MIT License
```

If the project uses a dataset with separate licensing terms, the dataset license must also be respected.

---

# ⚠️ Disclaimer

This project is an experimental artificial-intelligence and machine-learning research system.

The predictions are based on historical data and the target definition used during training. They should not be treated as guaranteed forecasts, official road-safety ratings, emergency guidance, or professional traffic-engineering advice.

The system should be independently validated before being used in any real-world safety-critical environment.

---

# 👨‍💻 Author

**Shervin Moosavi**

AI & Machine Learning Project

**INNOVERSE – Invention & Innovation Expo**

---

## ⭐ Project Pipeline at a Glance

```text
                ┌─────────────────────────┐
                │ Historical Accident Data│
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ Data Cleaning & Parsing │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ Target Construction     │
                │      Risk_Class          │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ Feature Engineering      │
                │ Time + Weather + Geo     │
                │ Infrastructure           │
                └────────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         ┌─────────┐    ┌─────────┐    ┌─────────┐
         │ XGBoost │    │LightGBM │    │ CatBoost│
         └────┬────┘    └────┬────┘    └────┬────┘
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌─────────────────┐
                    │ Level-2 Stacking│
                    │ Logistic Reg.   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Isotonic        │
                    │ Calibration     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Optimal F1      │
                    │ Threshold       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Risk Probability│
                    │ + Risk Class    │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
               Python Test       Flask REST API
```

---

## 🌟 Final Note

The project demonstrates how multiple machine-learning techniques can be combined into a single end-to-end road-risk assessment system:

**Data → Feature Engineering → Geographic Modeling → Ensemble Learning → Stacking → Calibration → API → Application**

The architecture is designed to be extensible, allowing future versions to incorporate real-time traffic, weather, road-network information, explainable AI, and true accident-occurrence forecasting.

---

# 🇮🇷 مستندات فارسی

## 🚦 معرفی پروژه

**سامانه هوشمند پیش‌بینی ریسک تصادفات رانندگی و شناسایی مناطق پرخطر**

این پروژه برای **پنجمین دوره نمایشگاه اختراعات و نوآوری INNOVERSE – نسخه آنلاین** توسعه داده شده است.

هدف سامانه، تحلیل داده‌های تاریخی تصادفات و استفاده از الگوریتم‌های یادگیری ماشین برای ارزیابی کلاس ریسک یک سناریوی تصادف است.

سامانه اطلاعات مختلفی را بررسی می‌کند، از جمله:

- موقعیت جغرافیایی
- ایالت
- زمان وقوع
- روز هفته
- ماه
- فاصله
- دما
- رطوبت
- فشار هوا
- میزان دید
- سرعت باد
- زیرساخت‌های جاده
- تقاطع‌ها
- چراغ راهنمایی
- راه‌آهن و ایستگاه
- شرایط ساعات پرترافیک
- روزهای تعطیل آخر هفته

---

## 🎯 هدف پروژه

اهداف اصلی پروژه عبارت‌اند از:

1. تحلیل داده‌های تاریخی تصادفات.
2. استخراج الگوهای زمانی، جغرافیایی، آب‌وهوایی و زیرساختی.
3. شناسایی شرایط مرتبط با ریسک بالاتر.
4. ساخت یک سیستم یادگیری ماشین چندمدلی.
5. ترکیب خروجی چند مدل مختلف.
6. کالیبره کردن احتمال نهایی.
7. ارائه مدل از طریق REST API.
8. فراهم کردن زیرساخت برای اتصال به یک وب‌اپلیکیشن یا اپلیکیشن موبایل.

---

# ⭐ قابلیت‌های اصلی

### 🤖 یادگیری ماشین چندمدلی

سه مدل اصلی در پروژه استفاده شده‌اند:

- XGBoost
- LightGBM
- CatBoost

هر سه مدل روی ویژگی‌های مهندسی‌شده آموزش داده می‌شوند و سپس خروجی آن‌ها وارد مرحله Stacking می‌شود.

---

### 🔀 Stacking دو سطحی

معماری سیستم دو سطح دارد.

در سطح اول:

```text
XGBoost
LightGBM
CatBoost
```

احتمال ریسک را تولید می‌کنند.

سپس خروجی این سه مدل به یک مدل سطح دوم داده می‌شود:

```text
Logistic Regression
```

بنابراین:

```text
XGBoost ──┐
          │
LightGBM ─┼──> Meta-Learner ──> Calibration ──> Risk
          │
CatBoost ─┘
```

---

### 🗺️ خوشه‌بندی جغرافیایی

برای پیدا کردن الگوهای جغرافیایی از:

```python
MiniBatchKMeans(n_clusters=50)
```

استفاده شده است.

مختصات:

```text
Start_Lat
Start_Lng
```

به خوشه‌های جغرافیایی تبدیل می‌شوند و ویژگی زیر ساخته می‌شود:

```text
Geo_Cluster
```

---

# 📊 داده مورد استفاده

فایل اصلی داده:

```text
US_Accidents_March23.csv
```

این داده شامل اطلاعات تاریخی تصادفات جاده‌ای در آمریکا است.

اطلاعات مورد استفاده پروژه شامل موارد زیر است:

### اطلاعات تصادف

```text
Severity
Start_Time
End_Time
```

### موقعیت

```text
Start_Lat
Start_Lng
State
```

### آب‌وهوا

```text
Temperature(F)
Humidity(%)
Pressure(in)
Visibility(mi)
Wind_Speed(mph)
```

### زیرساخت

```text
Amenity
Bump
Crossing
Give_Way
Junction
No_Exit
Railway
Roundabout
Station
Stop
Traffic_Calming
Traffic_Signal
```

---

# 🎯 تعریف Risk Class

در نسخه فعلی پروژه، متغیر هدف:

```text
Risk_Class
```

است.

این متغیر به صورت ترکیبی از سه شرط ساخته می‌شود:

```text
Severity >= 3
```

یا:

```text
Weather Hazard > 25
AND
Severity >= 2
```

یا:

```text
Duration_Minutes > 240
```

در نتیجه:

```text
0 = Low Risk Class
1 = High Risk Class
```

### ⚠️ نکته مهم

این Target از اطلاعات خود رکوردهای تصادف ساخته شده است.

بنابراین مدل فعلی از نظر علمی بیشتر یک **سیستم طبقه‌بندی ریسک بر اساس تعریف Target موجود در داده‌های تاریخی** است و نباید آن را بدون توضیح به عنوان پیش‌بینی قطعی وقوع تصادف در آینده معرفی کرد.

برای ساخت یک مدل واقعی «احتمال وقوع تصادف قبل از وقوع»، باید داده‌هایی شامل نمونه‌های تصادف و عدم وقوع تصادف در بازه‌های زمانی و مکانی مشخص طراحی شوند.

---

# 🧩 مهندسی ویژگی‌ها

پروژه چندین ویژگی جدید تولید می‌کند.

## ⏰ ویژگی‌های زمانی

برای ساعت:

```text
Hour_Sin
Hour_Cos
```

برای ماه:

```text
Month_Sin
Month_Cos
```

برای روز هفته:

```text
DayOfWeek_Sin
DayOfWeek_Cos
```

این کار باعث می‌شود مدل ماهیت چرخه‌ای زمان را بهتر یاد بگیرد.

---

## 🚗 ساعت شلوغی

ویژگی:

```text
Is_Rush_Hour
```

برای ساعات پرترافیک روزهای کاری تعریف می‌شود:

```text
07:00
08:00
09:00
16:00
17:00
18:00
```

---

## 📅 آخر هفته

ویژگی:

```text
Is_Weekend
```

برای روزهای شنبه و یکشنبه در تعریف عددی دیتاست فعال می‌شود.

---

## 🌡️ اثر دما و باد

ویژگی:

```text
Wind_Chill_Effect
```

از رابطه زیر ساخته می‌شود:

```text
Temperature - (Wind Speed × 0.7)
```

---

## 📏 تبدیل لگاریتمی فاصله

برای کاهش اثر مقادیر بسیار بزرگ:

```text
Log_Distance = log(1 + Distance)
```

ساخته می‌شود.

---

## 🚦 امتیاز زیرساختی

تمام ویژگی‌های زیرساختی با یکدیگر جمع می‌شوند:

```text
Amenity
Bump
Crossing
Give_Way
Junction
No_Exit
Railway
Roundabout
Station
Stop
Traffic_Calming
Traffic_Signal
```

و ویژگی:

```text
Infrastructure_Risk_Score
```

ساخته می‌شود.

---

# ⚖️ مدیریت عدم تعادل کلاس‌ها

در صورتی که تعداد نمونه‌های کلاس مثبت و منفی برابر نباشد، پروژه وزن کلاس مثبت را محاسبه می‌کند:

```text
Positive Weight =
Negative Samples / Positive Samples
```

سپس این مقدار در مدل‌های XGBoost، LightGBM و CatBoost به عنوان:

```text
scale_pos_weight
```

استفاده می‌شود.

هدف این است که مدل نسبت به کلاس کم‌تعداد حساسیت بیشتری داشته باشد.

---

# 🔬 اعتبارسنجی پنج‌گانه

پروژه از:

```text
StratifiedKFold
```

با تنظیمات:

```text
5 Folds
Shuffle = True
Random State = 42
```

استفاده می‌کند.

در هر Fold:

1. داده آموزش و اعتبارسنجی جدا می‌شود.
2. اطلاعات تاریخی State فقط از بخش آموزش Fold محاسبه می‌شود.
3. مدل XGBoost آموزش داده می‌شود.
4. مدل LightGBM آموزش داده می‌شود.
5. مدل CatBoost آموزش داده می‌شود.
6. پیش‌بینی Out-of-Fold تولید می‌شود.

این پیش‌بینی‌ها برای مرحله Stacking استفاده می‌شوند.

---

# 🧠 مدل سطح دوم

خروجی سه مدل:

```text
OOF XGBoost
OOF LightGBM
OOF CatBoost
```

به شکل زیر ترکیب می‌شوند:

```python
X_meta_train = np.column_stack([
    oof_xgb,
    oof_lgb,
    oof_cat
])
```

سپس:

```python
LogisticRegression()
```

به عنوان Meta-Learner آموزش داده می‌شود.

---

# 🎯 کالیبراسیون احتمال

احتمال تولیدشده توسط Meta-Learner مستقیماً به عنوان احتمال نهایی استفاده نمی‌شود.

ابتدا با:

```python
IsotonicRegression(out_of_bounds='clip')
```

کالیبره می‌شود.

نتیجه:

```text
Calibrated Risk Probability
```

است.

---

# 📈 انتخاب Threshold

به جای استفاده اجباری از:

```text
0.50
```

پروژه بهترین Threshold را بر اساس F1 Score پیدا می‌کند.

سپس:

```text
Probability >= Best Threshold
```

به عنوان:

```text
High Risk
```

در نظر گرفته می‌شود.

---

# 🏗️ معماری کامل پروژه

```text
US Accidents Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Time Processing
        │
        ▼
Risk Class Construction
        │
        ▼
Feature Engineering
        │
        ├───────────────┐
        ▼               ▼
Geographical       Weather/Time
Clustering         Features
        │               │
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │   XGBoost     │
        ├───────────────┤
        │   LightGBM    │
        ├───────────────┤
        │   CatBoost    │
        └───────┬───────┘
                ▼
         OOF Predictions
                │
                ▼
       Logistic Regression
         Meta-Learner
                │
                ▼
      Isotonic Calibration
                │
                ▼
       Optimal F1 Threshold
                │
                ▼
      Final Risk Prediction
                │
          ┌─────┴─────┐
          ▼           ▼
       test.py      Flask API
```

---

# 📁 ساختار پروژه

ساختار پیشنهادی:

```text
INNOVERSE-Road-Risk/
│
├── train.py
├── test.py
├── app.py
│
├── US_Accidents_March23.csv
│
├── innoverse_gold_model.pkl
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

| فایل | کاربرد |
|---|---|
| `train.py` | آموزش کل سیستم |
| `test.py` | تست مدل |
| `app.py` | سرور Flask و API |
| `US_Accidents_March23.csv` | دیتاست تاریخی |
| `innoverse_gold_model.pkl` | مدل آموزش‌دیده |
| `.env.example` | نمونه تنظیمات محیطی |
| `.gitignore` | فایل‌های خارج از Git |
| `README.md` | مستندات پروژه |

---

# ⚙️ نصب پروژه

ابتدا محیط مجازی بسازید.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

سپس:

```bash
pip install -r requirements.txt
```

یا:

```bash
pip install numpy pandas scikit-learn joblib
pip install xgboost lightgbm catboost
pip install flask flask-cors
```

---

# 🚀 آموزش مدل

ابتدا فایل دیتاست را در پوشه پروژه قرار دهید:

```text
US_Accidents_March23.csv
```

سپس:

```bash
python train.py
```

در پایان فایل زیر ساخته می‌شود:

```text
innoverse_gold_model.pkl
```

این فایل تمام اجزای لازم برای Inference را نگهداری می‌کند.

---

# 🧪 اجرای تست

بعد از آموزش:

```bash
python test.py
```

این فایل:

1. مدل‌ها را Load می‌کند.
2. ورودی را پردازش می‌کند.
3. Feature Engineering را اجرا می‌کند.
4. XGBoost را اجرا می‌کند.
5. LightGBM را اجرا می‌کند.
6. CatBoost را اجرا می‌کند.
7. Stacking را اجرا می‌کند.
8. Probability را Calibration می‌کند.
9. Threshold را اعمال می‌کند.
10. نتیجه نهایی را نمایش می‌دهد.

---

# 🌐 اجرای API

برای اجرای Backend:

```bash
python app.py
```

سرور به صورت پیش‌فرض روی:

```text
http://localhost:8000
```

اجرا می‌شود.

---

# ❤️ بررسی وضعیت سرور

Endpoint:

```http
GET /api/health
```

مثال:

```bash
curl http://localhost:8000/api/health
```

نمونه پاسخ:

```json
{
  "status": "online",
  "service": "INNOVERSE Stacking Engine",
  "model_loaded": true,
  "optimal_threshold": 0.4285
}
```

---

# 🔮 API پیش‌بینی ریسک

Endpoint:

```http
POST /api/predict
```

نوع محتوا:

```text
application/json
```

---

# 📤 نمونه درخواست

```json
{
  "Start_Lat": 41.8781,
  "Start_Lng": -87.6298,
  "State": "IL",
  "Distance(mi)": 12.5,
  "Hour": 2,
  "DayOfWeek": 4,
  "Month": 1,
  "Temperature(F)": -10.0,
  "Humidity(%)": 98.0,
  "Pressure(in)": 28.10,
  "Visibility(mi)": 0.1,
  "Wind_Speed(mph)": 50.0,
  "Amenity": 0,
  "Bump": 0,
  "Crossing": 1,
  "Give_Way": 0,
  "Junction": 1,
  "No_Exit": 0,
  "Railway": 0,
  "Roundabout": 0,
  "Station": 0,
  "Stop": 0,
  "Traffic_Calming": 0,
  "Traffic_Signal": 1
}
```

---

# 📥 نمونه پاسخ

```json
{
  "status": "success",
  "risk_status": "HIGH RISK",
  "calibrated_prob": 0.82,
  "prediction_class": 1,
  "base_models": {
    "xgb_prob": 0.85,
    "lgb_prob": 0.79,
    "cat_prob": 0.81
  },
  "best_threshold": 0.4285
}
```

مقادیر بالا فقط نمونه فرمت پاسخ هستند و مقدار واقعی توسط مدل آموزش‌دیده محاسبه می‌شود.

---

# 📊 معنی خروجی‌ها

| خروجی | معنی |
|---|---|
| `status` | وضعیت درخواست API |
| `risk_status` | وضعیت High Risk یا Low Risk |
| `calibrated_prob` | احتمال کالیبره‌شده |
| `prediction_class` | کلاس نهایی 0 یا 1 |
| `xgb_prob` | خروجی XGBoost |
| `lgb_prob` | خروجی LightGBM |
| `cat_prob` | خروجی CatBoost |
| `best_threshold` | آستانه بهینه مدل |

---

# 🧪 سناریوهای آزمایشی

## 🚨 سناریوی پرخطر

در `test.py` یک سناریوی سخت شامل مواردی مانند:

- شب
- دید بسیار کم
- باد شدید
- رطوبت بالا
- دمای پایین
- Junction
- Crossing
- Traffic Signal

قرار داده شده است.

این سناریو برای بررسی رفتار مدل در شرایط دشوار طراحی شده است.

---

## ✅ سناریوی کم‌خطرتر

سناریوی دوم شرایط مطلوب‌تری دارد، مانند:

- دید مناسب
- دمای مناسب
- باد کم
- رطوبت پایین
- زیرساخت‌های خطرساز کمتر

نتیجه واقعی باید از مدل دریافت شود و نباید مقدار آن از قبل فرض شود.

---

# 💾 فایل مدل

مدل با Joblib ذخیره می‌شود:

```text
innoverse_gold_model.pkl
```

این فایل شامل:

```text
XGBoost Models
LightGBM Models
CatBoost Models
KMeans
Scaler
Meta-Learner
Calibrator
Features
State Risk Mapping
Global Mean
Best Threshold
```

است.

بنابراین فقط داشتن مدل‌های XGBoost یا LightGBM کافی نیست و تمام Artifact باید حفظ شود.

---

# 🛡️ سیستم Fallback

در `app.py` یک سیستم جایگزین نیز وجود دارد.

اگر فایل مدل اصلی پیدا نشود یا قابل Load نباشد، Flask می‌تواند وارد حالت Fallback شود.

این حالت بر اساس قواعد ساده‌ای مثل:

```text
Visibility
Temperature
Wind Speed
Hour
Junction
Crossing
Traffic Signal
Distance
```

یک احتمال تقریبی ایجاد می‌کند.

### ⚠️ توجه

Fallback مدل آموزش‌دیده اصلی نیست.

برای ارائه رسمی پروژه، مسابقه و ارزیابی علمی باید از:

```text
innoverse_gold_model.pkl
```

استفاده شود.

---

# 📈 ارزیابی مدل

در زمان آموزش، ROC-AUC مدل‌های پایه محاسبه می‌شود:

```text
XGBoost ROC-AUC
LightGBM ROC-AUC
CatBoost ROC-AUC
```

همچنین ROC-AUC نهایی Stacking به صورت Out-of-Fold محاسبه می‌شود.

برای انتخاب Threshold نیز F1 Score استفاده شده است.

معیارهای مهم:

```text
ROC-AUC
Precision
Recall
F1-Score
Probability Calibration
```

---

# ⚠️ محدودیت‌های علمی

این پروژه محدودیت‌هایی دارد:

- Target از داده‌های تصادف ساخته شده است.
- مدل رابطه علت و معلولی را اثبات نمی‌کند.
- داده‌ها تاریخی هستند.
- ترافیک لحظه‌ای در مدل فعلی وجود ندارد.
- رفتار راننده مستقیماً اندازه‌گیری نمی‌شود.
- بسته بودن جاده‌ها به صورت Real-Time لحاظ نشده است.
- شرایط لحظه‌ای جاده به صورت کامل مدل نشده است.
- خوشه‌بندی جغرافیایی بر اساس داده‌های موجود انجام شده است.
- احتمال خروجی مدل تضمین وقوع یا عدم وقوع تصادف نیست.

---

# 🚀 قابلیت‌های پیشنهادی برای نسخه بعدی

در نسخه‌های آینده می‌توان موارد زیر را اضافه کرد:

### 🚗 داده ترافیک Real-Time

```text
Traffic Density
Average Speed
Congestion
Road Closure
Traffic Incidents
```

### 🌧️ آب‌وهوای لحظه‌ای

```text
Rain
Snow
Fog
Ice
Storm
Weather Radar
```

### 🗺️ تحلیل جغرافیایی پیشرفته

```text
H3
Road Network
Spatial-Temporal Models
Graph Neural Networks
```

### 🧠 Explainable AI

با:

```text
SHAP
LIME
Permutation Importance
```

می‌توان توضیح داد چرا ریسک یک سناریو بالا تشخیص داده شده است.

مثلاً:

```text
Risk: 81%

Main Factors:
1. Very Low Visibility
2. High Wind Speed
3. Night Time
4. Junction
5. Severe Weather
```

---

# 🌐 اتصال به Frontend

یک Frontend می‌تواند از API استفاده کند:

```javascript
async function predictRisk(data) {
    const response = await fetch(
        "http://localhost:8000/api/predict",
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        }
    );

    return await response.json();
}
```

به این ترتیب معماری می‌تواند به شکل زیر باشد:

```text
React / HTML / JavaScript
          │
          ▼
      Flask API
          │
          ▼
   Machine Learning
          │
          ▼
   Risk Prediction
```

---

# 🏆 بخش مسابقه

این پروژه برای:

**The 5th Edition of the INNOVERSE Invention & Innovation Expo – Online Edition**

توسعه داده شده است.

تمرکز اصلی پروژه استفاده از هوش مصنوعی برای تحلیل ریسک تصادفات جاده‌ای و شناسایی الگوهای پرخطر بر اساس داده‌های تاریخی است.

نوآوری اصلی سیستم در ترکیب موارد زیر است:

```text
Historical Accident Data
        +
Feature Engineering
        +
Geographical Clustering
        +
XGBoost
        +
LightGBM
        +
CatBoost
        +
Stacking
        +
Probability Calibration
        +
REST API
```

---

# 📌 توضیح کوتاه برای GitHub

می‌توان برای توضیحات Repository از این متن استفاده کرد:

> یک سامانه هوشمند تحلیل و ارزیابی ریسک تصادفات رانندگی با استفاده از XGBoost، LightGBM، CatBoost، خوشه‌بندی جغرافیایی، Stacking، کالیبراسیون احتمال و Flask REST API که برای پنجمین دوره نمایشگاه اختراعات و نوآوری INNOVERSE توسعه داده شده است.

---

# 🔑 کلیدواژه‌ها

```text
Artificial Intelligence
Machine Learning
Road Safety
Accident Prediction
Risk Prediction
Traffic Safety
XGBoost
LightGBM
CatBoost
Stacking
Ensemble Learning
Isotonic Regression
Geospatial Machine Learning
KMeans
Python
Flask
REST API
INNOVERSE
```

---

# ⚠️ سلب مسئولیت

این پروژه یک سیستم تحقیقاتی و آزمایشی مبتنی بر هوش مصنوعی و یادگیری ماشین است.

خروجی مدل بر اساس داده‌های تاریخی و تعریف Target مورد استفاده در زمان آموزش تولید می‌شود و نباید به عنوان تضمین وقوع یا عدم وقوع تصادف، رتبه‌بندی رسمی ایمنی جاده، توصیه اضطراری یا جایگزین نظر متخصصان ایمنی و مهندسی ترافیک استفاده شود.

قبل از استفاده از سیستم در محیط‌های واقعی و حساس به ایمنی، اعتبارسنجی مستقل و گسترده ضروری است.

---

# 👨‍💻 توسعه‌دهنده

**Shervin Moosavi**

پروژه هوش مصنوعی و یادگیری ماشین

**INNOVERSE – Invention & Innovation Expo**

---

# ⭐ خلاصه نهایی معماری

```text
داده‌های تاریخی تصادفات
          │
          ▼
پاک‌سازی داده
          │
          ▼
استخراج ویژگی‌های زمانی
          │
          ▼
ساخت Risk Class
          │
          ▼
مهندسی ویژگی
          │
          ▼
خوشه‌بندی جغرافیایی
          │
          ▼
┌─────────┬─────────┬─────────┐
│ XGBoost │ LightGBM│ CatBoost│
└────┬────┴────┬────┴────┬────┘
     └─────────┼─────────┘
               ▼
          Stacking
               │
               ▼
       Logistic Regression
               │
               ▼
      Isotonic Calibration
               │
               ▼
       Optimal Threshold
               │
               ▼
       Risk Probability
               │
        ┌──────┴──────┐
        ▼             ▼
      test.py      Flask API
```

---

## 🌟 نتیجه

این پروژه یک Pipeline کامل هوش مصنوعی را از:

**داده → پردازش → مهندسی ویژگی → مدل‌های Ensemble → Stacking → Calibration → پیش‌بینی → API**

پیاده‌سازی می‌کند و می‌تواند در آینده با داده‌های Real-Time، ترافیک، آب‌وهوا، نقشه، Explainable AI و مدل‌های پیشرفته‌تر توسعه داده شود.
