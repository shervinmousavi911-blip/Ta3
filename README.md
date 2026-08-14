# 🚦 TrafficVision AI — Predictive Road Accident Risk & Spatial Hotspot Intelligence Platform

<div align="center">

[![INNOVERSE Expo](https://img.shields.io/badge/Competition-5th%20INNOVERSE%20Invention%20%26%20Innovation%20Expo-ff6b6b?style=for-the-badge&logo=target)](#-team--acknowledgements)
[![Developed By](https://img.shields.io/badge/Developed%20By-Team%20ta3-4f46e5?style=for-the-badge&logo=github)](#-team--acknowledgements)
[![Dataset](https://img.shields.io/badge/Dataset-US%20Accidents%20(7.7M%20Records)-10b981?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/datasets/kerynhan/us-accidents-march23)
[![Python](https://img.shields.io/badge/Python-3.10%2B%20%7C%20Flask-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-Express%20%7C%20React-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**An End-to-End Deep Ensemble Machine Learning Platform for Proactive Accident Risk Forecasting, Spatial Hotspot Identification, and Real-Time Transportation Safety Auditing.**

*Submitted to **The 5th Edition of the INNOVERSE Invention & Innovation Expo – Online Edition** by **Team ta3**.*

[Key Features](#-key-features) • [ML Architecture](#-machine-learning-architecture) • [Dataset & kagglehub](#-dataset-acquisition--features) • [System Architecture](#-system-architecture) • [Quick Start](#-installation--quick-start) • [API Reference](#-api-endpoints)

</div>

---

## 📌 Executive Summary

Road traffic accidents represent a massive global public safety and socioeconomic crisis. Traditional traffic management systems remain purely **reactive**—responding to accidents after they happen. 

**TrafficVision AI**, engineered by **Team ta3**, transforms traffic management into a **proactive, data-driven intelligence ecosystem**. Trained and validated on over **7.7 Million historical US accident records across 47 real-world features**, the system leverages an advanced **Level-2 Stacking Ensemble** combining **XGBoost**, **LightGBM**, and **CatBoost** base classifiers calibrated via **Isotonic Regression**. 

Coupled with live weather telemetry from the **US National Oceanic and Atmospheric Administration (NOAA / NWS API)**, TrafficVision AI delivers millisecond-latency risk predictions, identifies high-density accident spatial clusters, and provides urban planners and drivers with actionable safety intelligence.

---

## ✨ Key Features

| Module | Description | Core Capabilities |
| :--- | :--- | :--- |
| 🗺️ **Spatial Hotspot Map & 24h Time-Lapse** | Interactive geospatial visualization of historical accident hotspots across the US. | Heatmap density layers, 24-hour dynamic accident playback, radius proximity search, clustering filters by state/severity. |
| ⚡ **Stacking ML Risk Score Engine** | Production-grade multi-model accident probability engine. | Level-1 base model breakdown (XGBoost, LightGBM, CatBoost), calibrated meta-learner output, optimal cutoff thresholding. |
| 🌦️ **Live NOAA/NWS Weather Telemetry** | Real-time weather data synchronization from US National Weather Service. | Real-time wind speed, temperature, visibility, precipitation, humidity, and atmospheric pressure integration. |
| 🧪 **What-If Risk Simulator** | Interactive counterfactual scenario testing sandbox. | Live sensitivity curves, feature permutation analysis, infrastructure impact testing (junctions, crossings, signals). |
| 🚗 **Trip Safety Advisor** | Dynamic origin-to-destination route risk evaluator. | Segment-by-segment hazard detection, high-risk window warnings, alternative route recommendations. |
| 📊 **47-Column Dataset Profiler** | Comprehensive exploratory data analysis (EDA) interface. | Interactive feature distribution, correlation matrix, missing value diagnostics, categorical encoding inspector. |
| 📑 **Executive Safety Audit Report (PDF)** | Automated comprehensive risk audit generator. | Downloadable PDF report with risk breakdowns, weather summaries, AI recommendations, and mitigation guidelines. |
| 🌐 **Bilingual Support (EN / FA)** | Native full internationalization support. | Complete English & Persian interface with right-to-left (RTL) styling and dynamic typography. |

---

## 🧠 Machine Learning Architecture

The prediction engine utilizes a high-performance **Two-Tier Stacking Ensemble Architecture** designed for high precision and calibrated probability outputs:

```
                      ┌────────────────────────────────────────┐
                      │    Raw Input (Temporal, Spatial,       │
                      │    Weather, Road Infrastructure)       │
                      └──────────────────┬─────────────────────┘
                                         │
                                         ▼
                      ┌────────────────────────────────────────┐
                      │   Feature Engineering & Preprocessing  │
                      │  - Cyclical Trig Encodings (Sin/Cos)   │
                      │  - Historical State Risk Target Map    │
                      │  - Geo-Spatial K-Means Clustering      │
                      │  - Wind Chill & Infrastructure Scores  │
                      └──────────────────┬─────────────────────┘
                                         │
                                         ▼
                      ┌────────────────────────────────────────┐
                      │        Standard Robust Scaler          │
                      └──────────────────┬─────────────────────┘
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         ▼                               ▼                               ▼
┌──────────────────┐           ┌──────────────────┐           ┌──────────────────┐
│  Level-1 Model A │           │  Level-1 Model B │           │  Level-1 Model C │
│     XGBoost      │           │     LightGBM     │           │     CatBoost     │
│   (5-Fold CV)    │           │   (5-Fold CV)    │           │   (5-Fold CV)    │
└────────┬─────────┘           └────────┬─────────┘           └────────┬─────────┘
         │                              │                              │
         │  p(XGB)                      │  p(LGB)                      │  p(CAT)
         └───────────────────────┬──────┴──────────────────────────────┘
                                 │
                                 ▼
                      ┌────────────────────────────────────────┐
                      │       Level-2 Meta-Learner             │
                      │   (Regularized Logistic Regressor)     │
                      └──────────────────┬─────────────────────┘
                                 │
                                 ▼
                      ┌────────────────────────────────────────┐
                      │    Isotonic Probability Calibration    │
                      │     (Threshold Cutoff: τ = 0.4285)     │
                      └──────────────────┬─────────────────────┘
                                         │
                                         ▼
                      ┌────────────────────────────────────────┐
                      │ Output: Calibrated Risk Probability %  │
                      │ + Binary Classification (High/Low)     │
                      └────────────────────────────────────────┘
```

### 🔬 Feature Engineering Highlights
1. **Temporal Cyclical Transformation**: Converts `Hour (0-23)`, `Month (1-12)`, and `DayOfWeek (0-6)` into continuous trigonometric sine and cosine features:
   $$\text{Hour}_{\sin} = \sin\left(\frac{2\pi \cdot \text{Hour}}{24}\right), \quad \text{Hour}_{\cos} = \cos\left(\frac{2\pi \cdot \text{Hour}}{24}\right)$$
2. **Atmospheric Interaction Index**: Wind-Chill compound formula factoring temperature and wind velocity:
   $$\text{Wind\_Chill\_Effect} = \text{Temperature(F)} - 0.7 \times \text{Wind\_Speed(mph)}$$
3. **Infrastructure Composite Risk Score**: Aggregates presence of critical road markers (Traffic Signals, Crossings, Junctions, Roundabouts, Stops, Railways):
   $$\text{Infrastructure\_Risk\_Score} = \sum_{k=1}^{12} \mathbb{I}(\text{Feature}_k = \text{True})$$
4. **Target-Encoded State Baseline**: Smoothed out-of-fold historical accident severity rate mapped per US state.
5. **Geographic Cluster Mapping**: Pre-trained spatial K-Means clustering ($k=15$) mapped from `(Start_Lat, Start_Lng)`.

---

## 📥 Dataset Acquisition & Features

The model is trained and evaluated on the **US Accidents Dataset (March 2023 Update)**, comprising **7.7+ Million records** spanning from February 2016 to March 2023 across 49 US states.

### ⚡ Automated Dataset Download (via `kagglehub`)

You can automatically download the official dataset in Python using `kagglehub`:

```bash
pip install kagglehub
```

```python
import kagglehub

# Download latest version of US Accidents dataset
path = kagglehub.dataset_download("kerynhan/us-accidents-march23")

print("Path to dataset files:", path)
```

The downloaded directory contains `US_Accidents_March23.csv` (approximately 3.0 GB uncompressed, 7,728,394 rows and 46+ features).

### 📋 Key Feature Taxonomy

| Category | Key Variables |
| :--- | :--- |
| **Spatial & Identification** | `ID`, `Severity`, `Start_Time`, `End_Time`, `Start_Lat`, `Start_Lng`, `Distance(mi)`, `City`, `County`, `State`, `Zipcode` |
| **Atmospheric & Climate** | `Temperature(F)`, `Wind_Chill(F)`, `Humidity(%)`, `Pressure(in)`, `Visibility(mi)`, `Wind_Direction`, `Wind_Speed(mph)`, `Precipitation(in)`, `Weather_Condition` |
| **Road Infrastructure Flags** | `Amenity`, `Bump`, `Crossing`, `Give_Way`, `Junction`, `No_Exit`, `Railway`, `Roundabout`, `Station`, `Stop`, `Traffic_Calming`, `Traffic_Signal` |
| **Temporal & Lighting** | `Sunrise_Sunset`, `Civil_Twilight`, `Nautical_Twilight`, `Astronomical_Twilight`, `Is_Rush_Hour`, `Is_Weekend` |

---

## 🏗️ System Architecture

TrafficVision AI is built as an enterprise-grade full-stack hybrid platform with dual fallback microservices:

```
[ Web Browser Client (React 18 + Tailwind CSS + Leaflet.js) ]
                             │
                             │ REST API Calls / JSON
                             ▼
  [ Node.js / Express Gateway (Port 3000) ]
       │                                │
       ├─► Native ML Fallback Engine    ├─► NOAA / NWS Weather Telemetry API
       │                                │
       └─► Proxy Forwarding ────────────┴─► [ Python Flask Microservice (Port 8000) ]
                                                  │
                                                  ├─► XGBoost + LightGBM + CatBoost
                                                  ├─► Meta-Learner & Scalers
                                                  └─► Scikit-Learn Inference Pipeline
```

---

## 🚀 Installation & Quick Start

### 📋 Prerequisites
- **Node.js**: v18.0.0 or higher
- **Python**: 3.10 or higher (optional, for Python Flask backend)
- **Git**

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/trafficvision-ai.git
cd trafficvision-ai
```

### 2️⃣ Install Dependencies

**Node.js dependencies:**
```bash
npm install
```

**Python dependencies (optional for Python Flask service):**
```bash
pip install -r requirements.txt
# Or manually install:
pip install flask flask-cors pandas numpy scikit-learn joblib xgboost lightgbm catboost
```

### 3️⃣ Launch the Application

**Option A — Standard Single-Command Launch (Express + Automatic Python Backend):**
```bash
npm run dev
```
The server will boot on `http://localhost:3000`.

**Option B — Standalone Python Microservice:**
```bash
python app.py
```
Flask runs on `http://localhost:8000`.

---

## 📡 API Endpoints

### 1. Health Check
```http
GET /api/health
```
**Response:**
```json
{
  "status": "healthy",
  "service": "TrafficVision AI Analytics Platform",
  "engine": "Stacking Ensemble (XGBoost + LightGBM + CatBoost)"
}
```

### 2. Accident Risk Prediction
```http
POST /api/predict
Content-Type: application/json
```
**Request Body:**
```json
{
  "State": "CA",
  "City": "Los Angeles",
  "Start_Lat": 34.0522,
  "Start_Lng": -118.2437,
  "Hour": 17,
  "DayOfWeek": 4,
  "Month": 11,
  "Distance(mi)": 0.5,
  "Temperature(F)": 45.0,
  "Wind_Speed(mph)": 18.0,
  "Visibility(mi)": 2.5,
  "Humidity(%)": 85,
  "Pressure(in)": 29.8,
  "Precipitation(in)": 0.4,
  "Weather_Condition": "Heavy Rain",
  "Junction": true,
  "Traffic_Signal": true,
  "Crossing": false,
  "Stop": false
}
```
**Response:**
```json
{
  "status": "success",
  "risk_status": "HIGH RISK",
  "risk_probability": 0.7842,
  "confidence_score": "78.42%",
  "base_models": {
    "xgb_prob": 0.8120,
    "lgb_prob": 0.7750,
    "cat_prob": 0.7656
  },
  "best_threshold": 0.4285,
  "risk_factors": [
    "Adverse Weather / Low Visibility (< 3.0 mi)",
    "Evening Rush Hour Traffic Peak (17:00)",
    "Hazardous Road Complex (Junction / Signal)"
  ]
}
```

### 3. Live NOAA Weather Sync
```http
GET /api/weather/live?lat=34.0522&lng=-118.2437
```

---

## 🛠️ Technology Stack

- **Frontend**: React 18, Tailwind CSS, Lucide Icons, Leaflet.js, React-Leaflet, Recharts, Framer Motion
- **Backend**: Node.js, Express, TypeScript, Python 3.10+, Flask, Flask-CORS
- **Machine Learning**: Scikit-Learn, XGBoost, LightGBM, CatBoost, Joblib, Pandas, NumPy
- **Telemetry & APIs**: NOAA / NWS Weather API, Open-Meteo API
- **Build & Bundler**: Vite, ESBuild, TSX

---

## 👥 Team & Acknowledgements

**Project developed and submitted by:**
### 🏆 Team ta3

Presented at:
**The 5th Edition of the INNOVERSE Invention & Innovation Expo – Online Edition**

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](https://opensource.org/licenses/MIT) page for details.

---

<div align="center">

Made with ❤️ by **Team ta3** for **INNOVERSE 2026**

</div>
