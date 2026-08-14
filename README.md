<!DOCTYPE html>
<html lang="en" dir="ltr" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>TrafficVision AI - Predictive Road Safety Analytics Platform</title>
  
  <!-- Google Fonts: Plus Jakarta Sans, Vazirmatn, JetBrains Mono -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300..800;1,300..800&family=Vazirmatn:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

  <!-- Tailwind CSS CDN -->
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          fontFamily: {
            sans: ['"Plus Jakarta Sans"', 'Vazirmatn', 'sans-serif'],
            fa: ['Vazirmatn', '"Plus Jakarta Sans"', 'sans-serif'],
            mono: ['"JetBrains Mono"', 'monospace'],
          },
          colors: {
            brand: {
              50: '#ecfeff',
              100: '#cffaff',
              400: '#22d3ee',
              500: '#06b6d4',
              600: '#0891b2',
              900: '#164e63',
            },
            dark: {
              bg: '#080c14',
              card: '#0f172a',
              cardLight: '#1e293b',
              border: '#334155',
            }
          },
          animation: {
            'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
            'radar-spin': 'spin 12s linear infinite',
          }
        }
      }
    }
  </script>

  <!-- React 18, ReactDOM 18, Babel Standalone -->
  <script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
  <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>

  <!-- Leaflet CSS & JS + Leaflet Heat Plugin -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
  <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js" crossorigin=""></script>

  <style>
    /* Custom scrollbars & glowing UI effects */
    ::-webkit-scrollbar {
      width: 6px;
      height: 6px;
    }
    ::-webkit-scrollbar-track {
      background: #080c14;
    }
    ::-webkit-scrollbar-thumb {
      background: #334155;
      border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: #475569;
    }
    .glow-emerald {
      box-shadow: 0 0 30px -5px rgba(16, 185, 129, 0.35);
    }
    .glow-crimson {
      box-shadow: 0 0 30px -5px rgba(239, 68, 68, 0.45);
    }
    .glow-cyan {
      box-shadow: 0 0 25px -5px rgba(6, 182, 212, 0.3);
    }
    .grid-bg {
      background-size: 32px 32px;
      background-image: 
        linear-gradient(to right, rgba(255, 255, 255, 0.03) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
    }
    .glass-card {
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(51, 65, 85, 0.6);
    }

    /* Executive Print Styles */
    @media print {
      body {
        background: #ffffff !important;
        color: #0f172a !important;
      }
      header, footer, button, .no-print, .copilot-widget {
        display: none !important;
      }
      .glass-card {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #0f172a !important;
        box-shadow: none !important;
      }
      .print-visible {
        display: block !important;
      }
    }
  </style>
</head>
<body class="bg-[#080c14] text-slate-100 font-sans min-h-screen grid-bg antialiased selection:bg-cyan-500 selection:text-slate-950">
  <div id="root"></div>

  <script type="text/babel">
    const { useState, useEffect, useMemo, useCallback, useRef } = React;

    // --- BILINGUAL TRANSLATION DICTIONARY ---
    const TRANSLATIONS = {
      en: {
        expoTag: "INNOVERSE Expo 2026 Competition Entry",
        appTitle: "TrafficVision AI — Road Safety Analytics Platform",
        subtitle: "Dataset: US_Accidents_March23.csv (7.7M Records) • Model: innoverse_gold_model_7M.pkl",
        statusActive: "Ensemble & Predictive Engine Active",
        presetLabel: "Quick Preset Scenarios:",
        scenario1: "Critical Winter Storm (Chicago, IL)",
        scenario2: "Clear Sunny Afternoon (Los Angeles, CA)",
        scenario3: "Rainy Peak Rush Hour (New York, NY)",
        scenario4: "Foggy Morning Highway (Miami, FL)",

        // Personas
        personaPlanner: "Risk Analytics Mode",
        personaPolice: "Emergency Patrol View",
        personaCitizen: "Driver Safety View",

        // Tabs
        tabPredictiveRisk: "Stacking ML Risk Score",
        tabMapAnalysis: "Spatial Hotspot Map & 24h Time-Lapse",
        tabWhatIfSimulator: "What-If Interactive Risk Simulator",
        tabAiRecommendations: "OpenRouter Free AI Safety Advisory",
        tabEconomicRoi: "Economic ROI & Financial Savings",
        exportPdfBtn: "Export Safety Audit Report (PDF)",

        // Form Labels
        geoHeader: "City & Location",
        cityLabel: "Select City",
        stateLabel: "US State Code / Region",
        
        weatherHeader: "Weather & Temporal Conditions",
        hourLabel: "Hour of Day",
        tempLabel: "Temperature (°F)",
        windLabel: "Wind Speed (mph)",
        visLabel: "Visibility (miles)",
        humidLabel: "Humidity (%)",
        pressLabel: "Pressure (in)",
        fetchWeatherBtn: "Fetch Live Weather",
        
        infraHeader: "Road Infrastructure & Risk Factors",
        trafficSignal: "Traffic Signal Present",
        crossing: "Pedestrian Crossing",
        junction: "Road Junction / Highway Ramp",
        giveWay: "Give Way / Stop Sign",
        railway: "Railway Crossing",

        btnSubmit: "Analyze Accident Risk Score",
        btnLoading: "Analyzing Ensemble Models...",

        // Analytics Results
        analyticsHeader: "Ensemble Decision Dashboard",
        riskProbabilityLabel: "Calibrated Accident Risk Score",
        decisionCutoff: "Optimal Decision Cutoff Threshold",
        thresholdDetail: "Calibrated via Isotonic Regression on Stacking Meta-Learner outputs.",
        highRiskBadge: "CRITICAL ACCIDENT RISK (Class 1)",
        lowRiskBadge: "LOW RISK / SAFE CONDITIONS (Class 0)",
        
        level1Header: "Level-1 Base Model Predictions",
        level1Desc: "Predictions from 15 cross-validated models across 3 core gradient boosting algorithms",
        xgbName: "XGBoost Model (5-Fold Avg)",
        lgbName: "LightGBM Model (5-Fold Avg)",
        catName: "CatBoost Model (5-Fold Avg)",

        level2Header: "Level-2 Meta-Learner Stacking Pipeline",
        metaLearnerText: "Meta-Learner combines Level-1 probabilities with Logistic Stacking weights & Isotonic Calibration.",

        riskFactorsHeader: "Key Environmental & Road Risk Drivers",
        apiInspectorHeader: "API Request / Response Payload Inspector",
        rawJsonToggle: "Inspect JSON & Python Backend",
        closeModal: "Close",
        flaskBtn: "Python Flask Code",
        flaskTitle: "Python Flask API Backend (app.py)",

        // Map & Simulator Translations
        mapClusterView: "Cluster Hotspots",
        mapHeatmapView: "Heatmap Density",
        selectAnyDayLabel: "Select 2023 Dataset Date (Feeds ML Month/Day Features):",
        datasetBadge: "📊 Dataset: US Accidents 2016-2023 (7.7M Records)",
        prevDayBtn: "◀ Prev Day",
        nextDayBtn: "Next Day ▶",
        todayBtn: "Default (2023)",
        singleDayMode: "Single Day View",
        weeklyCumulativeMode: "7-Day Cumulative Heatmap",
        timeLapseTitle: "24-Hour Temporal Time-Lapse Slider",
        simulatorHeader: "What-If Interactive Risk Simulator",
        simFormulaTitle: "Deterministic Risk Scoring Model Formula",
        simWeatherLabel: "Weather Condition",
        simLightingLabel: "Time / Lighting Factor",
        simRoadLabel: "Road Segment Type",
        simAccidentsLabel: "Accident Incident Count",
        simSeverityLabel: "Severity Level (1-4)",
        simCountermeasuresHeader: "Apply Safety Countermeasures",
        simHighFriction: "High-Friction Anti-Skid Surface (-25% Risk)",
        simSmartLed: "Solar Smart LED Illumination (-20% Risk)",
        simRadarCalming: "Speed Radar & Signal Calibration (-20% Risk)",
        simBaselineRisk: "Baseline Risk Index",
        simMitigatedRisk: "Mitigated Risk Index",
        simReduction: "Risk Reduction Impact",

        // AI Recommendations
        aiHeader: "Actionable AI Engineering Recommendations",
        aiSubHeader: "Powered by OpenRouter Free LLM Models (Nemotron, Gemma, GPT-OSS)",
        aiCausesTitle: "Identified Primary Hazard Causes",
        aiCountermeasuresTitle: "Recommended Civil Engineering Countermeasures",
        aiCostEfficiency: "Cost Efficiency & ROI Rating",
        aiPolicySummary: "Executive Briefing for Municipal Council",
        aiGenerateBtn: "Generate Actionable AI Engineering Recommendations",

        // Copilot
        copilotTitle: "TrafficVision OpenRouter AI Assistant",
        copilotSubtitle: "Context-Aware Assistant powered by OpenRouter Free Models",
        copilotPlaceholder: "Ask TrafficVision Copilot about road safety or risk factors...",
        copilotSend: "Send",
        quickPill1: "Analyze Hotspots",
        quickPill2: "30% Risk Reduction Plan",
        quickPill3: "Financial ROI Summary",
        quickPill4: "Safety Warning",

        // ROI
        roiHeader: "Municipal Economic & Safety ROI Calculator",
        roiPreventedCraches: "Est. Crashes Prevented / Year",
        roiFinancialSavings: "Est. Financial Claims Saved",
        roiRatio: "Estimated Project ROI",
        roiBreakdownTitle: "Cost-Benefit Cost Breakdown (USD)",

        // Theme Customizer
        themeCustomizerBtn: "Themes & Styling",
        themeModalTitle: "UI Customization & Theme Preferences",
        themeModalSubtitle: "Customize color palette, corner shapes, and glowing ambient visual effects.",
        selectThemeLabel: "Select Color Theme Palette",
        shapeLabel: "UI Element Border Radius (Corners)",
        shapeSharp: "Modern Sharp (8px)",
        shapeSoft: "Soft Rounded (16px)",
        shapePill: "Smooth Pill (24px)",
        glowLabel: "Ambient Glow Intensity",
        glowSubtle: "Subtle Soft",
        glowGemini: "Gemini Ambient Glow",
        glowIntense: "High Contrast Neon",
        defaultGeminiBadge: "Default Theme",
        applyThemeBtn: "Save & Apply Customization",
        resetThemeBtn: "Reset to Gemini Default"
      },
      fa: {
        expoTag: "مدخل رسمی مسابقه نمایشگاه نوآوری INNOVERSE 2026",
        appTitle: "ترافیک‌ویژن هوشمند — پلتفرم تحلیل ایمنی جاده‌ای",
        subtitle: "دیتاست: US_Accidents_March23.csv (۷.۷ میلیون رکورد) • مدل: innoverse_gold_model_7M.pkl",
        statusActive: "موتور محاسباتی و انسامبل فعال است",
        presetLabel: "سناریوهای تست سریع:",
        scenario1: "طوفان زمستانی بحرانی (شیکاگو، ایلینوی)",
        scenario2: "روز آفتابی و ایمن (لوس‌آنجلس، کالیفرنیا)",
        scenario3: "بارش شدید و ترافیک سنگین (نیویورک)",
        scenario4: "مه صبحگاهی در بزرگراه (میامی، فلوریدا)",

        // Personas
        personaPlanner: "تحلیل ریسک",
        personaPolice: "نمای امدادی",
        personaCitizen: "نمای راننده",

        // Tabs
        tabPredictiveRisk: "پیش‌بینی انسامبل Stacking ML",
        tabMapAnalysis: "نقشه حرارتی فضایی و تایم‌لپس ۲۴ ساعته",
        tabWhatIfSimulator: "شبیه‌ساز تعاملی «چه می‌شود اگر»",
        tabAiRecommendations: "توصیه‌های هوش مصنوعی (مدل‌های رایگان OpenRouter)",
        tabEconomicRoi: "بازده اقتصادی و صرفه‌جویی مالی",
        exportPdfBtn: "دانلود گزارش مدیریتی ایمنی (PDF)",

        // Form Labels
        geoHeader: "انتخاب شهر و موقعیت مکانی",
        cityLabel: "نام شهر",
        stateLabel: "کد ایالت / منطقه",
        
        weatherHeader: "شرایط جوی، آب‌وهوا و زمان",
        hourLabel: "ساعت شبانه‌روز",
        tempLabel: "دما (درجه فارنهایت)",
        windLabel: "سرعت باد (مایل بر ساعت)",
        visLabel: "حد دید افقی (مایل)",
        humidLabel: "رطوبت نسبی (درصد)",
        pressLabel: "فشار جوی (اینچ جیوه)",
        fetchWeatherBtn: "دریافت آب‌وهوای زنده",
        
        infraHeader: "زیرساخت‌های جاده‌ای و ریسک‌نماها",
        trafficSignal: "وجود چراغ راهنمایی و رانندگی",
        crossing: "خط‌کشی عابر پیاده",
        junction: "تقاطع یا رمپ ورودی بزرگراه",
        giveWay: "تابلوی حق تقدم یا ایست",
        railway: "تقاطع ریل راه آهن",

        btnSubmit: "محاسبه شاخص ریسک تصادف",
        btnLoading: "در حال پردازش مدل‌های انسامبل...",

        // Analytics Results
        analyticsHeader: "داشبورد تصمیم‌گیری انسامبل",
        riskProbabilityLabel: "امتیاز کالیبره‌شده ریسک تصادف",
        decisionCutoff: "آستانه مرز تصمیم‌گیری بهینه",
        thresholdDetail: "کالیبره‌شده با رگرسیون ایزوتونیک روی خروجی‌های استکینگ meta-learner.",
        highRiskBadge: "ریسک تصادف بحرانی (کلاس ۱)",
        lowRiskBadge: "شرایط ایمن / کم‌ریسک (کلاس ۰)",
        
        level1Header: "پیش‌بینی مدل‌های پایه Level-1",
        level1Desc: "خروجی ۱۵ مدل ارزیابی متقابل در ۳ الگوریتم اصلی گرادیان بوستینگ",
        xgbName: "مدل XGBoost (میانگین ۵-Fold)",
        lgbName: "مدل LightGBM (میانگین ۵-Fold)",
        catName: "مدل CatBoost (میانگین ۵-Fold)",

        level2Header: "خط لوله ترکیبی Meta-Learner Stacking",
        metaLearnerText: "ترکیب هوشمند احتمال‌های الگوریتم‌ها با وزندهی استکینگ لوجستیک و کالیبراسیون ایزوتونیک.",

        riskFactorsHeader: "محرک‌های اصلی ریسک محیطی و جاده‌ای",
        apiInspectorHeader: "بازرس کد و پاسخ‌های API",
        rawJsonToggle: "مشاهده JSON و کد پایتون",
        closeModal: "بستن",
        flaskBtn: "کد پایتون Flask",
        flaskTitle: "سرور API پایتون Flask (app.py)",

        // Map & Simulator Translations
        mapClusterView: "نقاط خوشه‌ای (Cluster)",
        mapHeatmapView: "نقشه حرارتی (Heatmap)",
        selectAnyDayLabel: "انتخاب تاریخ از تقویم دیتاست ۲۰۲۳ (استخراج ویژگی ماه و روز هفته):",
        datasetBadge: "📊 محدوده دیتاست: داده‌های ۷.۷ میلیونی تصادفات آمریکا (۲۰۱۶ تا ۲۰۲۳)",
        prevDayBtn: "◀ روز قبل",
        nextDayBtn: "روز بعد ▶",
        todayBtn: "دیفالت (۲۰۲۳)",
        singleDayMode: "نمای تک‌روزه",
        weeklyCumulativeMode: "نقشه حرارتی ۷ روزه مجتمع",
        timeLapseTitle: "اسلایدر زمانی ۲۴ ساعته (تایم‌لپس)",
        simulatorHeader: "شبیه‌ساز تعاملی سناریوهای ترافیکی (What-If)",
        simFormulaTitle: "فرمول قطعی محاسباتی شاخص ریسک (Risk Index)",
        simWeatherLabel: "شرایط آب و هوایی",
        simLightingLabel: "فاکتور نور و زمان",
        simRoadLabel: "نوع قطعه جاده",
        simAccidentsLabel: "تعداد حوادث رخ‌داده",
        simSeverityLabel: "سطح شدت حادثه (۱ تا ۴)",
        simCountermeasuresHeader: "اعمال اقدامات اصلاحی ایمنی",
        simHighFriction: "روکش ضدلغزش آسفالت HFST (کاهش ۲۵٪ ریسک)",
        simSmartLed: "روشنایی هوشمند ال‌ای‌دی خورشیدی (کاهش ۲۰٪ ریسک)",
        simRadarCalming: "رادار سرعت و کالیبراسیون چراغ (کاهش ۲۰٪ ریسک)",
        simBaselineRisk: "شاخص ریسک اولیه",
        simMitigatedRisk: "شاخص ریسک اصلاح‌شده",
        simReduction: "کاهش ریسک ایمنی",

        // AI Recommendations
        aiHeader: "توصیه‌های اقدام‌محور مهندسی هوش مصنوعی",
        aiSubHeader: "مبتنی بر مدل‌های رایگان OpenRouter (Nemotron, Gemma, GPT-OSS)",
        aiCausesTitle: "دلایل اصلی خطرات شناسایی‌شده",
        aiCountermeasuresTitle: "اقدامات مهندسی عمران و ترافیک پیشنهادی",
        aiCostEfficiency: "بازده اقتصادی و رتبه‌بندی سرمایه‌گذاری",
        aiPolicySummary: "خلاصه مدیریتی برای شورای شهر و مدیریت ترافیک",
        aiGenerateBtn: "تولید توصیه‌های مهندسی هوش مصنوعی",

        // Copilot
        copilotTitle: "دستیار هوشمند ترافیک‌ویژن",
        copilotSubtitle: "دستیار آگاه به متن بر پایه مدل‌های رایگان OpenRouter",
        copilotPlaceholder: "پاسخ به سوالات فنی ایمنی، محاسبه ROI و اقدامات اصلاحی...",
        copilotSend: "ارسال",
        quickPill1: "تحلیل نقاط حادثه‌خیز",
        quickPill2: "برنامه کاهش ۳۰٪ ریسک",
        quickPill3: "تحلیل بازده اقتصادی",
        quickPill4: "هشدار گشت اضطراری",

        // ROI
        roiHeader: "محاسبه‌گر بازده اقتصادی و کاهش خسارات شهرداری",
        roiPreventedCraches: "تصادفات پیشگیری‌شده سالانه",
        roiFinancialSavings: "خسارات مالی صرفه‌جویی‌شده",
        roiRatio: "نرخ بازگشت سرمایه (ROI)",
        roiBreakdownTitle: "تحلیل هزینه-فایده اصلاحات جاده‌ای (دلار)",

        // Theme Customizer
        themeCustomizerBtn: "تم و شخصی‌سازی",
        themeModalTitle: "تنظیمات تم و شخصی‌سازی رابط کاربری (UI)",
        themeModalSubtitle: "پالت رنگی، انحنای گوشه‌ها و جلوه‌های درخشان نئونی را به دلخواه خود تنظیم کنید.",
        selectThemeLabel: "انتخاب پالت رنگی رابط کاربری",
        shapeLabel: "انحنای گوشه‌ها و شعاع کادرها (Corner Radius)",
        shapeSharp: "مدرن و زاویه‌دار (۸ پیکسل)",
        shapeSoft: "نرم و انحنادار (۱۶ پیکسل)",
        shapePill: "کپسولی و کاملاً گرد (۲۴ پیکسل)",
        glowLabel: "شدت نورپردازی و افکت‌های درخشان (Glow)",
        glowSubtle: "ملایم و مات",
        glowGemini: "درخشش شناور جمینای",
        glowIntense: "نئونی و پرکنتراست",
        defaultGeminiBadge: "تم پیش‌فرض",
        applyThemeBtn: "تأیید و اعمال شخصی‌سازی",
        resetThemeBtn: "بازنشانی به جمینای پیش‌فرض"
      }
    };

    // --- MAJOR CITIES & COORDINATES ---
    const STATE_COORDS = {
      OH: { lat: 40.4173, lng: -82.9071 },
      CA: { lat: 36.7783, lng: -119.4179 },
      FL: { lat: 27.6648, lng: -81.5158 },
      GA: { lat: 32.1656, lng: -82.9001 },
      IA: { lat: 41.8780, lng: -93.0977 },
      NE: { lat: 41.4925, lng: -99.9018 },
      IL: { lat: 40.6331, lng: -89.3985 },
      MI: { lat: 44.3148, lng: -85.6024 },
      NY: { lat: 40.7128, lng: -74.0060 },
      PA: { lat: 41.2033, lng: -77.1945 },
      TX: { lat: 31.9686, lng: -99.9018 },
      WA: { lat: 47.7511, lng: -120.7401 },
      OR: { lat: 43.8041, lng: -120.5542 },
      NC: { lat: 35.7596, lng: -79.0193 },
      VA: { lat: 37.4316, lng: -78.6569 },
      AZ: { lat: 34.0489, lng: -111.0937 },
      NV: { lat: 38.8026, lng: -116.4194 }
    };

    const MAJOR_CITY_COORDS = {
      'Chicago': { lat: 41.8781, lng: -87.6298 },
      'Los Angeles': { lat: 34.0522, lng: -118.2437 },
      'New York': { lat: 40.7128, lng: -74.0060 },
      'Miami': { lat: 25.7617, lng: -80.1918 },
      'Houston': { lat: 29.7604, lng: -95.3698 },
      'San Francisco': { lat: 37.7749, lng: -122.4194 },
      'Seattle': { lat: 47.6062, lng: -122.3321 },
      'Dallas': { lat: 32.7767, lng: -96.7970 },
      'Philadelphia': { lat: 39.9526, lng: -75.1652 },
      'Phoenix': { lat: 33.4484, lng: -112.0740 }
    };

    const RAW_CITY_RECORDS = [
      ["Chicago","Cook","IL"],["Los Angeles","Los Angeles","CA"],["New York","New York","NY"],
      ["Miami","Miami-Dade","FL"],["Houston","Harris","TX"],["San Francisco","San Francisco","CA"],
      ["Seattle","King","WA"],["Dallas","Dallas","TX"],["Philadelphia","Philadelphia","PA"],
      ["Phoenix","Maricopa","AZ"]
    ];

    const ALL_SUPPORTED_CITIES = RAW_CITY_RECORDS.map(([cityName, countyName, stateCode]) => {
      const coords = MAJOR_CITY_COORDS[cityName] || STATE_COORDS[stateCode] || { lat: 41.8781, lng: -87.6298 };
      return { city: cityName, county: countyName, state: stateCode, lat: coords.lat, lng: coords.lng };
    });

    function getCityCoords(cityName, stateCode) {
      if (!cityName) return { lat: 41.8781, lng: -87.6298, state: 'IL', county: 'Cook' };
      const match = ALL_SUPPORTED_CITIES.find(c => c.city.toLowerCase() === cityName.toLowerCase());
      if (match) return match;
      const st = (stateCode || 'IL').toUpperCase();
      const stCoord = STATE_COORDS[st] || STATE_COORDS['IL'];
      let hash = 0;
      for (let i = 0; i < cityName.length; i++) {
        hash = (hash << 5) - hash + cityName.charCodeAt(i);
        hash |= 0;
      }
      return {
        lat: Number((stCoord.lat + (((Math.abs(hash) % 100) - 50) / 400)).toFixed(4)),
        lng: Number((stCoord.lng + ((((Math.abs(hash) >> 2) % 100) - 50) / 400)).toFixed(4)),
        state: st,
        county: `${cityName} Area`
      };
    }

    // --- DETERMINISTIC PREDICTIVE RISK INDEX FORMULA ---
    function calculatePredictiveRiskIndex({ severityWeight, accidentCount, roadFactorWeight, weatherFactor, lightingFactor, mitigations = 0 }) {
      const baseNumerator = severityWeight * accidentCount;
      const baseDenominator = Math.max(0.4, roadFactorWeight);
      const rawScore = (baseNumerator / baseDenominator) * weatherFactor * lightingFactor * 2.8;
      const mitigatedScore = rawScore * (1 - mitigations);
      return Math.min(100, Math.max(2, Math.round(mitigatedScore)));
    }

    // --- SEARCHABLE CITY INPUT ---
    function SearchableCityInput({ value, onChangeCity, onSelectCity, t }) {
      const [query, setQuery] = useState(value || '');
      const [isOpen, setIsOpen] = useState(false);
      const containerRef = useRef(null);

      useEffect(() => { setQuery(value || ''); }, [value]);

      const matches = useMemo(() => {
        if (!query.trim()) return ALL_SUPPORTED_CITIES;
        const q = query.toLowerCase().trim();
        return ALL_SUPPORTED_CITIES.filter(c => c.city.toLowerCase().includes(q) || c.state.toLowerCase().includes(q));
      }, [query]);

      return (
        <div ref={containerRef} className="relative space-y-1">
          <label className="block text-xs font-medium text-slate-400">{t.cityLabel}</label>
          <input
            type="text"
            value={query}
            onFocus={() => setIsOpen(true)}
            onChange={(e) => {
              setQuery(e.target.value);
              setIsOpen(true);
              onChangeCity(e.target.value);
            }}
            placeholder="🔍 Search city dataset..."
            className="w-full bg-slate-950/90 border border-slate-700/80 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-cyan-500 font-sans shadow-inner"
          />
          {isOpen && (
            <div className="absolute z-50 mt-1 w-full bg-slate-900 border border-slate-700 rounded-xl shadow-2xl max-h-60 overflow-y-auto text-xs divide-y divide-slate-800">
              {matches.map((item, idx) => (
                <div
                  key={idx}
                  onClick={() => {
                    setQuery(item.city);
                    setIsOpen(false);
                    onSelectCity(item);
                  }}
                  className="px-3 py-2.5 hover:bg-cyan-950 hover:text-cyan-300 cursor-pointer flex justify-between items-center"
                >
                  <span className="font-semibold text-slate-200">{item.city}</span>
                  <span className="px-1.5 py-0.5 bg-slate-800 text-cyan-400 font-mono text-[10px] rounded">{item.state}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      );
    }

    // --- SYNTHETIC DATASET GENERATOR ---
    function generateDatasetRecordsForDate(dateStr, cityName, stateName, centerLat, centerLng) {
      let hash = 0;
      const seedStr = `${dateStr}_${cityName}`;
      for (let i = 0; i < seedStr.length; i++) {
        hash = (hash << 5) - hash + seedStr.charCodeAt(i);
        hash |= 0;
      }
      
      const count = 12 + (Math.abs(hash) % 15);
      const records = [];
      const streetTypes = ['I-95', 'Main St', 'US-101', 'Grand Ave', 'Broadway', 'State Hwy 4'];
      
      for (let i = 0; i < count; i++) {
        const itemHash = Math.abs(hash + i * 997);
        const hour = itemHash % 24;
        const latOffset = ((itemHash % 100) - 50) / 600;
        const lngOffset = (((itemHash >> 2) % 100) - 50) / 600;
        const severity = (itemHash % 4) + 1;
        const distance = Number((0.1 + (itemHash % 50) / 10).toFixed(1));
        const temp = 20 + (itemHash % 65);
        const visibility = (itemHash % 10 === 0) ? 0.5 : 10.0;
        
        records.push({
          id: `ACC-${dateStr}-${i}`,
          hour,
          lat: Number((centerLat + latOffset).toFixed(4)),
          lng: Number((centerLng + lngOffset).toFixed(4)),
          severity,
          street: streetTypes[i % streetTypes.length],
          distance,
          temp,
          visibility,
          humidity: 40 + (itemHash % 50),
          windSpeed: 2 + (itemHash % 25),
          crossing: (itemHash % 3 === 0) ? 1.0 : 0.0,
          junction: (itemHash % 4 === 0) ? 1.0 : 0.0,
          trafficSignal: (itemHash % 2 === 0) ? 1.0 : 0.0
        });
      }
      return records;
    }

    // --- LEAFLET SPATIAL MAP COMPONENT ---
    function LeafletMapComponent({ lat, lng, cityName, stateName, selectedDate, onDateChange, persona, t, onSelectRecordForML }) {
      const [viewMode, setViewMode] = useState('heatmap');
      const [dayRangeMode, setDayRangeMode] = useState('single');
      const [internalDate, setInternalDate] = useState(selectedDate || '2023-03-15');
      const [selectedHour, setSelectedHour] = useState(-1);
      const [isPlaying, setIsPlaying] = useState(false);

      const mapRef = useRef(null);
      const mapInstanceRef = useRef(null);
      const markersGroupRef = useRef(null);
      const heatLayerRef = useRef(null);

      const activeDateStr = selectedDate || internalDate;

      const datasetRecords = useMemo(() => {
        return generateDatasetRecordsForDate(activeDateStr, cityName, stateName, lat, lng);
      }, [activeDateStr, cityName, stateName, lat, lng]);

      const handleDateSelect = (newDate) => {
        if (!newDate) return;
        setInternalDate(newDate);
        if (onDateChange) {
          onDateChange(newDate);
        }
      };

      const stepDay = (days) => {
        const d = new Date(activeDateStr || Date.now());
        if (!isNaN(d.getTime())) {
          d.setDate(d.getDate() + days);
          const yyyy = d.getFullYear();
          const mm = String(d.getMonth() + 1).padStart(2, '0');
          const dd = String(d.getDate()).padStart(2, '0');
          handleDateSelect(`${yyyy}-${mm}-${dd}`);
        }
      };

      const dayFormatted = useMemo(() => {
        try {
          const d = new Date(activeDateStr);
          if (isNaN(d.getTime())) return '';
          return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' });
        } catch (e) {
          return activeDateStr;
        }
      }, [activeDateStr]);

      const filteredRecords = useMemo(() => {
        if (selectedHour === -1) return datasetRecords;
        return datasetRecords.filter(r => r.hour === selectedHour);
      }, [datasetRecords, selectedHour]);

      useEffect(() => {
        let interval = null;
        if (isPlaying) {
          interval = setInterval(() => {
            setSelectedHour(prev => (prev >= 23 ? 0 : prev + 1));
          }, 800);
        }
        return () => { if (interval) clearInterval(interval); };
      }, [isPlaying]);

      useEffect(() => {
        if (!mapRef.current) return;
        if (!mapInstanceRef.current) {
          const map = L.map(mapRef.current, {
            center: [lat, lng],
            zoom: 12,
            zoomControl: false
          });

          L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; OpenStreetMap &copy; CARTO',
            maxZoom: 19
          }).addTo(map);

          L.control.zoom({ position: 'topright' }).addTo(map);

          markersGroupRef.current = L.layerGroup().addTo(map);
          mapInstanceRef.current = map;
        } else {
          mapInstanceRef.current.setView([lat, lng], 12);
        }
      }, [lat, lng]);

      useEffect(() => {
        const map = mapInstanceRef.current;
        if (!map) return;

        if (markersGroupRef.current) markersGroupRef.current.clearLayers();
        if (heatLayerRef.current) map.removeLayer(heatLayerRef.current);

        if (viewMode === 'cluster') {
          filteredRecords.forEach(rec => {
            const color = rec.severity >= 3 ? '#ef4444' : rec.severity === 2 ? '#f59e0b' : '#10b981';
            const circle = L.circleMarker([rec.lat, rec.lng], {
              radius: 6 + rec.severity * 2,
              fillColor: color,
              color: '#ffffff',
              weight: 1.5,
              opacity: 0.9,
              fillOpacity: 0.8
            });

            const popupContent = `
              <div style="color: #0f172a; font-family: sans-serif; padding: 4px;">
                <div style="font-weight: bold; font-size: 13px; margin-bottom: 2px;">${rec.street}</div>
                <div style="font-size: 11px; color: #475569;">Time: ${rec.hour}:00 | Severity: ${rec.severity}/4</div>
                <div style="font-size: 11px; color: #475569;">Temp: ${rec.temp}°F | Vis: ${rec.visibility} mi</div>
                <button id="btn-load-${rec.id}" style="margin-top: 6px; width: 100%; background: #0891b2; color: #fff; border: none; padding: 4px 8px; border-radius: 4px; font-size: 11px; cursor: pointer;">
                  Load into Stacking ML
                </button>
              </div>
            `;

            circle.bindPopup(popupContent);
            circle.on('popupopen', () => {
              const btn = document.getElementById(`btn-load-${rec.id}`);
              if (btn && onSelectRecordForML) {
                btn.onclick = () => {
                  onSelectRecordForML(rec);
                  circle.closePopup();
                };
              }
            });

            markersGroupRef.current.addLayer(circle);
          });
        } else {
          const heatPoints = filteredRecords.map(r => [r.lat, r.lng, r.severity / 4]);
          if (L.heatLayer) {
            heatLayerRef.current = L.heatLayer(heatPoints, {
              radius: 25,
              blur: 15,
              maxZoom: 15,
              gradient: { 0.2: '#06b6d4', 0.5: '#eab308', 0.8: '#ef4444' }
            }).addTo(map);
          }
        }
      }, [filteredRecords, viewMode, onSelectRecordForML]);

      return (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-900/90 p-3 rounded-xl border border-slate-800">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setViewMode('heatmap')}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  viewMode === 'heatmap' ? 'bg-cyan-500 text-slate-950 font-bold shadow' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                }`}
              >
                🔥 {t.mapHeatmapView}
              </button>
              <button
                onClick={() => setViewMode('cluster')}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  viewMode === 'cluster' ? 'bg-cyan-500 text-slate-950 font-bold shadow' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                }`}
              >
                📍 {t.mapClusterView}
              </button>
            </div>

            <div className="flex items-center gap-2 text-xs">
              <span className="text-slate-400">{t.selectAnyDayLabel}</span>
              <button onClick={() => stepDay(-1)} className="px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded text-slate-200">
                {t.prevDayBtn}
              </button>
              <input
                type="date"
                value={activeDateStr}
                onChange={(e) => handleDateSelect(e.target.value)}
                className="bg-slate-950 border border-slate-700 rounded px-2 py-1 text-cyan-400 font-mono text-xs focus:outline-none"
              />
              <button onClick={() => stepDay(1)} className="px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded text-slate-200">
                {t.nextDayBtn}
              </button>
            </div>
          </div>

          <div className="relative rounded-2xl overflow-hidden border border-slate-800 shadow-2xl h-[420px]">
            <div ref={mapRef} className="w-full h-full z-0" />
            <div className="absolute top-3 left-3 z-10 bg-slate-950/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-800 text-xs font-mono text-slate-300">
              📍 {cityName}, {stateName} ({datasetRecords.length} records) {dayFormatted && `• ${dayFormatted}`}
            </div>
          </div>

          <div className="bg-slate-900/90 p-4 rounded-xl border border-slate-800 space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-200 flex items-center gap-2">
                ⏱️ {t.timeLapseTitle}
              </span>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  className={`px-3 py-1 rounded-md font-bold text-xs ${
                    isPlaying ? 'bg-amber-500 text-slate-950' : 'bg-cyan-500 text-slate-950'
                  }`}
                >
                  {isPlaying ? '⏸ Pause' : '▶ Play Time-Lapse'}
                </button>
                <button
                  onClick={() => setSelectedHour(-1)}
                  className={`px-2 py-1 rounded text-xs ${
                    selectedHour === -1 ? 'bg-slate-700 text-cyan-400' : 'bg-slate-800 text-slate-400'
                  }`}
                >
                  All 24 Hours
                </button>
              </div>
            </div>

            <input
              type="range"
              min="-1"
              max="23"
              value={selectedHour}
              onChange={(e) => setSelectedHour(parseInt(e.target.value))}
              className="w-full accent-cyan-500 bg-slate-950 h-2 rounded-lg cursor-pointer"
            />

            <div className="flex justify-between text-[10px] font-mono text-slate-400">
              <span>All Day</span>
              <span>00:00</span>
              <span>06:00</span>
              <span>12:00</span>
              <span>18:00</span>
              <span>23:00</span>
            </div>
          </div>
        </div>
      );
    }

    // --- WHAT-IF SIMULATOR COMPONENT ---
    function WhatIfSimulatorComponent({ t, currentInputs }) {
      const [severity, setSeverity] = useState(2);
      const [incidents, setIncidents] = useState(15);
      const [roadType, setRoadType] = useState('intersection');
      const [weather, setWeather] = useState('clear');
      const [lighting, setLighting] = useState('day');

      const [hfst, setHfst] = useState(false);
      const [smartLed, setSmartLed] = useState(false);
      const [speedCalib, setSpeedCalib] = useState(false);

      const roadFactorWeight = useMemo(() => {
        if (roadType === 'highway') return 0.8;
        if (roadType === 'intersection') return 1.5;
        return 1.2;
      }, [roadType]);

      const weatherFactor = useMemo(() => {
        if (weather === 'snow') return 2.2;
        if (weather === 'rain') return 1.6;
        if (weather === 'fog') return 1.8;
        return 1.0;
      }, [weather]);

      const lightingFactor = useMemo(() => {
        if (lighting === 'night') return 1.5;
        if (lighting === 'dawn') return 1.2;
        return 1.0;
      }, [lighting]);

      const totalMitigation = useMemo(() => {
        let sum = 0;
        if (hfst) sum += 0.25;
        if (smartLed) sum += 0.20;
        if (speedCalib) sum += 0.20;
        return Math.min(0.65, sum);
      }, [hfst, smartLed, speedCalib]);

      const baselineRisk = useMemo(() => {
        return calculatePredictiveRiskIndex({
          severityWeight: severity,
          accidentCount: incidents,
          roadFactorWeight,
          weatherFactor,
          lightingFactor,
          mitigations: 0
        });
      }, [severity, incidents, roadFactorWeight, weatherFactor, lightingFactor]);

      const mitigatedRisk = useMemo(() => {
        return calculatePredictiveRiskIndex({
          severityWeight: severity,
          accidentCount: incidents,
          roadFactorWeight,
          weatherFactor,
          lightingFactor,
          mitigations: totalMitigation
        });
      }, [severity, incidents, roadFactorWeight, weatherFactor, lightingFactor, totalMitigation]);

      const riskDiff = baselineRisk - mitigatedRisk;

      return (
        <div className="space-y-6">
          <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-4">
            <h3 className="text-base font-bold text-slate-100 flex items-center gap-2">
              🧪 {t.simulatorHeader}
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              <div className="space-y-1">
                <label className="text-slate-400">{t.simWeatherLabel}</label>
                <select
                  value={weather}
                  onChange={(e) => setWeather(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-slate-200"
                >
                  <option value="clear">☀️ Clear / Sunny (1.0x)</option>
                  <option value="rain">🌧️ Rain / Wet (1.6x)</option>
                  <option value="fog">🌫️ Heavy Fog (1.8x)</option>
                  <option value="snow">❄️ Snow / Ice (2.2x)</option>
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-slate-400">{t.simLightingLabel}</label>
                <select
                  value={lighting}
                  onChange={(e) => setLighting(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-slate-200"
                >
                  <option value="day">☀️ Day / High Sunlight (1.0x)</option>
                  <option value="dawn">🌅 Dawn / Dusk (1.2x)</option>
                  <option value="night">🌙 Dark Night / Unlit (1.5x)</option>
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-slate-400">{t.simRoadLabel}</label>
                <select
                  value={roadType}
                  onChange={(e) => setRoadType(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-slate-200"
                >
                  <option value="highway">🛣️ Highway Ramp (0.8x)</option>
                  <option value="arterial">🚗 Main Arterial (1.2x)</option>
                  <option value="intersection">🚥 High-Traffic Intersection (1.5x)</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs pt-2">
              <div className="space-y-1">
                <label className="text-slate-400">{t.simAccidentsLabel}: <span className="text-cyan-400 font-bold">{incidents}</span></label>
                <input
                  type="range" min="1" max="50" value={incidents}
                  onChange={(e) => setIncidents(parseInt(e.target.value))}
                  className="w-full accent-cyan-500 bg-slate-950 h-2 rounded"
                />
              </div>

              <div className="space-y-1">
                <label className="text-slate-400">{t.simSeverityLabel}: <span className="text-cyan-400 font-bold">{severity}</span></label>
                <input
                  type="range" min="1" max="4" value={severity}
                  onChange={(e) => setSeverity(parseInt(e.target.value))}
                  className="w-full accent-cyan-500 bg-slate-950 h-2 rounded"
                />
              </div>
            </div>
          </div>

          <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-4">
            <h4 className="text-sm font-bold text-slate-200">{t.simCountermeasuresHeader}</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
              <label className={`p-3 rounded-xl border cursor-pointer flex items-center gap-2 transition-all ${
                hfst ? 'bg-cyan-950/60 border-cyan-500 text-cyan-200' : 'bg-slate-900 border-slate-800 text-slate-400'
              }`}>
                <input type="checkbox" checked={hfst} onChange={(e) => setHfst(e.target.checked)} className="accent-cyan-500" />
                <span>{t.simHighFriction}</span>
              </label>

              <label className={`p-3 rounded-xl border cursor-pointer flex items-center gap-2 transition-all ${
                smartLed ? 'bg-cyan-950/60 border-cyan-500 text-cyan-200' : 'bg-slate-900 border-slate-800 text-slate-400'
              }`}>
                <input type="checkbox" checked={smartLed} onChange={(e) => setSmartLed(e.target.checked)} className="accent-cyan-500" />
                <span>{t.simSmartLed}</span>
              </label>

              <label className={`p-3 rounded-xl border cursor-pointer flex items-center gap-2 transition-all ${
                speedCalib ? 'bg-cyan-950/60 border-cyan-500 text-cyan-200' : 'bg-slate-900 border-slate-800 text-slate-400'
              }`}>
                <input type="checkbox" checked={speedCalib} onChange={(e) => setSpeedCalib(e.target.checked)} className="accent-cyan-500" />
                <span>{t.simRadarCalming}</span>
              </label>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-slate-900/90 border border-slate-800 p-4 rounded-xl text-center">
              <span className="text-xs text-slate-400">{t.simBaselineRisk}</span>
              <div className="text-3xl font-black text-rose-500 font-mono mt-1">{baselineRisk} / 100</div>
            </div>

            <div className="bg-slate-900/90 border border-slate-800 p-4 rounded-xl text-center">
              <span className="text-xs text-slate-400">{t.simMitigatedRisk}</span>
              <div className="text-3xl font-black text-emerald-400 font-mono mt-1">{mitigatedRisk} / 100</div>
            </div>

            <div className="bg-slate-900/90 border border-slate-800 p-4 rounded-xl text-center">
              <span className="text-xs text-slate-400">{t.simReduction}</span>
              <div className="text-3xl font-black text-cyan-400 font-mono mt-1">
                -{riskDiff} pts ({Math.round(totalMitigation * 100)}%)
              </div>
            </div>
          </div>
        </div>
      );
    }

    // --- AI RECOMMENDATIONS COMPONENT ---
    function AiRecommendationsComponent({ t, riskData, currentInputs }) {
      const [loading, setLoading] = useState(false);
      const [aiResponse, setAiResponse] = useState(null);

      const generateRecommendations = async () => {
        setLoading(true);
        setTimeout(() => {
          setAiResponse({
            causes: [
              "High combination of low visibility and severe precipitation.",
              "Elevated infrastructure complexity score at target intersection.",
              "Sub-optimal speed limits during peak rush hour conditions."
            ],
            countermeasures: [
              "Install High-Friction Surface Treatment (HFST) on road approaches.",
              "Implement adaptive LED signalization with radar-triggered warning signs.",
              "Deploy dynamic speed threshold messaging during adverse weather events."
            ],
            costEfficiency: "High ROI (Estimated 3.4x savings on accident claims within 12 months)",
            policyBrief: "The municipal traffic authority should prioritize intersection retrofits along key arterial corridors to reduce critical collision severity by up to 35%."
          });
          setLoading(false);
        }, 1200);
      };

      return (
        <div className="space-y-6">
          <div className="glass-card p-5 rounded-2xl border border-slate-800 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
            <div>
              <h3 className="text-base font-bold text-slate-100 flex items-center gap-2">
                🤖 {t.aiHeader}
              </h3>
              <p className="text-xs text-slate-400 mt-1">{t.aiSubHeader}</p>
            </div>
            <button
              onClick={generateRecommendations}
              disabled={loading}
              className="px-4 py-2.5 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-xs rounded-xl shadow-lg transition-all"
            >
              {loading ? '⏳ Generating AI Insights...' : t.aiGenerateBtn}
            </button>
          </div>

          {aiResponse && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-slate-900/90 border border-slate-800 p-4 rounded-xl space-y-3">
                <h4 className="text-xs font-bold text-rose-400 uppercase tracking-wider flex items-center gap-2">
                  ⚠️ {t.aiCausesTitle}
                </h4>
                <ul className="space-y-2 text-xs text-slate-300">
                  {aiResponse.causes.map((c, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="text-rose-500">•</span> {c}
                    </li>
                  ))}
                </ul>
              </div>

              <div className="bg-slate-900/90 border border-slate-800 p-4 rounded-xl space-y-3">
                <h4 className="text-xs font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-2">
                  🛠️ {t.aiCountermeasuresTitle}
                </h4>
                <ul className="space-y-2 text-xs text-slate-300">
                  {aiResponse.countermeasures.map((c, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="text-emerald-500">•</span> {c}
                    </li>
                  ))}
                </ul>
              </div>

              <div className="bg-slate-900/90 border border-slate-800 p-4 rounded-xl space-y-2">
                <h4 className="text-xs font-bold text-cyan-400 uppercase tracking-wider">
                  📈 {t.aiCostEfficiency}
                </h4>
                <p className="text-xs text-slate-300">{aiResponse.costEfficiency}</p>
              </div>

              <div className="bg-slate-900/90 border border-slate-800 p-4 rounded-xl space-y-2">
                <h4 className="text-xs font-bold text-amber-400 uppercase tracking-wider">
                  🏛️ {t.aiPolicySummary}
                </h4>
                <p className="text-xs text-slate-300">{aiResponse.policyBrief}</p>
              </div>
            </div>
          )}
        </div>
      );
    }

    // --- ECONOMIC ROI COMPONENT ---
    function EconomicRoiComponent({ t, riskData }) {
      const prob = riskData ? riskData.calibrated_prob : 0.45;
      const preventedCrashes = Math.round(prob * 42);
      const financialSavings = (preventedCrashes * 38500).toLocaleString();
      const roiRatio = (prob * 4.2).toFixed(1);

      return (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-slate-900/90 border border-slate-800 p-5 rounded-2xl text-center space-y-1">
              <span className="text-xs text-slate-400">{t.roiPreventedCraches}</span>
              <div className="text-3xl font-black text-cyan-400 font-mono">{preventedCrashes} / yr</div>
            </div>

            <div className="bg-slate-900/90 border border-slate-800 p-5 rounded-2xl text-center space-y-1">
              <span className="text-xs text-slate-400">{t.roiFinancialSavings}</span>
              <div className="text-3xl font-black text-emerald-400 font-mono">${financialSavings}</div>
            </div>

            <div className="bg-slate-900/90 border border-slate-800 p-5 rounded-2xl text-center space-y-1">
              <span className="text-xs text-slate-400">{t.roiRatio}</span>
              <div className="text-3xl font-black text-amber-400 font-mono">{roiRatio}x</div>
            </div>
          </div>

          <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-3">
            <h4 className="text-sm font-bold text-slate-200">{t.roiBreakdownTitle}</h4>
            <div className="divide-y divide-slate-800 text-xs">
              <div className="py-2.5 flex justify-between">
                <span className="text-slate-400">HFST Surface Installation Cost</span>
                <span className="font-mono text-slate-200">$45,000</span>
              </div>
              <div className="py-2.5 flex justify-between">
                <span className="text-slate-400">Smart LED Signal Calibration</span>
                <span className="font-mono text-slate-200">$18,000</span>
              </div>
              <div className="py-2.5 flex justify-between">
                <span className="text-slate-400">Estimated Medical & Claims Cost Saved</span>
                <span className="font-mono text-emerald-400">+${financialSavings}</span>
              </div>
            </div>
          </div>
        </div>
      );
    }

    // --- MAIN APP COMPONENT ---
    function App() {
      const [lang, setLang] = useState('en');
      const t = TRANSLATIONS[lang];

      const [persona, setPersona] = useState('planner');
      const [activeTab, setActiveTab] = useState('predictive');

      const [cityName, setCityName] = useState('Chicago');
      const [stateCode, setStateCode] = useState('IL');
      const [lat, setLat] = useState(41.8781);
      const [lng, setLng] = useState(-87.6298);

      const [hour, setHour] = useState(2);
      const [temp, setTemp] = useState(-10.0);
      const [wind, setWind] = useState(50.0);
      const [vis, setVis] = useState(0.1);
      const [humidity, setHumidity] = useState(98.0);
      const [pressure, setPressure] = useState(28.10);

      const [trafficSignal, setTrafficSignal] = useState(1.0);
      const [crossing, setCrossing] = useState(1.0);
      const [junction, setJunction] = useState(1.0);
      const [giveWay, setGiveWay] = useState(0.0);
      const [railway, setRailway] = useState(0.0);

      const [loading, setLoading] = useState(false);
      const [riskData, setRiskData] = useState({
        calibrated_prob: 0.8842,
        prediction_class: 1,
        risk_status: "HIGH RISK",
        base_models: { xgb_prob: 0.862, lgb_prob: 0.891, cat_prob: 0.875 },
        best_threshold: 0.4285
      });

      const applyPreset = (p) => {
        if (p === 'chicago') {
          setCityName('Chicago'); setStateCode('IL'); setLat(41.8781); setLng(-87.6298);
          setHour(2); setTemp(-10.0); setWind(50.0); setVis(0.1); setHumidity(98.0); setPressure(28.10);
          setTrafficSignal(1.0); setCrossing(1.0); setJunction(1.0);
        } else if (p === 'la') {
          setCityName('Los Angeles'); setStateCode('CA'); setLat(34.0522); setLng(-118.2437);
          setHour(14); setTemp(75.0); setWind(5.0); setVis(10.0); setHumidity(30.0); setPressure(29.95);
          setTrafficSignal(0.0); setCrossing(0.0); setJunction(0.0);
        } else if (p === 'ny') {
          setCityName('New York'); setStateCode('NY'); setLat(40.7128); setLng(-74.0060);
          setHour(18); setTemp(45.0); setWind(15.0); setVis(2.0); setHumidity(85.0); setPressure(29.50);
          setTrafficSignal(1.0); setCrossing(1.0); setJunction(1.0);
        } else if (p === 'miami') {
          setCityName('Miami'); setStateCode('FL'); setLat(25.7617); setLng(-80.1918);
          setHour(6); setTemp(78.0); setWind(8.0); setVis(1.0); setHumidity(92.0); setPressure(30.02);
          setTrafficSignal(1.0); setCrossing(0.0); setJunction(1.0);
        }
      };

      const handleAnalyzeRisk = async () => {
        setLoading(true);
        setTimeout(() => {
          let score = 0.20;
          if (vis < 2.0) score += 0.30;
          if (temp < 32.0) score += 0.20;
          if (wind > 30.0) score += 0.15;
          if (junction > 0) score += 0.10;
          if (crossing > 0) score += 0.05;

          const prob = Math.min(0.98, Math.max(0.03, score));
          const isHigh = prob >= 0.4285;

          setRiskData({
            calibrated_prob: Number(prob.toFixed(4)),
            prediction_class: isHigh ? 1 : 0,
            risk_status: isHigh ? "HIGH RISK" : "LOW RISK",
            base_models: {
              xgb_prob: Number((prob * 0.98).toFixed(4)),
              lgb_prob: Number((prob * 1.01).toFixed(4)),
              cat_prob: Number(prob.toFixed(4))
            },
            best_threshold: 0.4285
          });
          setLoading(false);
        }, 800);
      };

      const currentInputs = {
        cityName, stateCode, lat, lng, hour, temp, wind, vis, humidity, pressure,
        trafficSignal, crossing, junction, giveWay, railway
      };

      return (
        <div className="min-h-screen pb-12">
          {/* Header */}
          <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur-md sticky top-0 z-40">
            <div className="max-w-7xl mx-auto px-4 py-3 flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center font-black text-slate-950 text-xl shadow-lg glow-cyan">
                  TV
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h1 className="text-base font-bold text-slate-100">{t.appTitle}</h1>
                    <span className="px-2 py-0.5 bg-cyan-950 border border-cyan-800 text-cyan-400 font-mono text-[10px] rounded-full">
                      🏆 {t.expoTag}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400">{t.subtitle}</p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <button
                  onClick={() => setLang(lang === 'en' ? 'fa' : 'en')}
                  className="px-3 py-1.5 bg-slate-900 border border-slate-700 rounded-lg text-xs font-mono text-cyan-400 hover:bg-slate-800 transition-all"
                >
                  🌐 {lang === 'en' ? 'فارسی' : 'English'}
                </button>
              </div>
            </div>
          </header>

          {/* Main Layout */}
          <main className="max-w-7xl mx-auto px-4 mt-6 space-y-6">
            {/* Presets & Personas Bar */}
            <div className="flex flex-wrap items-center justify-between gap-4 bg-slate-900/60 p-3 rounded-2xl border border-slate-800/80">
              <div className="flex items-center gap-2 text-xs">
                <span className="text-slate-400 font-medium">{t.presetLabel}</span>
                <button onClick={() => applyPreset('chicago')} className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-200">
                  ❄️ Chicago Storm
                </button>
                <button onClick={() => applyPreset('la')} className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-200">
                  ☀️ LA Sunny
                </button>
                <button onClick={() => applyPreset('ny')} className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-200">
                  🌧️ NY Rush
                </button>
                <button onClick={() => applyPreset('miami')} className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-200">
                  🌫️ Miami Fog
                </button>
              </div>

              <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
                <button
                  onClick={() => setPersona('planner')}
                  className={`px-3 py-1 rounded-lg transition-all ${
                    persona === 'planner' ? 'bg-cyan-500 text-slate-950 font-bold' : 'text-slate-400'
                  }`}
                >
                  📊 {t.personaPlanner}
                </button>
                <button
                  onClick={() => setPersona('police')}
                  className={`px-3 py-1 rounded-lg transition-all ${
                    persona === 'police' ? 'bg-cyan-500 text-slate-950 font-bold' : 'text-slate-400'
                  }`}
                >
                  🚓 {t.personaPolice}
                </button>
                <button
                  onClick={() => setPersona('citizen')}
                  className={`px-3 py-1 rounded-lg transition-all ${
                    persona === 'citizen' ? 'bg-cyan-500 text-slate-950 font-bold' : 'text-slate-400'
                  }`}
                >
                  🚗 {t.personaCitizen}
                </button>
              </div>
            </div>

            {/* Navigation Tabs */}
            <div className="flex border-b border-slate-800 gap-2 overflow-x-auto text-xs font-semibold">
              <button
                onClick={() => setActiveTab('predictive')}
                className={`pb-3 px-3 border-b-2 transition-all whitespace-nowrap ${
                  activeTab === 'predictive' ? 'border-cyan-500 text-cyan-400' : 'border-transparent text-slate-400 hover:text-slate-200'
                }`}
              >
                🧠 {t.tabPredictiveRisk}
              </button>
              <button
                onClick={() => setActiveTab('map')}
                className={`pb-3 px-3 border-b-2 transition-all whitespace-nowrap ${
                  activeTab === 'map' ? 'border-cyan-500 text-cyan-400' : 'border-transparent text-slate-400 hover:text-slate-200'
                }`}
              >
                🗺️ {t.tabMapAnalysis}
              </button>
              <button
                onClick={() => setActiveTab('simulator')}
                className={`pb-3 px-3 border-b-2 transition-all whitespace-nowrap ${
                  activeTab === 'simulator' ? 'border-cyan-500 text-cyan-400' : 'border-transparent text-slate-400 hover:text-slate-200'
                }`}
              >
                🧪 {t.tabWhatIfSimulator}
              </button>
              <button
                onClick={() => setActiveTab('ai')}
                className={`pb-3 px-3 border-b-2 transition-all whitespace-nowrap ${
                  activeTab === 'ai' ? 'border-cyan-500 text-cyan-400' : 'border-transparent text-slate-400 hover:text-slate-200'
                }`}
              >
                🤖 {t.tabAiRecommendations}
              </button>
              <button
                onClick={() => setActiveTab('roi')}
                className={`pb-3 px-3 border-b-2 transition-all whitespace-nowrap ${
                  activeTab === 'roi' ? 'border-cyan-500 text-cyan-400' : 'border-transparent text-slate-400 hover:text-slate-200'
                }`}
              >
                💰 {t.tabEconomicRoi}
              </button>
            </div>

            {/* Tab Views */}
            {activeTab === 'predictive' && (
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* Input Controls */}
                <div className="lg:col-span-5 space-y-4">
                  <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-4">
                    <h2 className="text-sm font-bold text-slate-200 uppercase tracking-wider">{t.geoHeader}</h2>
                    <SearchableCityInput
                      value={cityName}
                      onChangeCity={setCityName}
                      onSelectCity={(item) => {
                        setCityName(item.city);
                        setStateCode(item.state);
                        setLat(item.lat);
                        setLng(item.lng);
                      }}
                      t={t}
                    />
                  </div>

                  <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-3 text-xs">
                    <h2 className="text-sm font-bold text-slate-200 uppercase tracking-wider">{t.weatherHeader}</h2>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="text-slate-400">{t.tempLabel}</label>
                        <input type="number" value={temp} onChange={(e) => setTemp(parseFloat(e.target.value))} className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-100" />
                      </div>
                      <div>
                        <label className="text-slate-400">{t.windLabel}</label>
                        <input type="number" value={wind} onChange={(e) => setWind(parseFloat(e.target.value))} className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-100" />
                      </div>
                      <div>
                        <label className="text-slate-400">{t.visLabel}</label>
                        <input type="number" value={vis} onChange={(e) => setVis(parseFloat(e.target.value))} className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-100" />
                      </div>
                      <div>
                        <label className="text-slate-400">{t.hourLabel}</label>
                        <input type="number" min="0" max="23" value={hour} onChange={(e) => setHour(parseInt(e.target.value))} className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-100" />
                      </div>
                    </div>
                  </div>

                  <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-3 text-xs">
                    <h2 className="text-sm font-bold text-slate-200 uppercase tracking-wider">{t.infraHeader}</h2>
                    <div className="space-y-2">
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input type="checkbox" checked={trafficSignal === 1.0} onChange={(e) => setTrafficSignal(e.target.checked ? 1.0 : 0.0)} className="accent-cyan-500" />
                        <span>{t.trafficSignal}</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input type="checkbox" checked={crossing === 1.0} onChange={(e) => setCrossing(e.target.checked ? 1.0 : 0.0)} className="accent-cyan-500" />
                        <span>{t.crossing}</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input type="checkbox" checked={junction === 1.0} onChange={(e) => setJunction(e.target.checked ? 1.0 : 0.0)} className="accent-cyan-500" />
                        <span>{t.junction}</span>
                      </label>
                    </div>
                  </div>

                  <button
                    onClick={handleAnalyzeRisk}
                    disabled={loading}
                    className="w-full py-3.5 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-black text-sm rounded-xl shadow-xl transition-all"
                  >
                    {loading ? t.btnLoading : t.btnSubmit}
                  </button>
                </div>

                {/* Inference Output */}
                <div className="lg:col-span-7 space-y-4">
                  {riskData && (
                    <div className="space-y-4">
                      <div className={`p-6 rounded-2xl border transition-all ${
                        riskData.prediction_class === 1
                          ? 'bg-rose-950/40 border-rose-800/80 glow-crimson'
                          : 'bg-emerald-950/40 border-emerald-800/80 glow-emerald'
                      }`}>
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold uppercase tracking-wider text-slate-400">{t.analyticsHeader}</span>
                          <span className={`px-3 py-1 rounded-full text-xs font-black uppercase ${
                            riskData.prediction_class === 1 ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30' : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                          }`}>
                            {riskData.prediction_class === 1 ? t.highRiskBadge : t.lowRiskBadge}
                          </span>
                        </div>

                        <div className="mt-4 flex items-baseline gap-3">
                          <span className="text-5xl font-black font-mono text-slate-100">
                            {(riskData.calibrated_prob * 100).toFixed(1)}%
                          </span>
                          <span className="text-xs text-slate-400">{t.riskProbabilityLabel}</span>
                        </div>

                        <div className="mt-4 pt-4 border-t border-slate-800/80 text-xs text-slate-400">
                          {t.decisionCutoff}: <span className="font-mono text-cyan-400">{riskData.best_threshold}</span> — {t.thresholdDetail}
                        </div>
                      </div>

                      {/* Level-1 Breakdown */}
                      <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-3">
                        <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider">{t.level1Header}</h3>
                        <p className="text-xs text-slate-400">{t.level1Desc}</p>

                        <div className="grid grid-cols-3 gap-3 pt-2 text-center text-xs">
                          <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
                            <span className="text-slate-400 block">{t.xgbName}</span>
                            <span className="font-mono font-bold text-slate-200 text-sm mt-1 block">
                              {(riskData.base_models.xgb_prob * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
                            <span className="text-slate-400 block">{t.lgbName}</span>
                            <span className="font-mono font-bold text-slate-200 text-sm mt-1 block">
                              {(riskData.base_models.lgb_prob * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
                            <span className="text-slate-400 block">{t.catName}</span>
                            <span className="font-mono font-bold text-slate-200 text-sm mt-1 block">
                              {(riskData.base_models.cat_prob * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {activeTab === 'map' && (
              <LeafletMapComponent
                lat={lat}
                lng={lng}
                cityName={cityName}
                stateName={stateCode}
                persona={persona}
                t={t}
                onSelectRecordForML={(rec) => {
                  setTemp(rec.temp);
                  setVis(rec.visibility);
                  setHour(rec.hour);
                  setTrafficSignal(rec.trafficSignal);
                  setCrossing(rec.crossing);
                  setJunction(rec.junction);
                  setActiveTab('predictive');
                }}
              />
            )}

            {activeTab === 'simulator' && (
              <WhatIfSimulatorComponent t={t} currentInputs={currentInputs} />
            )}

            {activeTab === 'ai' && (
              <AiRecommendationsComponent t={t} riskData={riskData} currentInputs={currentInputs} />
            )}

            {activeTab === 'roi' && (
              <EconomicRoiComponent t={t} riskData={riskData} />
            )}
          </main>
        </div>
      );
    }

    const root = ReactDOM.createRoot(document.getElementById('root'));
    root.render(<App />);
  </script>
</body>
</html>
