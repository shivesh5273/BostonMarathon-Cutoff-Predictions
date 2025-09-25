# Boston Marathon Cutoff Predictions — 2026 & 2027

## 📌 Overview
The Boston Marathon is one of the most prestigious races in the world, with strict qualifying standards. Meeting the **Boston Qualifier (BQ)** time alone isn’t always enough — runners often need an additional cushion (called the **buffer**) because more people qualify than spots available.

This project applies **data science and machine learning** to forecast the likely cutoff buffers for the **2026** and **2027** Boston Marathons.

---

## 🎯 Objectives
- Predict the acceptance cutoff buffers for 2026 and 2027.
- Build and compare multiple machine learning models (Logistic Regression, SVM, Decision Trees, Random Forest, KNN, etc.).
- Evaluate results with calibration curves, ROC curves, and probability estimates.
- Provide both a **runner-friendly quick forecast** and a **full technical deep dive**.

---

## 📊 Key Findings
- **2026 predicted buffer**: ~6 minutes 36 seconds  
  - Realistic range (±1 SD): **5:34 – 7:39**
- **2027 predicted buffer**: ~7 minutes 04 seconds  
  - Realistic range (±1 SD): **6:01 – 8:06**
- **Change from 2026 → 2027**: About **+27 seconds stricter**

In plain English: Expect a slightly tighter cutoff in 2027. Runners aiming for Boston should plan for **at least ~7 minutes under their BQ standard** to be safe.

---
## Main Code file 
scratch.py
## Testing code file 
BQ_Cutoff_2027Forcast.py
___

## 📑 Reports
- **[Quick Forecast Report (PDF)](reports/BostonMarathon2026And2027Predictions.pdf)** — short, runner-friendly summary.  
- **[Full Technical Report (PDF)](reports/BostonMarathon2026And2027PredictionsReport.pdf)** — detailed machine learning project with code, models, and proofs.

---

## 🛠️ Methods
- **Data preprocessing**: normalization of times, buffer calculation, group encoding.
- **Models tested**: Logistic Regression, Linear SVM, Decision Tree, Random Forest, KNN, Perceptron, Adaline-like.
- **Evaluation metrics**: ROC-AUC, calibration, buffer probability tables.
- **Forecasting**: Linear regression on historical trends for future cutoff prediction.

---

## 🏃 Practical Guidance for Runners
- **Safe Zone**: ≥7 minutes under BQ time → near certainty (esp. for 2027).  
- **Likely In**: 5–7 minutes under → usually enough, but risky in tough years.  
- **Risky Zone**: <3 minutes under → high chance of missing out.  

---

## ⚠️ Limitations
- Predictions do not account for **BAA policy changes, registration numbers, or weather conditions**.  
- COVID-affected years (2021–2023) were excluded.  
- Linear trend models may not capture all real-world complexities.  

---

## 🙌 Acknowledgments
This project was inspired by conversations with my **running club**. I built it partly for **fun and coding practice**, and partly to help runners plan realistic race strategies.  

---

## 🔗 Connect
If you’d like to discuss marathon cutoffs, data science in sports, or ML modeling:  
- 📧 [Your Email or Contact]  
- 💼 [LinkedIn Profile]  
- 🏃 [Running Club Mention, optional]
