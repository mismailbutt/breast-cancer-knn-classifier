# Breast Cancer Classification using KNN 🩺

A KNN-based machine learning model that classifies breast tumors as **malignant** or **benign** using the Breast Cancer Wisconsin dataset.

**Decode Labs — Industrial Training Kit**

## 📌 Overview
Classifies breast cancer tumors as **Malignant** or **Benign** using **K-Nearest Neighbors (KNN)**, based on 30 tumor measurements. Uses the built-in `sklearn` Breast Cancer Wisconsin Dataset — no external file needed.

## 🛠️ Tech Stack
Python 3 · scikit-learn (KNeighborsClassifier, StandardScaler)

## ⚙️ How It Works
1. Load dataset (569 samples, 30 features, 2 classes)
2. Split into 80% train / 20% test (`stratify=y`)
3. Scale features with `StandardScaler` (KNN is distance-based)
4. Train `KNeighborsClassifier(k=5)`
5. Predict & evaluate (accuracy, confusion matrix, classification report)

## ▶️ How to Run
```bash
git clone https://github.com/mismailbutt/breast-cancer-knn-classifier
cd breast-cancer-knn-classifier
pip install scikit-learn
python main.py
```

## 📊 Results
```
Accuracy: 0.956

Confusion Matrix:
[[39  3]
 [ 2 70]]

              precision  recall  f1-score
   malignant      0.95    0.93     0.94
      benign      0.96    0.97     0.97
```
**~95.6% accuracy** — recall matters more than raw accuracy here, since missing a malignant case is riskier than a false alarm.

## 📈 Possible Improvements
- Tune `k` or use `GridSearchCV`
- Compare with Logistic Regression / SVM / Random Forest
- Add cross-validation

## 👤 Author
**Ismail** — BS AI Student, UET Lahore
Intern at Decode labs

---

## 📫 Contact
**LinkedIn:** https://www.linkedin.com/in/mismailbutt

**GitHub:** https://github.com/mismailbutt
