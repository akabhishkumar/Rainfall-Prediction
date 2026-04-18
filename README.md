# Rainfall-Prediction
# 🌦️ Rain Prediction Web App

This project is a **Machine Learning + Web Application** that predicts whether it will rain tomorrow based on weather conditions.

It uses a **Logistic Regression model** trained on the Australian weather dataset and is deployed using a simple **Flask web app**.

---

## 🚀 Features

* Predicts **Rain Tomorrow (Yes/No)**
* Uses real weather features like temperature, humidity, rainfall
* Clean and simple web interface
* End-to-end pipeline:

  * Data preprocessing
  * Model training
  * Web deployment

---

## 📂 Project Structure

```
project/
│
├── app.py                  # Flask backend
├── aussie_rain.joblib     # Saved ML model
├── LogisticRegression.ipynb  # Training notebook
├── weatherAUS.csv         # Dataset
│
└── templates/
    └── index.html         # Frontend UI
```

---

## 🧠 Machine Learning Details

* Model: Logistic Regression
* Target: `RainTomorrow`
* Encoding: Yes → 1, No → 0
* Preprocessing:

  * Missing values → Imputer
  * Scaling → MinMaxScaler
  * Encoding → OneHotEncoder

---

## ⚙️ Requirements

Install dependencies using:

```
pip install -r requirements.txt
```

### 📦 Required Libraries

* Flask
* pandas
* numpy
* scikit-learn
* joblib

---

## 📄 requirements.txt

```
flask
pandas
numpy
scikit-learn
joblib
```

---

## ▶️ How to Run the Project

### 1️⃣ Clone or download the project

```
git clone <your-repo-link>
cd project
```

---

### 2️⃣ (Optional) Create virtual environment

#### Windows:

```
python -m venv venv
venv\Scripts\activate
```

#### Mac/Linux:

```
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Run Flask app

```
python app.py
```

---

### 5️⃣ Open in browser

```
http://127.0.0.1:5000/
```

---

## 🖥️ Usage

1. Enter weather details:

   * Temperature
   * Humidity
   * Rainfall
   * RainToday
2. Click **Predict**
3. Get result:

   * 🌧️ Rain Tomorrow
   * ☀️ No Rain

---

## ⚠️ Important Notes

* Make sure `index.html` is inside the **templates/** folder
* Model and preprocessing must match the features used in the form
* Do not change feature names unless retraining the model

---

## 🔧 Future Improvements

* Better UI (Bootstrap / Tailwind)
* Add more features (Location, Wind, Pressure)
* Deploy online (Render / Railway / AWS)
* Convert to Streamlit app

---

## 👨‍💻 Author

Abhishek Kumar

---

## 📌 Summary

This project demonstrates:

* End-to-end ML pipeline
* Model deployment using Flask
* Real-world dataset handling

---

⭐ If you like this project, consider giving it a star!
