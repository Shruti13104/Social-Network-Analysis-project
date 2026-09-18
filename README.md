# 🚀 InfluenceAI – Influencer Marketing Analysis

InfluenceAI is a **machine learning and data analytics web application** designed to analyze influencer marketing campaign performance.

The application uses campaign and influencer-related data to predict **engagement, reach, and expected sales**, while also providing interactive visualizations to help understand marketing performance.

## 📌 Features

* 📊 Influencer marketing campaign analysis
* 🤖 Machine learning-based predictions
* 📈 Engagement prediction
* 👥 Reach prediction
* 💰 Expected sales prediction
* 🥧 Interactive Pie Chart
* 📊 Histogram visualization
* 📉 Line Chart
* 📋 Campaign performance analysis
* 🎨 User-friendly Streamlit interface
* 📱 Interactive dashboard

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Streamlit**
* **Pickle**

## 📂 Project Structure

```text
InfluenceAI/
│
├── app.py
├── train_model.py
├── influencer_marketing_roi.csv
├── engagement_model.pkl
├── reach_model.pkl
├── sales_model.pkl
├── requirements.txt
└── README.md
```

> The exact `.pkl` model filenames may vary depending on your trained models.

## 📊 Dataset

The project uses an influencer marketing dataset containing information related to campaigns, influencers, engagement, reach, and sales/ROI.

The dataset is processed using **Pandas** before being used for model training and prediction.

## 🤖 Machine Learning

The project applies machine learning models to analyze marketing campaign data and generate predictions.

The general workflow is:

```text
Dataset
   ↓
Data Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Prediction
   ↓
Streamlit Dashboard
```

## 📈 Visualizations

InfluenceAI provides different visualizations for understanding campaign performance, including:

* Pie chart
* Histogram
* Line chart
* Campaign performance graphs
* Prediction-related charts

These visualizations make it easier to identify trends and patterns in influencer marketing data.

## ▶️ Run the Project Locally

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### 2. Open the Project Folder

```bash
cd InfluenceAI
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

## ☁️ Streamlit Deployment

The application can be deployed using **Streamlit Community Cloud**.

Basic deployment steps:

1. Upload the project files to GitHub.
2. Create a Streamlit Community Cloud application.
3. Select your GitHub repository.
4. Select the `main` branch.
5. Set the main file as:

```text
app.py
```

6. Deploy the application.

## 📁 Files Required for Deployment

Make sure the GitHub repository contains the files required by `app.py`, especially:

```text
app.py
requirements.txt
influencer_marketing_roi.csv
*.pkl
```

If your application trains models when deployed, also include the training script and required dataset.

## 🎯 Project Objective

The main objective of InfluenceAI is to use **data analytics and machine learning** to help analyze influencer marketing campaigns and understand how different campaign factors can affect engagement, reach, and sales.

## 🔮 Future Enhancements

* Real-time social media API integration
* Influencer recommendation system
* Sentiment analysis
* Campaign ROI optimization
* Advanced ML models
* Automated campaign reports
* Interactive influencer comparison
* AI-powered marketing recommendations

