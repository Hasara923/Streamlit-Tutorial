

# Breast Cancer Analysis and Prediction Web App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://app-tutorial-bsqeupdrqbcvyu7j6of4ee.streamlit.app/)

Welcome to the Breast Cancer Analysis and Prediction web app! This interactive application, built with Streamlit, allows users to explore a breast cancer dataset and use machine learning to predict the likelihood of a tumor being malignant or benign.

## Demo

Check out the live app in action here: [Breast Cancer Analysis Web App](https://app-tutorial-bsqeupdrqbcvyu7j6of4ee.streamlit.app/)

## Features

- **Visualize Data:** Dive into the dataset with scatter plots, bar graphs, histograms, and a correlation heatmap to uncover patterns and insights.
- **Explore Statistics:** Get a comprehensive statistical summary of the breast cancer dataset to understand its structure.
- **Predict Cancer Type:** Use a logistic regression model to predict whether a tumor is malignant or benign based on user input.
- **User-Friendly Interface:** Enjoy a simple and interactive interface that makes data exploration and prediction easy.

## Repository Overview

Here's what you'll find in this repository:

- `Main.py` - The main script that powers the Streamlit web app.
- `README.md` - This file, providing an overview and setup instructions.
- `breast-cancer.csv` - The dataset used for analysis and prediction.
- `breast1.png` and `breast2.png` - Images included in the app for visual appeal.
- `requirements.txt` - A list of Python libraries required to run the app.

## Getting Started

To get started with the app locally, follow these steps:

1. **Clone this repository:**
   ```bash
   git clone https://github.com/Hasara923/Streamlit-Tutorial.git
   cd Streamlit-Tutorial
   ```

2. **Install the required libraries:**
   Make sure you have Python installed, and then run:
   ```bash
   pip install -r requirements.txt
   ```

   The app requires the following libraries:
   - `streamlit>=1.0.0`
   - `pandas>=1.3.0`
   - `numpy>=1.21.0`
   - `matplotlib>=3.4.2`
   - `seaborn>=0.11.0`
   - `scikit-learn>=0.24.0`

3. **Launch the Streamlit app:**
   ```bash
   streamlit run Main.py
   ```
   Open your web browser and go to `http://localhost:8501` to see the app in action!

## How to Use the App

1. **Home Page:** Start here to explore the dataset with various visualizations and a correlation heatmap.
2. **Data Exploration:** Use interactive graphs to understand the distribution and relationships between different features.
3. **Prediction Page:** Enter values for various features, and the app will predict whether the breast cancer is malignant or benign.

## Screenshots

![Home page](image.png)

## License

This project is licensed under the MIT License. Feel free to use and modify it as you see fit!

## Acknowledgments

- This project uses the [Breast Cancer Data Set](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset) from kaggle.com.

