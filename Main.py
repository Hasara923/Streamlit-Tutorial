import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as  sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

st.title("Breast Cancer")

st.image("breast1.png",width=500)

st.title("Case study on Breast Canser Dataset")

data=pd.read_csv("breast-cancer.csv")
st.write("shape of a dataset",data.shape)

label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

menu=st.sidebar.radio("Menu",["Home","Prediction Details"])
if menu=="Home":
    st.image("breast2.png",width=500)
    st.header("Tabular Data of a Breast Canser")
    if st.checkbox("Tabular Data"):
       st.table(data.head(50))

    st.header("Statistical summary of a Dataframe")
    if st.checkbox("Statistics"):
        st.table(data.describe())

    if st.header("Correlation Graph"):
       fig,ax=plt.subplots(figsize=(20,15))
       sns.heatmap(data.corr(),annot=True,cmap="coolwarm")
       st.pyplot(fig)

    st.header("Visualizations")

          
    graph = st.selectbox("Choose the type of graph", ["Scatter Plot", "Bar Graph", "Histogram"])

    if graph == "Scatter Plot":
        x_col = st.selectbox("Select x-axis column", data.select_dtypes(include=[np.number]).columns)
        y_col = st.selectbox("Select y-axis column", data.select_dtypes(include=[np.number]).columns)
        hue_col = st.selectbox("Select hue column (optional)", ["None"] + list(data.select_dtypes(include=[object]).columns))
        hue_col = None if hue_col == "None" else hue_col

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=ax)
        ax.set_title(f"Scatter Plot of {x_col} vs {y_col}")
        st.pyplot(fig)

    elif graph == "Bar Graph":
        categorical_columns = data.select_dtypes(include=[object]).columns
        
        if len(categorical_columns) > 0:
            column_to_plot = st.selectbox("Select column to plot", categorical_columns)
            
            if column_to_plot:
                fig, ax = plt.subplots(figsize=(12, 6))
                counts = data[column_to_plot].value_counts().reset_index()
                counts.columns = [column_to_plot, 'Count']

                sns.barplot(x=column_to_plot, y='Count', data=counts, ax=ax)
                ax.set_title(f'Count of Occurrences for {column_to_plot}')
                st.pyplot(fig)
        else:
            st.write("No categorical columns available for plotting.")

    elif graph == "Histogram":
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            column_to_plot = st.selectbox("Select column to plot", numeric_columns)
            
            if column_to_plot:
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.histplot(data[column_to_plot], kde=True, ax=ax)
                ax.set_title(f'Histogram of {column_to_plot}')
                ax.set_xlabel(column_to_plot)
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
        else:
            st.write("No numeric columns available for plotting.")


if menu == "Prediction Details":
    st.title("Breast Cancer Prediction")
    
    # Prepare data for training
    features = [col for col in data.columns if col not in ['id', 'diagnosis']]
    X = data[features]
    y = data['diagnosis']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)
    st.write("Model Accuracy: ", accuracy_score(y_test, y_pred))
    st.write("Classification Report: ")
    st.text(classification_report(y_test, y_pred))

    # User input for prediction
    st.header("Enter Details for Prediction")
    user_input = np.array([st.number_input(f"{feature}", value=float(data[feature].mean())) for feature in features])
    
    # Reshape user input and predict
    user_input = user_input.reshape(1, -1)
    if st.button("Predict Breast Cancer Type"):
        prediction = model.predict(user_input)
        cancer_type = "Malignant" if prediction[0] == 1 else "Benign"
        st.write(f"Predicted Type of Cancer: {cancer_type}")