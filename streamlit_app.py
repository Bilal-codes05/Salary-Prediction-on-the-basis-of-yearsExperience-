import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Title
st.title("Linear Regression Model Metrics")

# Upload dataset
file_path="Salary_dataset.csv"
df = pd.read_csv(file_path)
st.write("### Dataset Preview")
st.write(df.head())

    # Assuming the dataset has 'YearsExperience' and 'Salary' columns
if 'YearsExperience' in df.columns and 'Salary' in df.columns:
        X = df[['YearsExperience']]
        y = df['Salary']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)

        # Display metrics
        st.write("### Model Metrics")
        st.write(f"**R² Score (Train):** {train_r2:.2f}")
        st.write(f"**R² Score (Test):** {test_r2:.2f}")
        st.write(f"**MSE (Train):** {train_mse:.2f}")
        st.write(f"**MSE (Test):** {test_mse:.2f}")
        st.write(f"**RMSE (Train):** {train_rmse:.2f}")
        st.write(f"**RMSE (Test):** {test_rmse:.2f}")

        # Scatter plot
        st.write("### Scatter Plot")
        fig, ax = plt.subplots()
        ax.scatter(X['YearsExperience'], y, color='blue', label='Salary')
        ax.plot(X['YearsExperience'], model.predict(X), color='red', label='Regression Line')
        ax.set_title('YearsExperience vs Salary')
        ax.set_xlabel('Years of Experience')
        ax.set_ylabel('Salary')
        ax.legend()
        st.pyplot(fig)

        # Salary Prediction
        st.write("### Predict Salary")
        years_experience = st.number_input("Enter Years of Experience:", min_value=0.0, step=0.1)
        if st.button("Predict"):
            predicted_salary = model.predict([[years_experience]])[0]
            st.write(f"**Predicted Salary:** ${predicted_salary:.2f}")
else:
        st.error("Dataset must contain 'YearsExperience' and 'Salary' columns.")
