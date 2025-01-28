# Linear Regression Model Metrics

This project provides a Streamlit application to analyze a dataset using a Linear Regression model. The app calculates and displays the following metrics:

1. **R² Score** (for training and test sets)
2. **Mean Squared Error (MSE)**
3. **Root Mean Squared Error (RMSE)**
4. Scatter plot visualization of the dataset and regression line.

## How to Run the App

1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <repository-directory>
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run streamlit_model_metrics.py
   ```

## Features

- Upload your CSV dataset.
- Automatic detection of required columns (`YearsExperience` and `Salary`).
- Displays regression metrics and scatter plot visualization.

## Requirements

The app requires Python and the libraries listed in `requirements.txt`. Ensure that your dataset includes `YearsExperience` and `Salary` columns.

## Example Dataset Format

| YearsExperience | Salary |
|-----------------|--------|
| 1.1             | 39343  |
| 1.3             | 46205  |
| 1.5             | 37731  |

## License

This project is open-source and available under the [MIT License](LICENSE).

