# Housing-Price-Prediction-Using-Linear-Regression

This project is part of **Task 3** of my **AI & ML Internship at Elevate Labs**.  
It focuses on using **linear regression** to predict housing prices based on various features.


## Objective

- Implement **simple and multiple linear regression** on a real-world dataset.
- Evaluate the model using **MAE, MSE, and R² score**.
- Interpret the importance of different features by analyzing model coefficients.
- Visualize the model’s predictions vs actual prices.


## Dataset

- **File:** `Housing.csv`
- Contains features such as:
  - `area`, `bedrooms`, `bathrooms`, `stories`, `parking`
  - `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `furnishingstatus`
- **Target:** `price` (the house price)


## Tools & Libraries Used

- `Pandas` – data manipulation
- `Matplotlib` & `Seaborn` – data visualization
- `Scikit-learn` – model building and evaluation


## Steps Performed

1. **Data Loading & Exploration**
   - Loaded the dataset using `pandas`.
   - Used `.info()` and `.describe()` to understand data types and ranges.

2. **Data Preprocessing**
   - Converted categorical columns to numerical using `pd.get_dummies()`.

3. **Train-Test Split**
   - Split data into training (80%) and testing (20%) using `train_test_split`.

4. **Model Training**
   - Used `LinearRegression()` from `sklearn` to train the model.

5. **Evaluation**
   - Calculated:
     - **Mean Absolute Error (MAE)**
     - **Mean Squared Error (MSE)**
     - **R² score**

6. **Feature Importance**
   - Printed model coefficients to understand the impact of each feature on the house price.

7. **Visualization**
   - Plotted **actual vs predicted prices** to see how well the model performs.


## Sample Output

-**MAE**: 112000.45
-**MSE**: 2.08e+10
-**R²**: 0.69

## Feature  Coefficient
      
0 area 134.89, 
1 bedrooms 49000.23

*(values vary depending on the data split)*


## Key Insights

- **Area** and **number of bedrooms** have the strongest positive impact on price.
- Houses with **guestrooms**, **air conditioning**, and **close to main roads** tend to be more expensive.
- The model captures around **68-70% variance** in prices.


## Project Structure

📁 Housing-Price-Prediction-Using-Linear-Regression
│
-├── Housing.csv # Dataset
-├── Task3.py # Python script with full implementation
-├── README.md # This project overview file


## Author

- **Yogesh Rajput**






