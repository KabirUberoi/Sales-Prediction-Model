Here’s a detailed README for your GitHub repository:

---

# Sales Prediction using Machine Learning Models

This repository contains the implementation of various machine learning models for predicting sales based on a given dataset of product and store attributes. The models were developed and evaluated as part of **Project 7** for the course ELL409.

## Project Overview

The task involves predicting the sales (`Item_Outlet_Sales`) of products in specific retail outlets based on features such as product type, visibility, price, and store details. The dataset includes 8,000 records and requires splitting into training and validation subsets. The project evaluates both standard machine learning models provided by `scikit-learn` and custom implementations from scratch.

### Dataset Features

- **Product Features:** `Item_Weight`, `Item_Fat_Content`, `Item_Visibility`, `Item_Type`, `Item_MRP`
- **Store Features:** `Outlet_Identifier`, `Outlet_Establishment_Year`, `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`
- **Target Variable:** `Item_Outlet_Sales`

The evaluation metric is the **Root Mean Squared Error (RMSE)**.

---

## Models Implemented

### Custom Implementations

1. **Linear Regression** - Implemented by **Shikhar**
2. **Ridge Regression** - Implemented by **Shikhar**
3. **Polynomial Regression** - Implemented by **Shikhar**
4. **Decision Tree Regressor** - Implemented by **Kabir**
5. **Random Forest** - Implemented by **Kabir**
6. **Gradient Boosting** - Implemented by **Kabir**

Each custom model adheres to its theoretical underpinnings and was optimized using hyperparameter tuning.

### `scikit-learn` Models

- Linear Regression
- Ridge Regression
- Polynomial Regression
- Gradient Boosting
- Random Forest
- Other benchmark models such as Lasso, AdaBoost, MLP and Support Vector Regressor.

---

## Files in the Repository

- **`regression.py`**: Custom implementation of regression models.
- **`regression_run.py`**: Script to run and evaluate regression models.
- **`GradientBoosting.ipynb`**: Implementation and evaluation of Gradient Boosting from scratch.
- **`Random_Forest.ipynb`**: Custom implementation of Random Forest.
- **`SP_Train.xlsx`**: Training dataset.
- **`Project_7.pdf`**: Project details and task description.
- **`Project_Report.pdf`**: Final project report containing methodology, results, and conclusions.
  
---

## Contributors

- **Kabir**: Decision Tree, Random Forest, Gradient Boosting
- **Shikhar Gupta**: Linear Regression, Ridge Regression
