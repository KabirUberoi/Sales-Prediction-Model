import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
train_path = 'SP_Train.xlsx'  # Update this path
train_data = pd.read_excel(train_path)

# Handle missing values
train_data['Item_Weight'].fillna(train_data['Item_Weight'].mean(), inplace=True)
train_data['Outlet_Size'].fillna('Missing', inplace=True)

# Encode categorical variables
train_data_encoded = pd.get_dummies(train_data, drop_first=True)

# Separate features and target
X = train_data_encoded.drop(columns=['Item_Outlet_Sales'])
y = np.array(train_data_encoded['Item_Outlet_Sales'])

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=0.85)  # Retain 85% of variance
X_pca = np.array(pca.fit_transform(X_scaled))

print(f"Shape after PCA: {X_pca.shape}")

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting Regressor
gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_regressor.fit(X_train, y_train)

# Predict on validation set
y_pred = gb_regressor.predict(X_val)

# Evaluate the model
mse = mean_squared_error(y_val, y_pred)
print(f"\nGradient Boosting Regressor Mean Squared Error: {mse:.4f}")
