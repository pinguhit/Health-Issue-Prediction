import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 2a: Define column names (from dataset documentation)
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
    'ca', 'thal', 'target'
]

# Step 2b: Load dataset
df = pd.read_csv('processed.cleveland.data', header=None, names=column_names)

# Step 2c: Replace missing values ('?') with NaN and drop or impute
df = df.replace('?', np.nan)
df = df.dropna()  # Simple approach: drop rows with missing data

# Convert columns to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col])

# Step 2d: Convert target to binary
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Step 2e: Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Step 2f: Split into train (60%) and temp (40%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# Step 2g: Split temp into cross-validation (20%) and test (20%)
X_cv, X_test, y_cv, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Step 2h: Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_cv = scaler.transform(X_cv)
X_test = scaler.transform(X_test)

from tensorflow.keras.regularizers import l1

model1 = Sequential([
    Dense(units = 32, activation = 'relu',kernel_regularizer = l1(0.01)),
    Dense(units = 10,activation = 'relu', kernel_regularizer = l1(0.01)),
    Dense(units = 5,activation = 'relu',kernel_regularizer = l1(0.01)),
    Dense(units = 1, activation = 'sigmoid',kernel_regularizer = l1(0.01))
])

# Optional: print dataset sizes
print(f"Train: {X_train.shape}, CV: {X_cv.shape}, Test: {X_test.shape}")

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

model1.compile(optimizer = Adam(learning_rate = 0.01),loss = BinaryCrossentropy(), metrics = [
    'accuracy',
    'Precision',
    'Recall'
]


model1.fit(X_train,y_train, epochs= 100, validation_data = (X_cv,y_cv))
