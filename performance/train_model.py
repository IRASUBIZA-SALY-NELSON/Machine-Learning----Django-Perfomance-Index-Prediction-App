import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

def train():
    """Train a Random Forest model and save it as a .pkl file"""
    # Load dataset
    df = pd.read_csv('StudentPerformance.csv')

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Encode categorical variable
    df['Extracurricular Activities'] = LabelEncoder().fit_transform(
        df['Extracurricular Activities']
    )

    # Prepare features and target
    X = df.drop('Performance Index', axis=1)
    y = df['Performance Index']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Performance:")
    print(f"  MSE: {mse:.2f}")
    print(f"  R² Score: {r2:.4f}")

    # Save model
    joblib.dump(model, 'performance/model.pkl')
    print("Model trained and saved to performance/model.pkl")
