import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class DataProcessor:
    def __init__(self, file_path="heart.csv"):
        self.df = pd.read_csv(file_path)
        self.scaler = StandardScaler()
        self.encoder = None
        self.preprocessor = None

    def preprocess(self):
        # Convert target column to binary
        self.df["HeartDisease"] = self.df["HeartDisease"].map({"Yes": 1, "No": 0})
        y = self.df["HeartDisease"]
        X = self.df.drop(columns=["HeartDisease", "PhysicalHealth", "MentalHealth"])  # Removed

        # Separate numerical and categorical columns
        numeric_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
        categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

        # Define column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", self.scaler, numeric_features),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
            ]
        )

        # Fit and transform the data
        X_processed = self.preprocessor.fit_transform(X)

        # Save the preprocessor pipeline
        joblib.dump(self.preprocessor, "preprocessor.pkl")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

        # Save processed dataset
        joblib.dump((X_train, X_test, y_train, y_test), "dataset.pkl")

        return X_train, X_test, y_train, y_test

# Run the data preprocessing and save artifacts
if __name__ == "__main__":
    processor = DataProcessor()
    X_train, X_test, y_train, y_test = processor.preprocess()