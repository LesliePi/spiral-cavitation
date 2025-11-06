import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

class CavitationRegressor:
    def __init__(self, data_path="data/training_data.csv"):
        self.data_path = data_path
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        df = pd.read_csv(self.data_path)
        X = df[['α', 'ω', 'ρ', 'μ', 'σ']]
        y = df['ω_c']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def train(self):
        self.model = GradientBoostingRegressor(n_estimators=200, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return {"MAE": mae, "R²": r2}

    def predict(self, α, ω, ρ, μ, σ):
        X_new = pd.DataFrame([{
            'α': α,
            'ω': ω,
            'ρ': ρ,
            'μ': μ,
            'σ': σ
        }])
        return self.model.predict(X_new)[0]
from src.ml_model import CavitationRegressor

reg = CavitationRegressor()
reg.load_data()
reg.train()
metrics = reg.evaluate()
print(f"MAE: {metrics['MAE']:.4f}, R²: {metrics['R²']:.4f}")

ω_c_pred = reg.predict(α=30, ω=500, ρ=998, μ=0.001, σ=0.0728)
print(f"Predicted ω_c: {ω_c_pred:.2f} rad/s")
