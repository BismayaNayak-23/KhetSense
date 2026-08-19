import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("data/farm_sensor_data.csv")

# Convert crop into numerical values
data = pd.get_dummies(data, columns=["crop"])

# ---------------- WATER MODEL ----------------

water_features = [
    "soil_moisture",
    "temperature",
    "humidity",
    "rainfall",
    "crop_stage"
]

X_water = data[water_features]
y_water = data["water_required"]

X_train, X_test, y_train, y_test = train_test_split(
    X_water,
    y_water,
    test_size=0.2,
    random_state=42
)

water_model = RandomForestRegressor(
    n_estimators=150,
    random_state=42
)

water_model.fit(X_train, y_train)

water_pred = water_model.predict(X_test)

print("Water Model")
print("MAE:", mean_absolute_error(y_test, water_pred))
print("R2:", r2_score(y_test, water_pred))

joblib.dump(water_model, "models/water_model.pkl")


# ---------------- FERTILIZER MODEL ----------------

fertilizer_features = [
    "nitrogen",
    "phosphorus",
    "potassium",
    "soil_ph",
    "crop_stage"
]

X_fert = data[fertilizer_features]
y_fert = data["fertilizer_required"]

X_train, X_test, y_train, y_test = train_test_split(
    X_fert,
    y_fert,
    test_size=0.2,
    random_state=42
)

fert_model = RandomForestRegressor(
    n_estimators=150,
    random_state=42
)

fert_model.fit(X_train, y_train)

fert_pred = fert_model.predict(X_test)

print("\nFertilizer Model")
print("MAE:", mean_absolute_error(y_test, fert_pred))
print("R2:", r2_score(y_test, fert_pred))

joblib.dump(fert_model, "models/fertilizer_model.pkl")

print("\nModels saved successfully!")