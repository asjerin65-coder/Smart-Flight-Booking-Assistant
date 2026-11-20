# Import required libraries
import sys
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# Compatibility shim for scikit-learn unpickling of ColumnTransformer
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass

# Create FastAPI application
app = FastAPI(
    title="Smart Flight Booking Assistant",
    description="Flight Price Prediction API with Original Random Forest Model",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained Random Forest pipeline
MODEL_PATH = Path(__file__).resolve().parent / "flight_price_rf_pipeline.pkl"
model = joblib.load(MODEL_PATH)
print("Random Forest model loaded successfully from:", MODEL_PATH)

# Test-set Mean Absolute Error baseline
TEST_MAE = 2748.92

# Inspect Random Forest pipeline
print("\n========== MODEL INSPECTION ==========")
print("Model type:", type(model))

if hasattr(model, "named_steps"):
    print("\nPipeline steps:")
    for name, step in model.named_steps.items():
        print(name, ":", type(step))
    rf_model = model.named_steps.get("model", model.steps[-1][1])
    if hasattr(rf_model, "feature_importances_"):
        print("\nTop 15 Feature Importances:")
        importances = rf_model.feature_importances_
        try:
            feature_names = model[:-1].get_feature_names_out()
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            })
            print(
                importance_df.sort_values(
                    by="Importance",
                    ascending=False
                ).head(15).to_string(index=False)
            )
        except Exception as e:
            print("Feature name extraction error:", e)

print("======================================\n")


# Define input data structure matching the original model features
class FlightDetails(BaseModel):
    airline: str
    source_city: str
    departure_time: str
    stops: int = Field(ge=0, le=2)
    arrival_time: str
    destination_city: str
    flight_class: str
    duration: float = Field(gt=0)
    days_left: int = Field(ge=1)


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to Smart Flight Booking Assistant API",
        "model_version": "1.0-RF-Original",
        "supported_features": [
            "airline", "source_city", "departure_time", "stops",
            "arrival_time", "destination_city", "flight_class", "duration", "days_left"
        ]
    }


# Prediction endpoint
@app.post("/predict")
def predict_price(flight: FlightDetails):
    input_data = pd.DataFrame([{
        "airline": flight.airline,
        "source_city": flight.source_city,
        "departure_time": flight.departure_time,
        "stops": flight.stops,
        "arrival_time": flight.arrival_time,
        "destination_city": flight.destination_city,
        "class": flight.flight_class,
        "duration": flight.duration,
        "days_left": flight.days_left
    }])

    prediction = model.predict(input_data)[0]

    return {
        "predicted_price": round(float(prediction), 2),
        "lower_price": round(max(0, float(prediction) - TEST_MAE), 2),
        "upper_price": round(float(prediction) + TEST_MAE, 2),
        "mae": TEST_MAE,
        "currency": "INR"
    }


# Feature Importance API
@app.get("/feature-importance")
def get_feature_importance():
    rf_model = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = rf_model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    # Sort features by importance
    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    # Return top 10 features
    top_features = importance_df.head(10)

    return {
        "features": top_features["feature"].tolist(),
        "importance": [
            round(float(value) * 100, 2)
            for value in top_features["importance"]
        ]
    }


# Live Fare Insights API (Directly generated from the active original Random Forest pipeline)
@app.get("/fare-insights")
def get_fare_insights():
    # 1. Feature Importance
    rf_model = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = rf_model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).head(10)

    feature_importance_data = {
        "features": importance_df["feature"].tolist(),
        "importance": [round(float(val) * 100, 2) for val in importance_df["importance"]]
    }

    # 2. Airline Comparison (Model predictions on DEL-BOM, Morning dep, Afternoon arr, Non-stop 0, dur=2.17h, days_left=25)
    airlines = ["AirAsia", "Air_India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"]
    airline_comparison = []
    for a in airlines:
        df_e = pd.DataFrame([{
            "airline": a, "source_city": "Delhi", "departure_time": "Morning",
            "stops": 0, "arrival_time": "Afternoon", "destination_city": "Mumbai",
            "class": "Economy", "duration": 2.17, "days_left": 25
        }])
        df_b = pd.DataFrame([{
            "airline": a, "source_city": "Delhi", "departure_time": "Morning",
            "stops": 0, "arrival_time": "Afternoon", "destination_city": "Mumbai",
            "class": "Business", "duration": 2.17, "days_left": 25
        }])
        airline_comparison.append({
            "airline": a.replace("_", " "),
            "economy": round(float(model.predict(df_e)[0]), 2),
            "business": round(float(model.predict(df_b)[0]), 2)
        })

    # 3. Route Comparison (Model predictions: Vistara, Economy, Morning dep, 0 stops, days_left=25)
    corridors = [
        ("Delhi", "Mumbai", 2.17),
        ("Mumbai", "Bangalore", 1.75),
        ("Chennai", "Delhi", 2.75),
        ("Hyderabad", "Mumbai", 1.50),
        ("Delhi", "Kolkata", 2.25),
        ("Bangalore", "Kolkata", 2.50),
        ("Delhi", "Bangalore", 2.75)
    ]
    route_comparison = []
    for src, dst, dur in corridors:
        df_r = pd.DataFrame([{
            "airline": "Vistara", "source_city": src, "departure_time": "Morning",
            "stops": 0, "arrival_time": "Afternoon", "destination_city": dst,
            "class": "Economy", "duration": dur, "days_left": 25
        }])
        route_comparison.append({
            "route": f"{src} → {dst}",
            "price": round(float(model.predict(df_r)[0]), 2),
            "duration_hours": dur
        })

    # 4. Stops Impact (Model predictions: Vistara, DEL -> BOM, Morning dep, Afternoon arr, Economy, days_left=25, duration constant=2.17h)
    stops_comparison = []
    stops_specs = [
        (0, "Non-stop (Direct)"),
        (1, "1 Stop"),
        (2, "2 or More Stops")
    ]
    for stops_code, label in stops_specs:
        df_s = pd.DataFrame([{
            "airline": "Vistara", "source_city": "Delhi", "departure_time": "Morning",
            "stops": stops_code, "arrival_time": "Afternoon", "destination_city": "Mumbai",
            "class": "Economy", "duration": 2.17, "days_left": 25
        }])
        stops_comparison.append({
            "stops": label,
            "price": round(float(model.predict(df_s)[0]), 2)
        })

    return {
        "model_metadata": {
            "pkl_filename": "flight_price_rf_pipeline.pkl",
            "model_type": "sklearn.pipeline.Pipeline (RandomForestRegressor + ColumnTransformer)",
            "includes_duration": True,
            "benchmark_assumptions": "DEL → BOM, 25 Days Window, Morning Departure, Non-stop Direct, Duration=2.17h"
        },
        "feature_importance": feature_importance_data,
        "airline_comparison": airline_comparison,
        "route_comparison": route_comparison,
        "stops_comparison": stops_comparison
    }


# 50-day booking trend endpoint
@app.post("/booking-trend")
def get_booking_trend(flight: FlightDetails):
    predictions = []

    for days in range(1, 51):
        input_data = pd.DataFrame([{
            "airline": flight.airline,
            "source_city": flight.source_city,
            "departure_time": flight.departure_time,
            "stops": flight.stops,
            "arrival_time": flight.arrival_time,
            "destination_city": flight.destination_city,
            "class": flight.flight_class,
            "duration": flight.duration,
            "days_left": days
        }])

        predicted_price = model.predict(input_data)[0]
        predictions.append({
            "days_left": days,
            "predicted_price": round(float(predicted_price), 2)
        })

    cheapest = min(
        predictions,
        key=lambda x: x["predicted_price"]
    )

    return {
        "booking_trend": predictions,
        "cheapest_days_left": cheapest["days_left"],
        "cheapest_price": cheapest["predicted_price"],
        "currency": "INR"
    }


# Cheaper alternatives endpoint
@app.post("/cheaper-alternatives")
def get_cheaper_alternatives(flight: FlightDetails):
    # Function to prepare input for prediction
    def predict_flight_price(airline, stops, departure_time):
        input_data = pd.DataFrame([{
            "airline": airline,
            "source_city": flight.source_city,
            "departure_time": departure_time,
            "stops": stops,
            "arrival_time": flight.arrival_time,
            "destination_city": flight.destination_city,
            "class": flight.flight_class,
            "duration": flight.duration,
            "days_left": flight.days_left
        }])
        prediction = model.predict(input_data)[0]
        return round(float(prediction), 2)

    # Original flight price
    original_price = predict_flight_price(
        flight.airline,
        flight.stops,
        flight.departure_time
    )

    airlines = [
        "AirAsia",
        "Air India",
        "GO_FIRST",
        "Indigo",
        "SpiceJet",
        "Vistara"
    ]

    stops_options = [0, 1, 2]

    departure_times = [
        "Early_Morning",
        "Morning",
        "Afternoon",
        "Evening",
        "Night",
        "Late_Night"
    ]

    # Compare alternative airlines
    airline_alternatives = []
    for airline in airlines:
        if airline == flight.airline:
            continue
        price = predict_flight_price(airline, flight.stops, flight.departure_time)
        if price < original_price:
            airline_alternatives.append({
                "airline": airline,
                "predicted_price": price,
                "savings": round(original_price - price, 2)
            })

    # Compare number of stops
    stops_alternatives = []
    for stops in stops_options:
        if stops == flight.stops:
            continue
        price = predict_flight_price(flight.airline, stops, flight.departure_time)
        if price < original_price:
            stops_alternatives.append({
                "stops": stops,
                "predicted_price": price,
                "savings": round(original_price - price, 2)
            })

    # Compare departure times
    time_alternatives = []
    for dt in departure_times:
        if dt == flight.departure_time:
            continue
        price = predict_flight_price(flight.airline, flight.stops, dt)
        if price < original_price:
            time_alternatives.append({
                "departure_time": dt,
                "predicted_price": price,
                "savings": round(original_price - price, 2)
            })

    # Sort alternatives by lowest predicted price
    airline_alternatives.sort(key=lambda x: x["predicted_price"])
    stops_alternatives.sort(key=lambda x: x["predicted_price"])
    time_alternatives.sort(key=lambda x: x["predicted_price"])

    return {
        "original_price": original_price,
        "airline_alternatives": airline_alternatives,
        "stops_alternatives": stops_alternatives,
        "time_alternatives": time_alternatives,
        "currency": "INR"
    }
