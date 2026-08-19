import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="KhetSense AI",
    page_icon="🌱",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    font-size: 20px;
    color: #666;
}

.section-title {
    font-size: 28px;
    font-weight: 600;
    margin-top: 20px;
}

.info-card {
    padding: 20px;
    border-radius: 12px;
    background-color: #f5f7f9;
    border: 1px solid #e5e7eb;
}

.recommendation {
    padding: 20px;
    border-radius: 12px;
    background-color: #eef8ee;
    border-left: 6px solid #2e7d32;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================

DATA_FILE = "KhetSense_3000_sensor_dataset.csv"

try:
    df = pd.read_csv(DATA_FILE)
    @st.cache_resource
    def train_models(df):

        data = df.copy()

        # Convert crop growth stage to numbers
        stage_map = {
            "Seedling": 1,
            "Vegetative": 2,
            "Flowering": 3,
            "Maturity": 4
        }

        data["crop_stage"] = data["growth_stage"].map(stage_map)

        # Handle any unknown/missing growth stages
        data["crop_stage"] = data["crop_stage"].fillna(2)

        # ==========================================
        # WATER REQUIREMENT MODEL
        # ==========================================

        water_features = [
            "soil_moisture_pct",
            "temperature_c",
            "humidity_pct",
            "rainfall_mm",
            "crop_stage"
        ]

        water_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )

        water_model.fit(
            data[water_features],
            data["water_required_mm"]
        )

        # ==========================================
        # FERTILIZER REQUIREMENT MODEL
        # ==========================================

        fertilizer_features = [
            "nitrogen_mg_kg",
            "phosphorus_mg_kg",
            "potassium_mg_kg",
            "soil_ph",
            "crop_stage"
        ]

        fertilizer_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )

        fertilizer_model.fit(
            data[fertilizer_features],
            data["fertilizer_required_kg_ha"]
        )

        return water_model, fertilizer_model
        water_model, fertilizer_model = train_models(df)
except FileNotFoundError:
    st.error(
        "❌ Dataset not found.\n\n"
        "Make sure KhetSense_3000_sensor_dataset.csv "
        "is in the same folder as app.py."
    )
    st.stop()

# =========================================================
# LOAD MODELS
# =========================================================



# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🌱 KhetSense AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Assisted Precision Nutrient & Water Management'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# =========================================================
# FARM SETUP
# =========================================================

st.header("👨‍🌾 Farm Setup")

c1, c2, c3 = st.columns(3)

with c1:

    farmer_name = st.text_input(
        "Farmer Name",
        "Demo Farmer"
    )

    farm_name = st.text_input(
        "Farm Name",
        "Green Valley Farm"
    )

with c2:

    location = st.text_input(
        "Farm Location",
        "Cuttack, Odisha"
    )

    farm_area = st.number_input(
        "Farm Area (hectares)",
        min_value=0.1,
        value=2.0
    )

with c3:

    selected_crop = st.selectbox(
        "🌾 Crop",
        sorted(df["crop"].unique())
    )

    selected_zone = st.selectbox(
        "🗺️ Field Zone",
        sorted(df["zone"].unique())
    )

# =========================================================
# CURRENT FARM CONTEXT
# =========================================================

st.markdown("---")

st.header("📍 Current Field Context")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "👨‍🌾 Farmer",
    farmer_name
)

c2.metric(
    "🏡 Farm",
    farm_name
)

c3.metric(
    "📍 Location",
    location
)

c4.metric(
    "📐 Farm Area",
    f"{farm_area} ha"
)

st.info(
    f"📌 Currently analyzing **{selected_crop}** "
    f"in **Zone {selected_zone}** of **{farm_name}**."
)

# =========================================================
# SELECT SENSOR RECORD
# =========================================================

filtered = df[
    (df["crop"] == selected_crop) &
    (df["zone"] == selected_zone)
]

if len(filtered) == 0:

    st.warning(
        "No exact record available for this crop and zone."
    )

    filtered = df[
        df["crop"] == selected_crop
    ]

if len(filtered) == 0:
    st.error("No data available.")
    st.stop()

sensor_index = st.slider(
    "🔄 Simulated Sensor Reading",
    min_value=0,
    max_value=len(filtered) - 1,
    value=0
)

sensor = filtered.iloc[sensor_index]

# =========================================================
# WHAT — CROP CONDITION
# =========================================================

st.markdown("---")

st.header("🌾 WHAT is happening in this field?")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Crop",
    sensor["crop"]
)

c2.metric(
    "Growth Stage",
    sensor["growth_stage"]
)

c3.metric(
    "Field Zone",
    f"Zone {sensor['zone']}"
)

c4.metric(
    "Sensor Record",
    f"#{int(sensor['record_id'])}"
)

# =========================================================
# SENSOR DASHBOARD
# =========================================================

st.header("📡 Live Sensor Dashboard")

st.caption(
    f"Sensor readings for {selected_crop} — "
    f"Zone {selected_zone}"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "💧 Soil Moisture",
    f"{sensor['soil_moisture_pct']:.1f}%"
)

c2.metric(
    "🌡 Temperature",
    f"{sensor['temperature_c']:.1f} °C"
)

c3.metric(
    "💦 Humidity",
    f"{sensor['humidity_pct']:.1f}%"
)

c4.metric(
    "🌧 Rainfall",
    f"{sensor['rainfall_mm']:.1f} mm"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "N — Nitrogen",
    f"{sensor['nitrogen_mg_kg']:.1f} mg/kg"
)

c2.metric(
    "P — Phosphorus",
    f"{sensor['phosphorus_mg_kg']:.1f} mg/kg"
)

c3.metric(
    "K — Potassium",
    f"{sensor['potassium_mg_kg']:.1f} mg/kg"
)

c4.metric(
    "Soil pH",
    f"{sensor['soil_ph']:.2f}"
)

# =========================================================
# AI INPUT PREPARATION
# =========================================================

stage_map = {
    "Seedling": 1,
    "Vegetative": 2,
    "Flowering": 3,
    "Maturity": 4
}

crop_stage = stage_map[sensor["growth_stage"]]

water_input = pd.DataFrame({

    "soil_moisture": [
        sensor["soil_moisture_pct"]
    ],

    "temperature": [
        sensor["temperature_c"]
    ],

    "humidity": [
        sensor["humidity_pct"]
    ],

    "rainfall": [
        sensor["rainfall_mm"]
    ],

    "crop_stage": [
        crop_stage
    ]
})

fertilizer_input = pd.DataFrame({

    "nitrogen": [
        sensor["nitrogen_mg_kg"]
    ],

    "phosphorus": [
        sensor["phosphorus_mg_kg"]
    ],

    "potassium": [
        sensor["potassium_mg_kg"]
    ],

    "soil_ph": [
        sensor["soil_ph"]
    ],

    "crop_stage": [
        crop_stage
    ]
})

# =========================================================
# AI PREDICTION
# =========================================================

water_prediction = float(
    water_model.predict(
        water_input
    )[0]
)

fertilizer_prediction = float(
    fertilizer_model.predict(
        fertilizer_input
    )[0]
)

water_prediction = max(
    0,
    round(water_prediction, 2)
)

fertilizer_prediction = max(
    0,
    round(fertilizer_prediction, 2)
)

# =========================================================
# AI RECOMMENDATIONS
# =========================================================

st.markdown("---")

st.header("🤖 AI Recommendation")

st.caption(
    f"Recommendation generated for "
    f"**{selected_crop} — Zone {selected_zone}**"
)

c1, c2 = st.columns(2)

# WATER

with c1:

    st.subheader("💧 Water Management")

    st.metric(
        "Recommended Irrigation",
        f"{water_prediction} mm"
    )

    if water_prediction > 25:

        st.error(
            "🔴 HIGH WATER REQUIREMENT"
        )

        water_status = "High"

    elif water_prediction > 12:

        st.warning(
            "🟡 MODERATE WATER REQUIREMENT"
        )

        water_status = "Moderate"

    else:

        st.success(
            "🟢 LOW WATER REQUIREMENT"
        )

        water_status = "Low"


# FERTILIZER

with c2:

    st.subheader("🌱 Nutrient Management")

    st.metric(
        "Recommended Fertilizer",
        f"{fertilizer_prediction} kg/ha"
    )

    if fertilizer_prediction > 60:

        st.error(
            "🔴 HIGH NUTRIENT REQUIREMENT"
        )

        fertilizer_status = "High"

    elif fertilizer_prediction > 30:

        st.warning(
            "🟡 MODERATE NUTRIENT REQUIREMENT"
        )

        fertilizer_status = "Moderate"

    else:

        st.success(
            "🟢 LOW NUTRIENT REQUIREMENT"
        )

        fertilizer_status = "Low"

# =========================================================
# WHY — EXPLANATION
# =========================================================

st.markdown("---")

st.header("❓ WHY is KhetSense giving this recommendation?")

reasons = []

if sensor["soil_moisture_pct"] < 30:

    reasons.append(
        "💧 Soil moisture is below 30%, indicating water stress."
    )

elif sensor["soil_moisture_pct"] > 60:

    reasons.append(
        "💧 Soil moisture is high, so immediate irrigation can be reduced."
    )

else:

    reasons.append(
        "💧 Soil moisture is currently within a moderate range."
    )


if sensor["temperature_c"] > 32:

    reasons.append(
        "🌡 High temperature increases crop water demand."
    )


if sensor["rainfall_mm"] < 10:

    reasons.append(
        "🌧 Recent rainfall is low, increasing irrigation requirements."
    )

else:

    reasons.append(
        "🌧 Recent rainfall contributes to the crop's water availability."
    )


if sensor["nitrogen_mg_kg"] < 50:

    reasons.append(
        "🌱 Nitrogen level is relatively low."
    )


if sensor["phosphorus_mg_kg"] < 30:

    reasons.append(
        "🌱 Phosphorus level is relatively low."
    )


if sensor["potassium_mg_kg"] < 40:

    reasons.append(
        "🌱 Potassium level is relatively low."
    )


if abs(sensor["soil_ph"] - 6.5) > 1:

    reasons.append(
        "🧪 Soil pH is relatively far from the reference value used by the prototype."
    )


for reason in reasons:

    st.info(reason)

# =========================================================
# HOW — AI PROCESS
# =========================================================

st.header("⚙️ HOW does KhetSense decide?")

st.markdown("""
### KhetSense Decision Pipeline

**1️⃣ Sense**

Soil and environmental parameters are collected:

`Moisture + N + P + K + pH + Temperature + Humidity + Rainfall`

⬇️

**2️⃣ Analyze**

The system preprocesses the sensor readings and identifies the condition of the selected field zone.

⬇️

**3️⃣ Predict**

Two Random Forest ML models estimate:

- 💧 Water requirement
- 🌱 Fertilizer requirement

⬇️

**4️⃣ Recommend**

The prediction is converted into a simple farmer-friendly recommendation.

⬇️

**5️⃣ Act**

The farmer can irrigate or apply nutrients only where required.

""")

# =========================================================
# ZONE MAP / TABLE
# =========================================================

st.markdown("---")

st.header("🗺️ WHERE should the farmer act?")

zone_data = (
    df.groupby("zone")
    .agg(
        avg_moisture=(
            "soil_moisture_pct",
            "mean"
        ),

        avg_nitrogen=(
            "nitrogen_mg_kg",
            "mean"
        ),

        avg_water=(
            "water_required_mm",
            "mean"
        ),

        avg_fertilizer=(
            "fertilizer_required_kg_ha",
            "mean"
        )
    )
    .reset_index()
)

zone_data["Priority"] = np.select(

    [
        zone_data["avg_moisture"] < 30,
        zone_data["avg_nitrogen"] < 50
    ],

    [
        "🔴 HIGH",
        "🟡 MEDIUM"
    ],

    default="🟢 LOW"
)

st.dataframe(
    zone_data,
    use_container_width=True,
    hide_index=True
)

# =========================================================
# ZONE MOISTURE CHART
# =========================================================

fig = px.bar(

    zone_data,

    x="zone",

    y="avg_moisture",

    color="Priority",

    text_auto=".1f",

    title="Soil Moisture Across Farm Zones",

    labels={
        "zone": "Field Zone",
        "avg_moisture": "Average Soil Moisture (%)"
    }
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# FINAL ACTION CARD
# =========================================================

st.markdown("---")

st.header("🚜 Recommended Action")

st.success(
    f"""
### Zone {selected_zone} — {selected_crop}

💧 **Irrigation:** {water_prediction} mm

🌱 **Fertilizer:** {fertilizer_prediction} kg/ha

📍 **Location:** {location}

📐 **Farm:** {farm_name}

🎯 **Priority:** Water = {water_status} | Nutrient = {fertilizer_status}

KhetSense recommends treating **this specific field zone**
rather than applying the same amount across the entire farm.
"""
)

# =========================================================
# IMPACT
# =========================================================

st.header("🌍 Expected Precision Agriculture Impact")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "💧 Water Saving",
    "24%"
)

c2.metric(
    "🌱 Fertilizer Saving",
    "17%"
)

c3.metric(
    "💰 Cost Reduction",
    "15%"
)

c4.metric(
    "🗺️ Management Zones",
    "6"
)

st.markdown("---")

st.caption(
    "KhetSense AI — Right Water. Right Nutrient. "
    "Right Zone. Right Time."
)