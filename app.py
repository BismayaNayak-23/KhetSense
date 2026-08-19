import streamlit as st
import pandas as pd
import numpy as np
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
# LOAD DATASET
# =========================================================

DATA_FILE = "KhetSense_3000_sensor_dataset.csv"

try:
    df = pd.read_csv(DATA_FILE)

except FileNotFoundError:
    st.error(
        "❌ Dataset not found.\n\n"
        "Make sure 'KhetSense_3000_sensor_dataset.csv' "
        "is in the same folder as app.py."
    )
    st.stop()


# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

required_columns = [
    "record_id",
    "crop",
    "zone",
    "growth_stage",
    "soil_moisture_pct",
    "temperature_c",
    "humidity_pct",
    "rainfall_mm",
    "nitrogen_mg_kg",
    "phosphorus_mg_kg",
    "potassium_mg_kg",
    "soil_ph",
    "water_required_mm",
    "fertilizer_required_kg_ha"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "❌ Dataset is missing required columns:"
    )

    st.write(missing_columns)

    st.stop()


# =========================================================
# TRAIN AI MODELS
# =========================================================

@st.cache_resource
def train_models(dataframe):

    data = dataframe.copy()

    # -----------------------------------------------------
    # Convert growth stage into numerical values
    # -----------------------------------------------------

    stage_map = {
        "Seedling": 1,
        "Vegetative": 2,
        "Flowering": 3,
        "Maturity": 4
    }

    data["crop_stage"] = (
        data["growth_stage"]
        .map(stage_map)
        .fillna(2)
    )

    # -----------------------------------------------------
    # WATER REQUIREMENT MODEL
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # FERTILIZER REQUIREMENT MODEL
    # -----------------------------------------------------

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


# =========================================================
# INITIALIZE AI MODELS
# =========================================================

with st.spinner("🤖 Initializing KhetSense AI models..."):

    water_model, fertilizer_model = train_models(df)


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

st.caption(
    "Prototype using simulated agricultural sensor data "
    "for precision farming."
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
        value=2.0,
        step=0.1
    )


with c3:

    crop_options = sorted(
        df["crop"].dropna().unique()
    )

    zone_options = sorted(
        df["zone"].dropna().unique()
    )

    selected_crop = st.selectbox(
        "🌾 Crop",
        crop_options
    )

    selected_zone = st.selectbox(
        "🗺️ Field Zone",
        zone_options
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
        "No exact record available for this crop and zone. "
        "Showing records from the selected crop instead."
    )

    filtered = df[
        df["crop"] == selected_crop
    ]


if len(filtered) == 0:

    st.error(
        "❌ No agricultural data is available "
        "for the selected crop."
    )

    st.stop()


# =========================================================
# SIMULATED SENSOR RECORD
# =========================================================

sensor_index = st.slider(
    "🔄 Simulated Sensor Reading",
    min_value=0,
    max_value=len(filtered) - 1,
    value=0
)

sensor = filtered.iloc[sensor_index]


st.caption(
    "⚠️ These are simulated sensor readings from the "
    "prototype dataset, not readings from physical sensors."
)


# =========================================================
# WHAT — CROP CONDITION
# =========================================================

st.markdown("---")

st.header("🌾 WHAT is happening in this field?")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Crop",
    str(sensor["crop"])
)

c2.metric(
    "Growth Stage",
    str(sensor["growth_stage"])
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

st.header("📡 Sensor Dashboard")

st.caption(
    f"Sensor readings for {selected_crop} — "
    f"Zone {selected_zone}"
)


# ---------------------------------------------------------
# ENVIRONMENTAL PARAMETERS
# ---------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "💧 Soil Moisture",
    f"{float(sensor['soil_moisture_pct']):.1f}%"
)

c2.metric(
    "🌡 Temperature",
    f"{float(sensor['temperature_c']):.1f} °C"
)

c3.metric(
    "💦 Humidity",
    f"{float(sensor['humidity_pct']):.1f}%"
)

c4.metric(
    "🌧 Rainfall",
    f"{float(sensor['rainfall_mm']):.1f} mm"
)


# ---------------------------------------------------------
# SOIL NUTRIENTS
# ---------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "N — Nitrogen",
    f"{float(sensor['nitrogen_mg_kg']):.1f} mg/kg"
)

c2.metric(
    "P — Phosphorus",
    f"{float(sensor['phosphorus_mg_kg']):.1f} mg/kg"
)

c3.metric(
    "K — Potassium",
    f"{float(sensor['potassium_mg_kg']):.1f} mg/kg"
)

c4.metric(
    "Soil pH",
    f"{float(sensor['soil_ph']):.2f}"
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

crop_stage = stage_map.get(
    sensor["growth_stage"],
    2
)


# =========================================================
# WATER MODEL INPUT
# =========================================================

water_input = pd.DataFrame({

    "soil_moisture_pct": [
        float(sensor["soil_moisture_pct"])
    ],

    "temperature_c": [
        float(sensor["temperature_c"])
    ],

    "humidity_pct": [
        float(sensor["humidity_pct"])
    ],

    "rainfall_mm": [
        float(sensor["rainfall_mm"])
    ],

    "crop_stage": [
        crop_stage
    ]
})


# =========================================================
# FERTILIZER MODEL INPUT
# =========================================================

fertilizer_input = pd.DataFrame({

    "nitrogen_mg_kg": [
        float(sensor["nitrogen_mg_kg"])
    ],

    "phosphorus_mg_kg": [
        float(sensor["phosphorus_mg_kg"])
    ],

    "potassium_mg_kg": [
        float(sensor["potassium_mg_kg"])
    ],

    "soil_ph": [
        float(sensor["soil_ph"])
    ],

    "crop_stage": [
        crop_stage
    ]
})


# =========================================================
# AI PREDICTION
# =========================================================

try:

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

except Exception as e:

    st.error(
        "❌ AI prediction failed."
    )

    st.exception(e)

    st.stop()


# =========================================================
# CLEAN PREDICTIONS
# =========================================================

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
    f"AI recommendation generated for "
    f"**{selected_crop} — Zone {selected_zone}**"
)

c1, c2 = st.columns(2)


# =========================================================
# WATER RECOMMENDATION
# =========================================================

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


# =========================================================
# FERTILIZER RECOMMENDATION
# =========================================================

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

st.header(
    "❓ WHY is KhetSense giving this recommendation?"
)

reasons = []


# Soil moisture explanation

if sensor["soil_moisture_pct"] < 30:

    reasons.append(
        "💧 Soil moisture is below 30%, "
        "indicating possible water stress."
    )

elif sensor["soil_moisture_pct"] > 60:

    reasons.append(
        "💧 Soil moisture is high, "
        "so immediate irrigation can be reduced."
    )

else:

    reasons.append(
        "💧 Soil moisture is currently within "
        "a moderate range."
    )


# Temperature explanation

if sensor["temperature_c"] > 32:

    reasons.append(
        "🌡 High temperature can increase "
        "crop water demand."
    )

elif sensor["temperature_c"] < 15:

    reasons.append(
        "🌡 Temperature is relatively low, "
        "which can reduce crop water demand."
    )

else:

    reasons.append(
        "🌡 Temperature is within a moderate "
        "range for the prototype analysis."
    )


# Rainfall explanation

if sensor["rainfall_mm"] < 10:

    reasons.append(
        "🌧 Recent rainfall is low, "
        "increasing irrigation requirements."
    )

else:

    reasons.append(
        "🌧 Recent rainfall contributes to "
        "crop water availability."
    )


# Nitrogen

if sensor["nitrogen_mg_kg"] < 50:

    reasons.append(
        "🌱 Nitrogen level is relatively low "
        "and may increase nutrient requirements."
    )

else:

    reasons.append(
        "🌱 Nitrogen level is not below the "
        "prototype's low-level threshold."
    )


# Phosphorus

if sensor["phosphorus_mg_kg"] < 30:

    reasons.append(
        "🌱 Phosphorus level is relatively low."
    )


# Potassium

if sensor["potassium_mg_kg"] < 40:

    reasons.append(
        "🌱 Potassium level is relatively low."
    )


# pH

if abs(sensor["soil_ph"] - 6.5) > 1:

    reasons.append(
        "🧪 Soil pH is relatively far from the "
        "reference value used by this prototype."
    )

else:

    reasons.append(
        "🧪 Soil pH is relatively close to the "
        "prototype reference value."
    )


# Display reasons

for reason in reasons:

    st.info(reason)


# =========================================================
# HOW — AI PROCESS
# =========================================================

st.markdown("---")

st.header(
    "⚙️ HOW does KhetSense decide?"
)

st.markdown("""
### KhetSense Decision Pipeline

**1️⃣ SENSE**

The system receives soil and environmental parameters:

`Moisture + N + P + K + pH + Temperature + Humidity + Rainfall`

⬇️

**2️⃣ ANALYZE**

The system preprocesses the selected field-zone readings and identifies the current agricultural condition.

⬇️

**3️⃣ PREDICT**

Two Random Forest machine-learning models estimate:

- 💧 Water requirement
- 🌱 Fertilizer requirement

⬇️

**4️⃣ RECOMMEND**

The predictions are converted into simple farmer-friendly recommendations.

⬇️

**5️⃣ ACT**

The farmer can use the recommendations to apply water and nutrients according to the selected field zone.

""")


# =========================================================
# WHERE — ZONE ANALYSIS
# =========================================================

st.markdown("---")

st.header(
    "🗺️ WHERE should the farmer act?"
)

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


# =========================================================
# ZONE PRIORITY
# =========================================================

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


# Round numerical values

zone_data["avg_moisture"] = (
    zone_data["avg_moisture"]
    .round(2)
)

zone_data["avg_nitrogen"] = (
    zone_data["avg_nitrogen"]
    .round(2)
)

zone_data["avg_water"] = (
    zone_data["avg_water"]
    .round(2)
)

zone_data["avg_fertilizer"] = (
    zone_data["avg_fertilizer"]
    .round(2)
)


# =========================================================
# DISPLAY ZONE TABLE
# =========================================================

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
# ZONE WATER REQUIREMENT CHART
# =========================================================

fig_water = px.bar(

    zone_data,

    x="zone",

    y="avg_water",

    text_auto=".1f",

    title="Average Water Requirement by Zone",

    labels={
        "zone": "Field Zone",
        "avg_water": "Water Requirement (mm)"
    }
)

st.plotly_chart(
    fig_water,
    use_container_width=True
)


# =========================================================
# ZONE FERTILIZER REQUIREMENT CHART
# =========================================================

fig_fertilizer = px.bar(

    zone_data,

    x="zone",

    y="avg_fertilizer",

    text_auto=".1f",

    title="Average Fertilizer Requirement by Zone",

    labels={
        "zone": "Field Zone",
        "avg_fertilizer": "Fertilizer Requirement (kg/ha)"
    }
)

st.plotly_chart(
    fig_fertilizer,
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

📐 **Farm Area:** {farm_area} ha

🎯 **Priority:**
Water = {water_status} |
Nutrient = {fertilizer_status}

KhetSense recommends treating **this specific field zone**
rather than applying the same amount across the entire farm.
"""
)


# =========================================================
# PROTOTYPE IMPACT INDICATORS
# =========================================================

st.markdown("---")

st.header(
    "🌍 Potential Precision Agriculture Impact"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "💧 Potential Water Saving",
    "24%"
)

c2.metric(
    "🌱 Potential Fertilizer Saving",
    "17%"
)

c3.metric(
    "💰 Potential Cost Reduction",
    "15%"
)

c4.metric(
    "🗺️ Management Zones",
    str(df["zone"].nunique())
)

st.caption(
    "⚠️ The saving and cost figures shown above are "
    "prototype/illustrative indicators, not experimentally "
    "validated field results."
)


# =========================================================
# AI MODEL INFORMATION
# =========================================================

st.markdown("---")

st.header("🧠 AI Model Information")

c1, c2, c3 = st.columns(3)

c1.info(
    """
**Water Model**

Random Forest Regressor

Inputs:
Moisture, Temperature,
Humidity, Rainfall,
Growth Stage
"""
)

c2.info(
    """
**Nutrient Model**

Random Forest Regressor

Inputs:
Nitrogen, Phosphorus,
Potassium, Soil pH,
Growth Stage
"""
)

c3.info(
    f"""
**Training Data**

{len(df):,} agricultural records

The current prototype uses
simulated sensor data.
"""
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🌱 KhetSense AI — "
    "Right Water. Right Nutrient. "
    "Right Zone. Right Time."
)