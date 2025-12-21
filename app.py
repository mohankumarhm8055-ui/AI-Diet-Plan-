import streamlit as st
import pandas as pd
from bmi_calculator import calculate_bmi, get_bmi_category
from bmr_tdee_calculator import calculate_bmr, calculate_tdee
from ml_diet_recommender import DietRecommender

# Initialize ML model
@st.cache_resource
def load_model():
    return DietRecommender()

recommender = load_model()

# App title
st.set_page_config(page_title="AI Diet Planner", layout="wide")
st.title("🤖 AI-Powered Personalized Diet Planner")
st.markdown("Using a simple Decision Tree to suggest diet types\n\n---")

# Sidebar for user inputs
st.sidebar.header("Enter Your Information")

# User inputs
age = st.sidebar.number_input("Age", min_value=15, max_value=100, value=30)
weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, format="%.1f")
height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, format="%.1f")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

activity_level = st.sidebar.selectbox(
    "Activity Level",
    ["Sedentary", "Low", "Moderate", "Active"]
)

health_condition = st.sidebar.selectbox(
    "Health Condition",
    ["None", "Diabetes", "Hypertension"]
)

goal = st.sidebar.selectbox(
    "Health Goal",
    ["Maintenance", "Loss", "Gain"]
)

# Calculate button
if st.sidebar.button("🔍 Generate AI Recommendation", type="primary"):
    # Calculate basic metrics
    bmi = calculate_bmi(weight, height)
    bmi_category = get_bmi_category(bmi)
    bmr = calculate_bmr(weight, height, age, gender)

    # Convert activity level for TDEE calculation
    activity_mapping = {
        "Sedentary": "sedentary",
        "Low": "low",
        "Moderate": "moderate",
        "Active": "active"
    }
    tdee = calculate_tdee(bmr, activity_mapping[activity_level])

    # Get AI recommendation (pass normalized lowercase values)
    recommended_diet = recommender.predict_diet(
        age, bmi, activity_mapping[activity_level], health_condition.lower(), goal.lower()
    )

    diet_details = recommender.get_diet_details(recommended_diet)

    # Display results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Your Health Metrics")
        st.metric("BMI", f"{bmi}", f"{bmi_category}")
        st.metric("BMR", f"{bmr} cal/day", "Basal Metabolic Rate")
        st.metric("TDEE", f"{tdee} cal/day", "Total Daily Energy Expenditure")

        # Healthy weight range
        height_m = height / 100.0
        healthy_min = round(18.5 * height_m ** 2, 1)
        healthy_max = round(24.9 * height_m ** 2, 1)
        st.info(f"💡 Healthy Weight Range: {healthy_min} - {healthy_max} kg")

    with col2:
        st.subheader("🤖 AI Diet Recommendation")
        st.success(f"*Recommended Diet: {recommended_diet}*")

        st.write(f"*Description:* {diet_details['description']}")
        st.write(f"*Macronutrient Ratio:* {diet_details['macros']}")

        st.write("*Recommended Foods:*")
        for food in diet_details['foods']:
            st.write(f"• {food}")

    # Display model confidence (simple explanation)
    st.subheader("🧠 AI Model Insights")
    st.write("This recommendation is based on a Decision Tree classifier trained on a small synthetic dataset for demonstration.")

    with st.expander("How the AI made this decision"):
        st.write("The AI considered these factors:")
        st.write(f"• Your BMI ({bmi}) - {bmi_category}")
        st.write(f"• Your age ({age}) and activity level ({activity_level})")
        st.write(f"• Your health condition ({health_condition})")
        st.write(f"• Your goal ({goal})")
        st.write("Based on similar profiles in the training data, this diet type looked suitable.")

# Footer / About
st.markdown("---")
st.subheader("🔬 About This AI System")
col1, col2, col3 = st.columns(3)
with col1:
    st.write("*Algorithm Used:*")
    st.write("Decision Tree Classifier")
with col2:
    st.write("*Features Considered:*")
    st.write("- Age\n- BMI\n- Activity Level\n- Health Conditions\n- Health Goals")
with col3:
    st.write("*Diet Types:*")
    st.write("- Balanced Diet\n- Low-Carb Diet\n- Mediterranean Diet\n- High-Protein Diet\n- Diabetic-Friendly Diet")    
