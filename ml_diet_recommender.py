
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

class DietRecommender:
    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=42)
        self.label_encoders = {}
        self.diet_types = {
            0: "Balanced Diet",
            1: "Low-Carb Diet",
            2: "Mediterranean Diet",
            3: "High-Protein Diet",
            4: "Diabetic-Friendly Diet"
        }
        self.train_model()

    def create_training_data(self):
        # Small synthetic dataset for demonstration purposes.
        data = {
            'age': [25, 30, 35, 40, 45, 50, 55, 60, 28, 32, 38, 42],
            'bmi': [18.5, 22.0, 25.5, 28.0, 31.2, 26.5, 24.0, 27.8, 19.5, 23.2, 29.0, 32.1],
            # Use consistent, lowercase categories: 'sedentary','low','moderate','active'
            'activity_level': ['sedentary', 'moderate', 'active', 'sedentary', 'low', 'moderate', 'active', 'low', 'moderate', 'active', 'sedentary', 'low'],
            'health_condition': ['none', 'none', 'hypertension', 'diabetes', 'diabetes', 'hypertension', 'none', 'diabetes', 'none', 'none', 'diabetes', 'hypertension'],
            'goal': ['maintenance', 'loss', 'loss', 'loss', 'loss', 'maintenance', 'gain', 'loss', 'maintenance', 'loss', 'loss', 'loss'],
            'diet_type': [0, 1, 1, 4, 4, 2, 3, 4, 0, 1, 4, 2]
        }
        return pd.DataFrame(data)

    def train_model(self):
        df = self.create_training_data()

        categorical_columns = ['activity_level', 'health_condition', 'goal']

        for col in categorical_columns:
            le = LabelEncoder()
            # Fit on lowercase strings to ensure predictable transforms
            df[col] = df[col].astype(str).str.lower()
            df[col + '_encoded'] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        X = df[['age', 'bmi', 'activity_level_encoded', 'health_condition_encoded', 'goal_encoded']]
        y = df['diet_type']

        self.model.fit(X, y)

    def predict_diet(self, age, bmi, activity_level, health_condition, goal):
        # Normalize inputs to lowercase strings that match training
        activity_level = str(activity_level).strip().lower()
        health_condition = str(health_condition).strip().lower()
        goal = str(goal).strip().lower()

        # Validate and map synonyms if necessary
        # Accept 'light' or 'lightly active' -> 'low', 'moderate'/'moderately active' -> 'moderate'
        if activity_level in ['light', 'lightly active']:
            activity_level = 'low'
        if activity_level in ['moderately active']:
            activity_level = 'moderate'
        if activity_level in ['very active', 'super active']:
            activity_level = 'active'

        # If encoder can't transform because of unseen label, fall back to a safe default
        try:
            activity_encoded = self.label_encoders['activity_level'].transform([activity_level])[0]
        except Exception:
            activity_encoded =  self.label_encoders['activity_level'].transform(['sedentary'])[0]

        try:
            condition_encoded = self.label_encoders['health_condition'].transform([health_condition])[0]
        except Exception:
            condition_encoded = self.label_encoders['health_condition'].transform(['none'])[0]

        try:
            goal_encoded = self.label_encoders['goal'].transform([goal])[0]
        except Exception:
            goal_encoded = self.label_encoders['goal'].transform(['maintenance'])[0]

        features = np.array([[age, bmi, activity_encoded, condition_encoded, goal_encoded]])
        prediction = self.model.predict(features)[0]

        return self.diet_types.get(prediction, "Balanced Diet")

    def get_diet_details(self, diet_type):
        diet_details = {
            "Balanced Diet": {
                "description": "A well-rounded diet with all food groups",
                "foods": ["Whole grains", "Lean proteins", "Fruits", "Vegetables", "Healthy fats"],
                "macros": "Carbs: 45-65%, Protein: 15-25%, Fat: 20-35%"
            },
            "Low-Carb Diet": {
                "description": "Reduced carbohydrate intake for weight loss",
                "foods": ["Lean meats", "Fish", "Eggs", "Leafy greens", "Nuts", "Avocado"],
                "macros": "Carbs: 10-20%, Protein: 25-35%, Fat: 45-65%"
            },
            "Mediterranean Diet": {
                "description": "Heart-healthy diet rich in olive oil and fish",
                "foods": ["Olive oil", "Fish", "Whole grains", "Fruits", "Vegetables", "Nuts"],
                "macros": "Carbs: 40-50%, Protein: 15-20%, Fat: 30-40%"
            },
            "High-Protein Diet": {
                "description": "Increased protein for muscle building",
                "foods": ["Chicken", "Fish", "Eggs", "Greek yogurt", "Quinoa", "Beans"],
                "macros": "Carbs: 30-40%, Protein: 30-40%, Fat: 20-30%"
            },
            "Diabetic-Friendly Diet": {
                "description": "Low glycemic index foods for blood sugar control",
                "foods": ["Whole grains", "Lean proteins", "Non-starchy vegetables", "Berries", "Nuts"],
                "macros": "Carbs: 40-45%, Protein: 20-25%, Fat: 30-35%"
            }
        }
        return diet_details.get(diet_type, diet_details["Balanced Diet"])
