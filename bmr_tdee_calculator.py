
def calculate_bmr(weight_kg, height_cm, age, gender):
    """Calculate BMR using Mifflin-St Jeor equation"""
    if gender.lower() == 'male':
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
    return round(bmr, 2)

def calculate_tdee(bmr, activity_level):
    """Calculate TDEE using activity factors.
    activity_level must be one of: 'sedentary', 'low', 'moderate', 'active', 'very active'.
    """
    factors = {
        'sedentary': 1.2,
        'low': 1.375,            # lightly active
        'moderate': 1.55,       # moderately active
        'active': 1.725,        # very active
        'very active': 1.9      # super active
    }
    key = activity_level.strip().lower()
    if key not in factors:
        raise ValueError(f"Unknown activity level: {activity_level}. Use one of {list(factors.keys())}")
    return round(bmr * factors[key], 2)
