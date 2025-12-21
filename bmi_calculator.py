
def calculate_bmi(weight, height):
    """Calculate BMI given weight (kg) and height (cm)."""
    height_m = height / 100.0
    return round(weight / (height_m ** 2), 2)

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"
