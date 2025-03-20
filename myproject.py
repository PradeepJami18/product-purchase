import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('Social_Network_Ads.csv')

# Encode categorical data
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Prepare features and target variable
X = df.drop(['Purchased', 'User ID'], axis=1)
y = df['Purchased']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Streamlit UI
st.set_page_config(page_title="Purchase Prediction App", layout="wide")

st.title("🛒 Purchase Prediction")
st.write("Predict whether a user will purchase a product based on age, salary, and gender.")

# Sidebar input
st.sidebar.header("User Details")
age = st.sidebar.number_input("Enter Age", min_value=18, max_value=80, value=25, step=1)
salary = st.sidebar.number_input("Enter Estimated Salary", min_value=10000, max_value=200000, value=50000, step=1000)
gender = st.sidebar.radio("Select Gender", options=["Male", "Female"])

# Convert gender to numerical value
gender_encoded = 0 if gender == "Male" else 1

if st.sidebar.button("Predict"):
    prediction = model.predict([[gender_encoded, age, salary]])
    
    # Display result
    if prediction == 1:
        st.success("🎉 You are likely to purchase the product!")
    else:
        st.error("😞 You are unlikely to purchase the product.")
