import streamlit as st
import pandas as pd
import joblib

# Load the trained model and preprocessor
data = joblib.load('balanced_titanic_model.pkl')
model = data['model']
preprocessor = data['preprocessor']

# Function to predict survival
def predict_survival(passenger_data):
  df_passenger = pd.DataFrame([passenger_data])
  X_passenger = preprocessor.transform(df_passenger)
  prediction = model.predict(X_passenger)
  return 'Passenger Survived' if prediction[0] == 1 else 'Passenger Did not survive'

# Streamlit UI enhancements
st.set_page_config(page_title="Titanic Survival Prediction", page_icon=":ship:")  # Set title and icon

# App layout with columns
col1, col2 = st.columns([3, 6])  # Adjust column widths as needed

# Title and subtitle in col1 (centered)
with col1:
  st.markdown("""
<h1 >Titanic Survival Prediction</h1>  
Explore if a passenger would have survived based on historical data.
""", unsafe_allow_html=True)

# User inputs in col2 with clear labels
with col2:
  st.subheader("Passenger Information:")
  sex = st.selectbox("Sex", ["male", "female"], key="sex")
  pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3], key="pclass")
  age = st.slider("Age", 0, 100, 30, key="age")
  sibsp = st.slider("Number of Siblings/Spouses Aboard (SibSp)", 0, 10, 0, key="sibsp")
  parch = st.slider("Number of Parents/Children Aboard (Parch)", 0, 10, 0, key="parch")
  fare = st.number_input("Fare", min_value=0.0, value=15.0, step=0.1, key="fare")
  embarked = st.selectbox("Port of Embarkation (Embarked)", ["C", "Q", "S"], key="embarked")

# Predict button in col2, styled for better visibility
with col2:
  predict_button = st.button("Predict", key="predict_button")

# Prediction result, displayed conditionally
  if predict_button:
    passenger_data = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }
    result = predict_survival(passenger_data)
    st.write(f'<h3 style="color: #2ecc71;">Prediction: {result}</h3>', unsafe_allow_html=True)




# Load custom CSS from styles.css
with open("static/styles.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Add "Made by" message at the bottom right corner
st.markdown("""
    <div class="footer">
        Made by Swarnabha
        <a href="https://github.com/swarnabha-dev/" target="_blank"><img src="https://img.icons8.com/?size=30&id=LoL4bFzqmAa0&format=png&color=000000" style="vertical-align: middle;"></a>
        <a href="https://www.instagram.com/swarnabha_halder/" target="_blank"><img src="https://img.icons8.com/fluent/30/000000/instagram-new.png" style="vertical-align: middle;"></a>
        <a href="https://www.linkedin.com/in/swarnabha-halder-627692254/" target="_blank"><img src="https://img.icons8.com/color/30/000000/linkedin.png" style="vertical-align: middle;"></a>
    </div>
    """, unsafe_allow_html=True)