import streamlit as st
import pickle
import numpy as np
import datetime

# Load your model
xgbc = pickle.load(open(r'/Users/dhritikothari/Downloads/xgbc.pkl', 'rb'))

def main():
    
    st.title("PCOS Prediction App")
    
    # Create a sidebar navigation menu
    page = st.sidebar.selectbox("Go to", [ "Explore PCOS","PCOS Prediction"])

    if page == "PCOS Prediction":
        show_pcos_prediction_page()
    elif page == "Explore PCOS":
        show_other_page()

def show_pcos_prediction_page():
    name = st.text_input("Enter your name:")
    if name:
        st.subheader(f"Welcome, {name}!")

    min_date = datetime.date(1950, 1, 1)
    max_date = datetime.date(2024, 12, 31)

    d = st.date_input("When's your birthday", datetime.date(1990, 7, 6), min_value=min_date, max_value=max_date)
    st.write('Your birthday is:', d)
 

    st.subheader("Please enter the following details")
    input_values = [
        st.number_input("What is your Age in years", value=20),
        st.number_input("What is your Weight in kg", value=50),
        st.number_input("What is your Height(Cm)", value=150),
        st.number_input("What is your BMI(Body Mass Index)", value=25),
        st.number_input("What is your Pulse rate (bpm)", value=80.0),
        st.number_input("What is your RR (breaths/min)", value=15),
        st.number_input("What is your Hemoglobin (g/dl)", value=14),
        st.number_input("Are you Cycles Regular or Irregular. Enter 2 for regular and 4 for irregular", value=2),
        st.number_input("What is your Cycle length(days)", value=4),
        st.number_input("Are you Pregnant (Y/N)", value=0),
        st.number_input("No. of aborptions (if any)", value=0),
        st.number_input("What is your Hip(inch)", value=30),
        st.number_input("What is your Waist(inch)", value=30),
        st.number_input("Have you experienced Weight gain(Y/N)", value=0),
        st.number_input("Have you experienced hair growth(Y/N)", value=0),
        st.number_input("Have you experienced Skin darkening (Y/N)", value=1),
        st.number_input("Have you experienced Hair loss(Y/N)", value=1),
        st.number_input("Have you experienced extreme Pimples(Y/N)", value=1),
        st.number_input("Do you reguarly eat Fast Food ?", value=1),
        st.number_input("Do you regularly Exercise ?", value=1),
    ]

    # Convert the input values to a list
    input_values = [val for val in input_values]

    # Predict the result
    result = predict(xgbc, input_values)

    # Display the result
    st.header("Result")
    st.write(result)

def show_other_page():
    

# Write your warm message and statement
    warm_message = """


    🌸 Hello, Beautiful! 🌸

    We're delighted to have you here. This space is dedicated to supporting your health journey with care, compassion, and understanding.

    PCOS, like any health concern, can feel overwhelming, but you're not alone. We're here to provide you with valuable insights and guidance every step of the way.

    Let's embark on this journey together, shall we?
    """

    main_statement = """
    ## PCOS Prediction Model

    This application utilizes an XGBoost model trained on physical factors such as height, weight, heart rate, and other relevant parameters to predict the likelihood of Polycystic Ovary Syndrome (PCOS) in individuals.

    **Model Accuracy:** Our model has been rigorously trained and tested, achieving an accuracy of **90.74%**.

    **Important Note:** While this model serves as a helpful tool for fast PCOS prediction, it is essential to understand its limitations. It is not a substitute for professional medical diagnosis. PCOS is a complex hormonal disorder that requires careful evaluation by a healthcare provider. 

    **Seek Medical Advice:** We strongly recommend consulting with a healthcare professional for accurate diagnosis and personalized treatment options. PCOS diagnosis involves a comprehensive assessment, including medical history, physical examination, and laboratory tests. Early detection and management are crucial for optimal health outcomes.

    **Privacy and Confidentiality:** Rest assured that your data privacy is our top priority. We adhere to strict confidentiality measures to ensure the security of your personal information.

    **Empowering Awareness:** Our goal is to empower individuals with knowledge about PCOS and encourage proactive healthcare management. By raising awareness and promoting early intervention, we aim to improve health outcomes and quality of life for individuals affected by PCOS.

    Thank you for using our application. Remember, your health is invaluable, and seeking medical guidance is always the best course of action.
    """

# Display the warm message and statement on Streamlit app
    st.markdown(warm_message)
    st.markdown(main_statement)

    
def predict(model, input_value):
    input_value = np.array(input_value).reshape(1, -1)
    probabilities = xgbc.predict_proba(input_value)

    probability_has_pcos = probabilities[0, 1]

    st.markdown(f"<p style='font-size:{"20px"}'>Probability of having PCOS: {probability_has_pcos*100} % </p>", unsafe_allow_html=True)

    threshold_has_pcos = 0.7
    threshold_has_mod_pcos = 0.4
    threshold_has_little_pcos = 0.0

    if probability_has_pcos >= threshold_has_pcos:
        return "There is a high chance that you may have PCOS. We strongly recommend consulting with a gynecologist, particularly if you've been consistently skipping menstrual cycles, for a thorough evaluation and personalized guidance on managing your health."
    elif probability_has_pcos >= threshold_has_mod_pcos:
        return "There appears to be a moderate likelihood that you may be experiencing Polycystic Ovary Syndrome (PCOS). We recommend seeking consultation with a gynecologist, especially if you are experiencing irregular menstrual cycles. While PCOS may not pose immediate health risks, adopting a balanced diet and incorporating regular exercise into your routine can help manage symptoms and promote overall well-being. "
    elif probability_has_pcos >= threshold_has_little_pcos:
        return "There appears to be a lesser likelihood that you may be experiencing Polycystic Ovary Syndrome (PCOS)"
    else:
        return "No PCOS"

if __name__ == "__main__":
    main()
