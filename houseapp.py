import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open(r"C:\Users\User\Spyder\linear_regression_housemodel.pkl",'rb'))
# Set the title of the Streamlit app
st.title("HOUSE PREDICTION APP")

# Add a brief description
st.write("This app predicts the rent based on bedrooms using a simple linear regression model.")

# Add input widget for user to enter years of experience
bedrooms = st.number_input("Enter the  bedrooms:", min_value=0.0, max_value=5.0, value=1.0, step=0.5)

# When the button is clicked, make predictions
if st.button("Predict price"):
    # Make a prediction using the trained model
    bedrooms_input = np.array([[bedrooms]])  # Convert the input to a 2D array for prediction
    prediction = model.predict(bedrooms_input)
    
    # Display the result
    st.success(f"The predicted price for {bedrooms} is: ${prediction[0]:,.2f}")
    
# Display information about the model
st.write("The model was trained using a dataset of bedrooms.")

