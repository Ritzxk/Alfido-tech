import streamlit as st
import joblib
import numpy as np

# Load the trained model and preprocessing pipeline
best_model = joblib.load('best_model.pkl')
pipe = joblib.load('pipe.pkl')

# Define the flower species labels
flower_species = {
    0: 'Iris-Setosa',
    1: 'Iris-Versicolor',
    2: 'Iris-Virginica'
}

# Set the page configuration (Only call it once and make it the first Streamlit command)
st.set_page_config(page_title="Iris Flower Species Prediction")

url1 = 'https://r4.wallpaperflare.com/wallpaper/519/115/814/flowers-4k-desktop-hd-wallpaper-98a6dd6870703c3880fc51ce88c2446a.jpg'
url2 = 'https://r4.wallpaperflare.com/wallpaper/539/743/614/flower-4k-desktop-background-hd-wallpaper-963a3cb432c8ed6668b38bc2580a1d94.jpg'
url3 = 'https://r4.wallpaperflare.com/wallpaper/739/420/15/beautiful-flower-4k-hd-desktop-wallpaper-5866cda8d0d05c58004c815eb882549a.jpg'
# Add custom CSS to set the background image
css = f"""
    <style>
        body {{
            background-image: url({url3});
            background-size: cover;
            background-repeat: repeat;
        }}
        .stApp {{
            background-color: rgba(0, 0, 0, 0.6);
            padding: 20px;
            border-radius: 10px;
        }}

    </style>
"""

# Push the CSS to the page
st.markdown(css, unsafe_allow_html=True)


# Set the title and introduction text
st.title("Iris Flower Species Prediction")
st.write("This app predicts the species of Iris flowers based on input features.")

# Create sliders for user input
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 7.0, 5.0)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

# Create a button to trigger the prediction
if st.sidebar.button("Predict"):
    # Create a NumPy array with the user's input
    user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Preprocess the user's input using the pipeline
    user_input_transformed = pipe.transform(user_input)

    # Make a prediction using the trained model
    prediction = best_model.predict(user_input_transformed)

    # Display the predicted flower species
    st.subheader("Prediction:")
    st.write(f"The predicted flower species is:")
    st.subheader(flower_species[prediction[0]])

    if prediction[0] == 0:
            st.image(
                  "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/675px-Kosaciec_szczecinkowaty_Iris_setosa.jpg",
                  width=500)
    elif prediction == 1:
          st.image(
                'https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/1200px-Iris_versicolor_3.jpg',
                width=500)
    else:
          st.image(
                'https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/1104px-Iris_virginica.jpg',
                width=500)

# Optionally, display the input feature values
st.sidebar.subheader("Selected Input Features")
st.sidebar.write(f"Sepal Length: {sepal_length} cm")
st.sidebar.write(f"Sepal Width: {sepal_width} cm")
st.sidebar.write(f"Petal Length: {petal_length} cm")
st.sidebar.write(f"Petal Width: {petal_width} cm")

# Add some information about the Iris dataset
st.sidebar.subheader("About the Iris Dataset")
st.sidebar.write("The Iris dataset contains measurements of four features of three species of Iris flowers.")

# Add a link to the dataset
st.sidebar.markdown('[Iris Dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set)')

# Add a footer
st.sidebar.text("Made with ❤️ by Ritesh")
