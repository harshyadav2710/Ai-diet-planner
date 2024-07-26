from dotenv import load_dotenv

load_dotenv() ## load all the environment variables

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## Function to load Google Gemini Pro Vision API And get response

def get_gemini_repsonse(input,image,prompt):
    model=genai.GenerativeModel('gemini-1.5-flash')
    response=model.generate_content([input,image[0],prompt])
    return response.text

def input_image_setup(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
    
##initialize our streamlit app

st.set_page_config(page_title="Packaged Prodigy|AI NUTRITIONIST AND DIETITIAN")

st.header("Packaged Prodigy: Enlightened Insights on Packaged Foods 🥫🍬")
input=st.text_input("Input Prompt: ",key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","webp"])
image=""   
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.image(image, caption="Uploaded Image.", use_column_width=True)


submit=st.button("Tell me about the food item")

input_prompt="""
You are an expert nutritionist and doctor where you need to see the ingredients of a packaged food or medicine or any other edible packaged items from the image
and output the harmful contents, beneficial contents, overall impact and additional information based on the user's health conditions.
The user will provide the health conditions and the image of the ingredients list of the food item.
Output:

Harmful Contents: List any ingredients in the food that could be harmful based on your health conditions. Explain the potential negative effects of these ingredients.
Beneficial Contents: Highlight any ingredients that could be beneficial for your health.

Overall Impact: Provide a clear recommendation on whether the food item is healthy, unhealthy, risky, or neutral for you to consume, considering your health conditions.

Additional Information: Offer suggestions for alternative food items that might be more suitable for your needs.

Nutrient Breakdown: Calculate and display the amount of key nutrients (calories, sugar, fat, protein) in the food item.

Give this food item a rating out of 100 based on its healthiness. Also add unit of healthiness.

Remember to change lines between each subpart of the output.
"""

## If submit button is clicked

if submit:
    image_data=input_image_setup(uploaded_file)
    response=get_gemini_repsonse(input_prompt,image_data,input)
    st.subheader("The Response is")
    st.write(response)