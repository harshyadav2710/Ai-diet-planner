from dotenv import load_dotenv

load_dotenv() ## load all the environment variables

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import re
from pymongo import MongoClient
import datetime

# Replace placeholders with your MongoDB connection details
connection_string ="mongodb://localhost:27017/"  # Update with your connection string if needed
client = MongoClient(connection_string)
db = client["DIET_TRACKER_DB"]  # Replace with your desired database name
collection = db["DIET_INTAKE"]  # Replace with your desired collection name

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def cal(text):
# Extract total calories
    total_calories_match = re.search(r"Total Calories: (\d+) calories", text)
    if total_calories_match:
        total_calories = int(total_calories_match.group(1))
    else:
        total_calories = None

    # Extract percentages using a loop (assuming consistent format)
    percentages = {}
    for line in text.splitlines():
        match = re.search(r"(\w+): (\d+)g", line)
        if match:
            nutrient, value = match.groups()
            percentages[nutrient] = int(value)
    
    lines = text.splitlines()

    # Extract names by splitting each line at the hyphen and taking the first element
    names = [line.split(" - ")[0] for line in lines if "-" in line]
    data = {
        "date_time": datetime.datetime.now(),
        "total_calories": total_calories,
        "percentages": percentages,
        "names": names
    }
    return data

def save_to_mongo_db(response):
    
    # Insert the data into the collection
    data = {
        "date_time": response['date_time'],  # Import datetime for current time
        "total_calories": response['total_calories'],
        "carbohydrates": response['percentages']['Carbohydrates'],
        "proteins": response['percentages']['Proteins'],
        "fats": response['percentages']['Fats'],
        "fibers": response['percentages']['Fibers'],
        "sugars": response['percentages']['Sugars'],
        "names": response['names'],
        # "text": response['text']  # Optional: Store the original text for reference
    }
    print(data)
    collection.insert_one(data)

    # Confirm the data was inserted
    st.write("Data inserted into MongoDB")    

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

st.set_page_config(page_title="Meal Tracker App|AI NUTRITIONIST AND DIETITIAN")

st.header("Introducing NutriScan: Your Meal Tracker Companion 🍇🍕🍔🍗🍑")
input=st.text_input("Input Prompt: ",key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","webp"])
image=""   
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", width=200, use_column_width=True)


submit=st.button("Tell me about the meal")

input_prompt="""
You are an expert in nutritionist where you need to see the food items from the image
               and calculate the total calories, also provide the details of every food items with calories intake
               strictly in below format
            
               1. Item 1 - no of calories
               2. Item 2 - no of calories
               ----
               ----
               Total Calories: xxx calories
Finally you must also mention whether the items are healthy or not.
Must also mention grams of carbohydatres, proteins, fats, fibers, sugars, 
and other things which you can observe in the food items strictly in below format
                Carbohydrates: xxg
                Proteins: xxg
                Fats: xxg
                Fibers: xxg
                Sugars: xxg
                Others: xxg
Give detailed information about the food items and its health impact.

"""

## If submit button is clicked

if submit:
    image_data=input_image_setup(uploaded_file)
    response=get_gemini_repsonse(input_prompt,image_data,input)
    st.subheader("The Response is")
    st.write(response)
    data=cal(response)
    # save = st.button("Save to Meal Tracker")
    save_to_mongo_db(data)
    st.write("Data Saved to MongoDB")