import streamlit as st
from datetime import datetime, timedelta
import csv
from pymongo import MongoClient
import pandas as pd 
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import  OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import tempfile

load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
# os.environ['GOOGLE_API_KEY' ]= api_key
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

connection_string ="mongodb://localhost:27017/"
client = MongoClient(connection_string)
db = client["DIET_TRACKER_DB"]  
collection = db["DIET_INTAKE"]  


def from_mongo_to_csv():
    # Define the output CSV file name
    csv_filename = "your_data.csv"

    # Get field names from the first document (assuming consistent structure)
    cursor = collection.find()
    first_doc = cursor.limit(1).next()  # Assuming documents have fields
    fieldnames = list(first_doc.keys())

    # Open the CSV file in write mode with UTF-8 encoding
    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate through documents and write to CSV
        for doc in collection.find():
            # Remove the "_id" field if you don't want it in the CSV
            doc.pop("_id", None)
            writer.writerow(doc)

    print(f"Data successfully exported to CSV file: {csv_filename}")

def cal_meal():

    # Load the CSV data into a DataFrame
    df = pd.read_csv('your_data.csv')

    # Convert columns to numeric (except for 'names')
    numeric_cols = ['fats', 'total_calories', 'proteins', 'fibers', 'sugars', 'carbohydrates']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Convert 'date_time' column to datetime
    df['date_time'] = pd.to_datetime(df['date_time'])

    # Get the date 30 days ago
    date_30_days_ago = datetime.now() - timedelta(days=30)

    # Filter the dataframe to only include the last 30 days
    df_last_30_days = df[df['date_time'] > date_30_days_ago]

    # Calculate mean, median, and mode for the last 30 days
    mean_values = df_last_30_days[numeric_cols].mean()
    median_values = df_last_30_days[numeric_cols].median()
    mode_values = df_last_30_days[numeric_cols].mode().iloc[0]  # Take the first mode if multiple exist

    # Convert the Series to dictionaries
    mean_values_dict = mean_values.to_dict()
    median_values_dict = median_values.to_dict()
    mode_values_dict = mode_values.to_dict()

    # Construct the strings
    mean_values_str = "Mean Nutrient Intake per meal (Last 30 Days): "
    for key, value in mean_values_dict.items():
        mean_values_str += f"{key}: {value}, "
    mean_values_str = mean_values_str.rstrip(", ")

    median_values_str = "Median Nutrient Intake per meal (Last 30 Days): "
    for key, value in median_values_dict.items():
        median_values_str += f"{key}: {value}, "
    median_values_str = median_values_str.rstrip(", ")

    mode_values_str = "Mode Nutrient Intake per meal (Last 30 Days): "
    for key, value in mode_values_dict.items():
        mode_values_str += f"{key}: {value}, "
    mode_values_str = mode_values_str.rstrip(", ")
    all_stats_str = f"{mean_values_str}\n{median_values_str}\n{mode_values_str}"
    # Print the strings
    return all_stats_str
def cal_day():
    df = pd.read_csv('your_data.csv')

    # Convert 'date_time' column to datetime format
    df['date_time'] = pd.to_datetime(df['date_time'])

    # Convert columns to numeric (except for 'names')
    numeric_cols = ['fats', 'total_calories', 'proteins', 'fibers', 'sugars', 'carbohydrates']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Extract date from 'date_time' column
    df['date'] = df['date_time'].dt.date

    # Group by date and calculate daily totals
    daily_totals = df.groupby('date')[numeric_cols].sum()

    # Filter data for the last 30 days
    last_30_days_totals = daily_totals.tail(30)

    # Calculate mean, median, and mode for the last 30 days
    mean_last_30_days = last_30_days_totals.mean()
    median_last_30_days = last_30_days_totals.median()
    mode_last_30_days = last_30_days_totals.mode().iloc[0] if not last_30_days_totals.mode().empty else np.nan

    mean_last_30_days_dict = mean_last_30_days.to_dict()
    median_last_30_days_dict = median_last_30_days.to_dict()
    mode_last_30_days_dict = mode_last_30_days.to_dict()

    # Construct the strings
    mean_last_30_days_str = "Mean Nutrient Intake per day (Last 30 Days): "
    for key, value in mean_last_30_days_dict.items():
        mean_last_30_days_str += f"{key}: {value}, "
    mean_last_30_days_str = mean_last_30_days_str.rstrip(", ")

    median_last_30_days_str = "Median Nutrient Intake per day (Last 30 Days): "
    for key, value in median_last_30_days_dict.items():
        median_last_30_days_str += f"{key}: {value}, "
    median_last_30_days_str = median_last_30_days_str.rstrip(", ")

    mode_last_30_days_str = "Mode Nutrient Intake per day (Last 30 Days): "
    for key, value in mode_last_30_days_dict.items():
        mode_last_30_days_str += f"{key}: {value}, "
    mode_last_30_days_str = mode_last_30_days_str.rstrip(", ")

    # Combine all strings
    all_stats_str = f"{mean_last_30_days_str}\n{median_last_30_days_str}\n{mode_last_30_days_str}"
    return all_stats_str



def get_pdf_text(pdf_docs):
    loader = PyPDFLoader(pdf_docs)
    docs = loader.load()
    return docs


def get_text_chunks(pdf_docs):
    text=get_pdf_text(pdf_docs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vector_ollama(pdf):
    doc_chunks= get_text_chunks(pdf)
    embeddings = OllamaEmbeddings()
    vector_store = FAISS.from_documents(doc_chunks, embedding=embeddings)
    # vector_store.save_local("faiss_index")
    return vector_store

def get_vector_google(pdf):
    doc_chunks= get_text_chunks(pdf)
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_documents(doc_chunks, embedding=embeddings)
    return vector_store


def get_deficiencies(pdf):
    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert pathologist.Analyse the report of the patient given as context
    and identify the potential deficiencies, excesses and diseases or health conditions in the key nutrients.                                               
    <context>
    {context}
    </context> 
    User Query :{input}
    """)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                             temperature=0)
    document_chain= create_stuff_documents_chain(llm, prompt_template)
    vector_store = get_vector_ollama(pdf)
    retriever=vector_store.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    # response = retriever_chain.invoke()
    #if question is mandatory the pass find defieciencies in the blood report
    response = retriever_chain.invoke({'input': 'Give me the analysis of the report including deficiencies and excesses in the blood report.Also provide information about any diseases or health conditions in the report.'})
    return response['answer']

def format_age(age):
      if age is not None:
        return f"age: {age} years"
      return ""

def format_height(height_ft, height_in):
      if height_ft is not None and height_in is not None:
        height_cm = int(height_ft * 30.48 + height_in * 2.54)
        return f"height: {height_cm} cm"
      return ""

def format_weight(weight):
      if weight is not None:
        return f"weight: {weight} kgs"
      return ""

def get_gemini_repsonse(input):
    model=genai.GenerativeModel('gemini-1.5-flash')
    prompt="""You are an expert dietitian.
   With the input provided from user about his body health, details, food preferences and 
   other details, generate a 7 day diet plan. 
   Think step by step before providing a detailed answer. 
   Create a personalized diet plan for a user with the input information:"""
    
    response=model.generate_content([prompt,input])
    return response.text

def main():
    prompt1="The past meal history of the patient is as follows: \n"
    prompt1+="Per meal statics are give below: \n"
    prompt1+=cal_meal()
    prompt1+="\n Per day statics are give below: \n"
    prompt1+=cal_day()
    st.set_page_config(page_title="AI Diet Planner | AI NUTRIONIST AND DIETITIAN")
    st.title("Your Personalized AI Diet Planner 👨‍⚕️")
    st.write("RAG Diet Planner redefines personalized nutrition. Upload your blood tests, share your details, and receive a custom diet plan tailored to your unique needs, empowering you to achieve your health goals effectively.")


    # User Information
    st.header("Tell us about yourself:")

    age = st.number_input("Age:", min_value=18)
    gender = st.selectbox("Gender:", ["Male", "Female", "Non-binary", "Prefer not to say"])
    height_ft = st.number_input("Height (ft):", min_value=3)
    height_in = st.number_input("Height (in):", min_value=0, max_value=11)
    weight = st.number_input("Weight (kgs):")

    activity_level = st.selectbox("Activity Level:", [
        "Sedentary (little to no exercise)",
        "Lightly Active (light exercise 1-3 days/week)",
        "Moderately Active (moderate exercise 3-5 days/week)",
        "Highly Active (hard exercise 6-7 days/week)",
        "Very Highly Active (very hard exercise & physical job)"
    ])

    medical_conditions = st.text_area("Do you have any medical conditions or allergies that may affect your diet?")

    # Dietary Preferences and Goals
    st.header("Your preferences and goals:")

    diet_choices = st.multiselect("Select any dietary preferences (optional):", [
        "Vegetarian",
        "Vegan",
        "Pescatarian",
        "Gluten-Free",
        "Dairy-Free",
        "Nut-Free",
        "No restrictions"
    ])

    disliked_foods = st.text_area("Are there any foods you particularly dislike?")

    diet_goal =  st.text_area("What is your primary diet goal?")


    # Additional Information (optional)
    st.header("Additional Information (optional):")

    additional_notes = st.text_area("Any additional notes or preferences you would like to share?")

    # Submit Button
    submit_button = st.button("Generate My Diet Plan")
    
    prompt_part3 = "User Informatiom:\n"
    prompt_part3 += format_age(age) + "\n"
    prompt_part3 += format_height(height_ft, height_in) + "\n"
    prompt_part3 += format_weight(weight) + "\n"
    if(activity_level!=None):
      prompt_part3 +="activity level: "+activity_level + "\n"
    if(medical_conditions != None):
        prompt_part3 += "medical conditions: " + medical_conditions + "\n"
    if(diet_choices != None):
      prompt_part3 += "dietary preferences: " + "\n".join(diet_choices) + "\n"
    if(disliked_foods != None):
      prompt_part3 += "disliked foods: " + disliked_foods + "\n"
    if(diet_goal != None):
        prompt_part3 += "diet goal: " + diet_goal + "\n"
    if additional_notes != None:
        prompt_part3 += "additional notes: " + additional_notes + "\n"



    prompt_part2=""
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your Report File and Click on the Submit & Process Button", type=["pdf"])
        if st.button("Submit & Process"):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(pdf_docs.read())
                tmp_file_name = tmp.name
            deficiencies = get_deficiencies(tmp_file_name)
            # deficiencies = get_deficiencies(pdf_docs)
            st.write("Upload suffessfully processed")
            prompt_part2=deficiencies
    print(prompt_part2)
    print("    c-------------c")
    print(prompt_part3)
    input_prompt = "These are user details and preferences"+prompt_part3 +".\n" 
    if prompt_part2 != "":
        input_prompt+= "The test report of the user contains"+prompt_part2

    if submit_button:
      ans=get_gemini_repsonse(input_prompt)
      st.header("Here's your Diet plan:")
      st.write(ans)
      
if __name__ == "__main__":
    from_mongo_to_csv()
    main()