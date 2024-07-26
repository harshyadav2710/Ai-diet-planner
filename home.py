import streamlit as st
import subprocess
# Function to display box with heading and text
def display_box(heading, text):
    st.write(f"## {heading}")
    st.write(text)

# Main function to create Streamlit app
def main():        
    st.set_page_config(page_title="HOME PAGE | AI NUTRIONIST AND DIETITIAN")
    st.title("AI NUTRIONIST AND DIETITIAN 🧑‍⚕️")
    col1, col2, col3 = st.columns([4,4,4])

    # Box 1 in column 1
    with col1.expander("",expanded=True):
         str='''"NutriScan: Capture, Analyze, Thrive. Effortlessly track your meals with AI-powered image analysis, saving detailed nutrient content to empower your health journey.
         "'''
         display_box("Introducing NutriScan: Your Meal Companion", str)
         if st.button('NutriScan'):
            subprocess.run(["streamlit", "run", "app2.py"])
    # Box 2 in column 2
    with col2.expander("",expanded=True):
        str='''"Packaged Prodigy simplifies understanding packaged foods. Snap a pic of the ingredients, add details like quantity, and receive enlightening insights instantly, helping you make informed choices."'''
        display_box("Packaged Prodigy: Enlightened Insights on Packaged Foods", str)
        if st.button('Packaged Prodigy'):
            subprocess.run(["streamlit", "run", "app.py"])
    # Box 3 in column 3
    with col3.expander("",expanded=True):
        str='''"AI Diet Planner redefines personalized nutrition. Upload your blood tests, share your details, and receive a custom diet plan tailored to your unique needs, empowering you to achieve your health goals effectively."'''
        display_box("AI Diet Planner: Personalized Nutrition, Redefined",str)
        if st.button('AI Diet Planner'):
            subprocess.run(["streamlit", "run", "dietplan.py"])    


if __name__ == "__main__":
    main()