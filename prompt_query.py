# modal_functions.py (or prompt_query.py)
import modal
import google.generativeai as genai
import os
import json
import re
# Remove: import streamlit as st  <--- This line should be removed

# Initialize a Modal App
app = modal.App("procurement-parser")

# Define the image for your Modal function, including necessary pip installs
modal_image = modal.Image.debian_slim().pip_install("google-generativeai")

@app.function(
    image=modal_image,
    # Make sure your Gemini API key is securely stored as a Modal secret
    secrets=[modal.Secret.from_name("my-gemini-secret")],
    timeout=120 # Adjust timeout as needed
)
def get_dictionary_from_prompt(user_prompt: str) -> dict:
    """
    Parses a user's request and converts it into a structured dictionary
    of product specifications using the Gemini API. This function runs on Modal.
    """
    try:
        # Configure Gemini API using the secret (환경 변수에서 읽어옴)
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
    except Exception as e:
        # In a Modal function, print statements go to Modal logs
        print(f"Error: Could not configure Gemini API. Details: {e}")
        return {}

    extraction_prompt = f"""
        You are an expert data extraction AI. Your task is to parse a user's
        request and convert it into a structured JSON object. The JSON object
        should be a dictionary where each key is the product name (string) and
        its value is a dictionary of that product's specifications.

        EXAMPLE
        User Prompt: "Silicon wafers, purity 99.999%, diameter 6 inches"
        Correct Output:
        json
        {{
          "Silicon wafers": {{
            "purity": "99.999%",
            "diameter": "6 inches"
          }}
        }}
        

        YOUR TASK
        User Prompt: "{user_prompt}"

        Respond ONLY with the single, valid JSON object.
    """

    try:
        response = model.generate_content(extraction_prompt)
        # Change this line: Use print() for debugging within Modal
        
        # Assuming the model reliably outputs JSON within curly braces
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
    # If match.group(0) is already a dictionary, no parsing is needed.
    # Otherwise, you would assign the dictionary directly.
            parsed_dict = match.group(0) 
            print(f"Successfully processed dictionary: {parsed_dict}") # Log success
            return parsed_dict
        else:
            print("Error: No valid dictionary object found in the model's response. Full response:")
            print(response.text) # Print full response if no dictionary found
            return {}
    except Exception as e:
        print(f"An error occurred during API call or parsing: {e}")
        return {}

# This block is for deploying the Modal app if you run this file directly
# (e.g., `modal deploy modal_functions.py`)
# if __name__ == "__main__":
#     @app.local_entrypoint()
#     def main():
#         # You can test the function locally via Modal CLI with this entrypoint
#         test_prompt = "I need high-purity gold nanoparticles, 20nm size, spherical shape, for biomedical applications."
#         result = get_dictionary_from_prompt.remote(test_prompt)
#         print("Test Result:", result)