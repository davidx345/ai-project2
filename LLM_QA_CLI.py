import string
import requests
import os
import json
from dotenv import load_dotenv

# Load environment variables if available
load_dotenv()

def preprocess_input(text):
    """
    Applies basic preprocessing: lowercasing, punctuation removal, and tokenization.
    Returns a tuple of (processed_text_string, tokens_list).
    """
    # Lowercasing
    text_lower = text.lower()
    
    # Punctuation removal
    text_no_punct = text_lower.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization (simple whitespace split)
    tokens = text_no_punct.split()
    
    return text_no_punct, tokens

def query_llm(prompt, api_key=None):
    """
    Sends the prompt to Google Gemini API.
    """
    
    if not api_key:
        # Check environment variable
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "Error: No API Key provided. Please set GEMINI_API_KEY in .env. (Mock Response: This is a simulated answer.)"

    # Gemini API Endpoint (using gemini-1.5-flash for speed/free tier)
    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Parse Gemini response
        try:
            return data['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            return f"API Error: Unexpected response format. {data}"
            
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"

def main():
    print("--- NLP Question-Answering System (CLI) ---")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        user_input = input("\nEnter your question: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
            
        if not user_input.strip():
            continue

        # Preprocessing
        processed_text, tokens = preprocess_input(user_input)
        print(f"\n[Processed]: {processed_text}")
        print(f"[Tokens]: {tokens}")
        
        # Query LLM
        print("\nQuerying LLM...")
        answer = query_llm(user_input)
        
        print(f"\n[LLM Answer]:\n{answer}")

if __name__ == "__main__":
    main()
