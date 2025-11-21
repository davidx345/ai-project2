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
    Sends the prompt to an LLM API.
    Defaults to Hugging Face Inference API (GPT-2 or similar small model for demo)
    if an API key is provided, otherwise returns a mock response.
    """
    
    # You can replace this with any LLM API endpoint (OpenAI, Cohere, Groq, etc.)
    # Using Hugging Face Inference API as an example
    API_URL = "https://api-inference.huggingface.co/models/gpt2"
    
    if not api_key:
        # Check environment variable
        api_key = os.getenv("HF_API_KEY")

    if not api_key:
        return "Error: No API Key provided. Please set HF_API_KEY in .env or provide it. (Mock Response: This is a simulated answer.)"

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": prompt}

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Parsing response depends on the specific API
        if isinstance(data, list) and 'generated_text' in data[0]:
            return data[0]['generated_text']
        elif 'error' in data:
            return f"API Error: {data['error']}"
        else:
            return str(data)
            
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
