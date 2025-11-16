import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_groq import ChatGroq
from config.config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, RESPONSE_MODES


def get_chatgroq_model(response_mode="Detailed"):
    """
    Initialize and return the Groq chat model.
    
    Args:
        response_mode (str): "Concise" or "Detailed"
    
    Returns:
        ChatGroq: Initialized Groq chat model
    """
    try:
        # Get max tokens based on response mode
        if response_mode in RESPONSE_MODES:
            max_tokens = RESPONSE_MODES[response_mode]["max_tokens"]
        else:
            max_tokens = 2048
        
        # Initialize Groq model
        groq_model = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=GROQ_TEMPERATURE,
            max_tokens=max_tokens
        )
        
        return groq_model
    
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")


def get_response_mode_instruction(response_mode):
    """
    Get the system instruction for a specific response mode.
    
    Args:
        response_mode (str): "Concise" or "Detailed"
    
    Returns:
        str: System instruction
    """
    if response_mode in RESPONSE_MODES:
        return RESPONSE_MODES[response_mode]["system_instruction"]
    else:
        return RESPONSE_MODES["Detailed"]["system_instruction"]
