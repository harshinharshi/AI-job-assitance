"""
Large Language Model (LLM) Configuration Module

This module provides a unified interface for loading and configuring different
LLM providers including OpenAI, Groq, and local Llama models. It supports
multiple model configurations with appropriate parameters for each provider.

Supported Providers:
    - OpenAI (GPT-4)
    - Groq (Llama models)
    - Local Llama (via Ollama)

Environment Variables Required:
    - OPENAI_API_KEY: For OpenAI models
    - GROQ_API_KEY: For Groq models
    - LLM_NAME: Specifies which provider to use ('openai', 'groq', 'llama3')
"""

import os
from typing import Union, Optional
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq


# Model configuration constants
OPENAI_MODELS = {
    'gpt-4': 'gpt-4-0125-preview',
    'gpt-4-turbo': 'gpt-4-turbo-2024-04-09'
}

GROQ_MODELS = {
    'llama3-70b': 'llama3-70b-8192',
    'mixtral': 'mixtral-8x7b-32768'
}

# Default model parameters
DEFAULT_TEMPERATURES = {
    'openai': 0.1,
    'groq': 0.2,
    'llama3': 0.0
}

# Local Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434/v1"


def load_llm(llm_name: str) -> Union[ChatOpenAI, ChatGroq]:
    """
    Load and configure a Language Learning Model based on the specified provider.
    
    This function initializes different LLM providers with appropriate configurations
    including API keys, model names, and parameters optimized for each provider.
    
    Args:
        llm_name (str): Name of the LLM provider to load. 
                       Options: 'openai', 'groq', 'llama3'
                       
    Returns:
        Union[ChatOpenAI, ChatGroq]: Configured LLM instance ready for use
        
    Raises:
        ValueError: If unsupported LLM name is provided
        KeyError: If required environment variables are missing
        
    Example:
        >>> llm = load_llm('openai')
        >>> response = llm.invoke("Hello, world!")
    """
    
    print(f"Initializing LLM provider: {llm_name}")
    
    if llm_name.lower() == 'openai':
        return _load_openai_model()
    elif llm_name.lower() == 'groq':
        return _load_groq_model()
    elif llm_name.lower() == 'llama3':
        return _load_local_llama_model()
    else:
        raise ValueError(
            f"Unsupported LLM name: {llm_name}. "
            f"Supported options: {list(DEFAULT_TEMPERATURES.keys())}"
        )


def _load_openai_model() -> ChatOpenAI:
    """
    Load and configure OpenAI GPT-4 model.
    
    Returns:
        ChatOpenAI: Configured OpenAI model instance
        
    Raises:
        KeyError: If OPENAI_API_KEY environment variable is not set
    """
    try:
        api_key = os.environ["OPENAI_API_KEY"]
        
        # Initialize OpenAI model with optimized parameters
        llm = ChatOpenAI(
            model_name=OPENAI_MODELS['gpt-4'],
            openai_api_key=api_key,
            temperature=DEFAULT_TEMPERATURES['openai'],
            streaming=True,  # Enable streaming for better UX
            max_tokens=None,  # No token limit
            timeout=60  # 60 second timeout
        )
        
        print("✅ OpenAI GPT-4 model loaded successfully")
        return llm
        
    except KeyError:
        raise KeyError(
            "OPENAI_API_KEY environment variable is required for OpenAI models. "
            "Please set it in your .env file or environment."
        )
    except Exception as e:
        print(f"❌ Error loading OpenAI model: {str(e)}")
        raise


def _load_groq_model() -> ChatGroq:
    """
    Load and configure Groq Llama model.
    
    Returns:
        ChatGroq: Configured Groq model instance
        
    Raises:
        KeyError: If GROQ_API_KEY environment variable is not set
    """
    try:
        api_key = os.environ["GROQ_API_KEY"]
        
        # Initialize Groq model with optimized parameters
        llm = ChatGroq(
            temperature=DEFAULT_TEMPERATURES['groq'],
            groq_api_key=api_key,
            model_name=GROQ_MODELS['llama3-70b'],  # Use Llama3 70B by default
            max_tokens=None,
            timeout=60
        )
        
        print("✅ Groq Llama3-70B model loaded successfully")
        print("⚠️  Note: Groq models may be unstable due to routing/token limit issues")
        return llm
        
    except KeyError:
        raise KeyError(
            "GROQ_API_KEY environment variable is required for Groq models. "
            "Please set it in your .env file or environment."
        )
    except Exception as e:
        print(f"❌ Error loading Groq model: {str(e)}")
        raise


def _load_local_llama_model() -> ChatOpenAI:
    """
    Load and configure local Llama model via Ollama.
    
    This function configures a local Llama model running through Ollama,
    which provides an OpenAI-compatible API interface.
    
    Returns:
        ChatOpenAI: Configured local Llama model instance
        
    Note:
        Requires Ollama to be running locally on port 11434
    """
    try:
        # Initialize local Llama model via Ollama
        llm = ChatOpenAI(
            model="llama3",
            base_url=OLLAMA_BASE_URL,
            temperature=DEFAULT_TEMPERATURES['llama3'],
            api_key="ollama",  # Dummy API key for local usage
            timeout=120  # Longer timeout for local models
        )
        
        print("✅ Local Llama3 model loaded successfully")
        print(f"🔗 Using Ollama endpoint: {OLLAMA_BASE_URL}")
        return llm
        
    except Exception as e:
        print(f"❌ Error loading local Llama model: {str(e)}")
        print("💡 Make sure Ollama is running locally on http://localhost:11434")
        raise


# Utility functions for model management

def get_available_models() -> dict:
    """
    Get information about all available models and their configurations.
    
    Returns:
        dict: Dictionary containing model information for each provider
    """
    return {
        'openai': {
            'models': OPENAI_MODELS,
            'default_temperature': DEFAULT_TEMPERATURES['openai'],
            'requires_api_key': True,
            'env_var': 'OPENAI_API_KEY'
        },
        'groq': {
            'models': GROQ_MODELS,
            'default_temperature': DEFAULT_TEMPERATURES['groq'],
            'requires_api_key': True,
            'env_var': 'GROQ_API_KEY'
        },
        'llama3': {
            'models': {'llama3': 'llama3'},
            'default_temperature': DEFAULT_TEMPERATURES['llama3'],
            'requires_api_key': False,
            'base_url': OLLAMA_BASE_URL
        }
    }


def validate_environment(llm_name: str) -> dict:
    """
    Validate that required environment variables are set for the specified LLM.
    
    Args:
        llm_name (str): Name of the LLM provider to validate
        
    Returns:
        dict: Validation results including status and missing variables
    """
    validation_result = {
        'valid': True,
        'missing_vars': [],
        'warnings': []
    }
    
    if llm_name.lower() == 'openai':
        if 'OPENAI_API_KEY' not in os.environ:
            validation_result['valid'] = False
            validation_result['missing_vars'].append('OPENAI_API_KEY')
    
    elif llm_name.lower() == 'groq':
        if 'GROQ_API_KEY' not in os.environ:
            validation_result['valid'] = False
            validation_result['missing_vars'].append('GROQ_API_KEY')
        validation_result['warnings'].append('Groq models may be unstable')
    
    elif llm_name.lower() == 'llama3':
        validation_result['warnings'].append('Requires Ollama running on localhost:11434')
    
    else:
        validation_result['valid'] = False
        validation_result['missing_vars'].append(f'Unsupported LLM: {llm_name}')
    
    return validation_result


def get_model_info(llm_name: str) -> Optional[dict]:
    """
    Get detailed information about a specific model provider.
    
    Args:
        llm_name (str): Name of the LLM provider
        
    Returns:
        Optional[dict]: Model information or None if not found
    """
    available_models = get_available_models()
    return available_models.get(llm_name.lower())


# Model switching and configuration management

def switch_model_config(llm_name: str, **kwargs) -> Union[ChatOpenAI, ChatGroq]:
    """
    Load a model with custom configuration parameters.
    
    Args:
        llm_name (str): Name of the LLM provider
        **kwargs: Custom configuration parameters
        
    Returns:
        Union[ChatOpenAI, ChatGroq]: Configured LLM instance
    """
    print(f"Loading {llm_name} with custom config: {kwargs}")
    
    # Override default parameters with custom ones
    custom_temp = kwargs.get('temperature', DEFAULT_TEMPERATURES.get(llm_name, 0.1))
    
    if llm_name.lower() == 'openai':
        return ChatOpenAI(
            model_name=kwargs.get('model', OPENAI_MODELS['gpt-4']),
            openai_api_key=os.environ["OPENAI_API_KEY"],
            temperature=custom_temp,
            **{k: v for k, v in kwargs.items() if k not in ['model', 'temperature']}
        )
    
    elif llm_name.lower() == 'groq':
        return ChatGroq(
            model_name=kwargs.get('model', GROQ_MODELS['llama3-70b']),
            groq_api_key=os.environ["GROQ_API_KEY"],
            temperature=custom_temp,
            **{k: v for k, v in kwargs.items() if k not in ['model', 'temperature']}
        )
    
    elif llm_name.lower() == 'llama3':
        return ChatOpenAI(
            model=kwargs.get('model', 'llama3'),
            base_url=kwargs.get('base_url', OLLAMA_BASE_URL),
            temperature=custom_temp,
            **{k: v for k, v in kwargs.items() if k not in ['model', 'base_url', 'temperature']}
        )
    
    else:
        raise ValueError(f"Unsupported LLM name: {llm_name}")