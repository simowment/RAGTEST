#!/usr/bin/env python3
"""
Manages OpenRouter LLM instances, API keys, and model fallback logic.
Cycles through key/model combinations to handle rate limits and adds intelligent backoff.
"""

import os
import time
from datetime import datetime
from itertools import cycle
from dotenv import load_dotenv
from llama_index.llms.openrouter import OpenRouter
from openai import RateLimitError

# Load environment variables from .env file
load_dotenv()

class LLMManager:
    """
    Manages OpenRouter LLM instances by cycling through API keys and models.
    Implements a backoff strategy for rate limit errors.
    """

    def __init__(self, llm_settings=None):
        self.api_keys = self._load_api_keys()
        self.models = self._load_models()
        self.llm_settings = llm_settings or {}

        if not self.api_keys:
            raise ValueError("No OpenRouter API keys found. Please set OPENROUTER_API_KEY_1, etc., in your .env file.")
        if not self.models:
            raise ValueError("No OpenRouter models found. Please set OPENROUTER_MODELS in your .env file.")

        # Create a list of all (key, model) combinations to cycle through
        self.configurations = [(key, model) for model in self.models for key in self.api_keys]
        self.config_cycler = cycle(self.configurations)
        self.current_config = next(self.config_cycler)

        print(f"LLM Manager initialized with {len(self.api_keys)} keys and {len(self.models)} models, yielding {len(self.configurations)} unique configurations.")

    def _load_api_keys(self):
        """Loads API keys from environment variables, filtering duplicates and placeholders."""
        keys = []
        i = 1
        while True:
            key_val = os.getenv(f"OPENROUTER_API_KEY_{i}")
            if key_val:
                key = key_val.strip()
                # Ignore placeholder keys
                if "YOUR_" in key.upper():
                    print(f"--> Warning: Skipping placeholder API key found in OPENROUTER_API_KEY_{i}.")
                # Ignore empty or duplicate keys
                elif key and key not in keys:
                    keys.append(key)
                elif key in keys:
                    print(f"--> Warning: Skipping duplicate API key found in OPENROUTER_API_KEY_{i}.")
                i += 1
            else:
                break
        return keys

    def _load_models(self):
        """Loads a comma-separated list of models from the OPENROUTER_MODELS env var."""
        models_str = os.getenv("OPENROUTER_MODELS")
        if models_str:
            return [model.strip() for model in models_str.split(',') if model.strip()]
        
        # Fallback to the old variable for backward compatibility
        model = os.getenv("OPENROUTER_MODEL")
        return [model] if model else []

    def get_llm(self):
        """Returns a new OpenRouter instance with the current configuration."""
        api_key, model_name = self.current_config
        print(f"--> Using LLM config: model='{model_name}', key='...{api_key[-4:]}'")
        
        final_settings = {
            "temperature": 0.1,
            "max_tokens": 4096,
            "context_window": 163840,
            **self.llm_settings,
            "model": model_name,
            "api_key": api_key,
        }
        return OpenRouter(**final_settings)

    def switch_to_next_config(self):
        """Switches to the next available key/model configuration."""
        self.current_config = next(self.config_cycler)
        key, model = self.current_config
        print(f"--> Switched configuration due to an error. Next attempt with model='{model}', key='...{key[-4:]}'")

    def handle_rate_limit(self, e: RateLimitError):
        """
        Parses the RateLimitError to implement a backoff strategy.
        Waits for the duration specified in the 'X-RateLimit-Reset' header,
        or a default duration if the header is not present.
        """
        wait_seconds = 15  # Default wait time in seconds
        try:
            # Attempt to parse the reset time from the error response body
            headers = e.body.get("error", {}).get("metadata", {}).get("headers", {})
            if headers and "X-RateLimit-Reset" in headers:
                reset_ms = int(headers["X-RateLimit-Reset"])
                wait_seconds = max(1, (reset_ms / 1000) - datetime.utcnow().timestamp())
                print(f"--> Rate limit reset time found. Waiting for {wait_seconds:.2f} seconds.")
            else:
                print(f"--> Rate limit reset time not in headers. Using default wait of {wait_seconds}s.")
        except (AttributeError, KeyError, TypeError, ValueError) as parse_error:
            print(f"--> Could not parse rate limit error for backoff time: {parse_error}. Using default wait.")
        
        time.sleep(wait_seconds)

async def managed_chat_request(chat_engine, question, llm_manager):
    """
    A robust, unified function to handle chat requests using the LLMManager.
    It performs retries with intelligent backoff and configuration switching.
    """
    # Limit the number of full-cycle retries to prevent infinite loops
    max_retries = len(llm_manager.configurations)
    
    for attempt in range(max_retries):
        try:
            # Ensure the engine uses the latest LLM from the manager
            chat_engine._llm = llm_manager.get_llm()
            response = await chat_engine.achat(question)
            return {"response": str(response)}
        
        except RateLimitError as e:
            print(f"--> Caught RateLimitError (Attempt {attempt + 1}/{max_retries}).")
            llm_manager.handle_rate_limit(e)
            if attempt < max_retries - 1:
                llm_manager.switch_to_next_config()
            # Loop will continue to the next attempt after the backoff
        
        except Exception as e:
            # Handle other, non-rate-limit exceptions
            print(f"--> An unexpected error occurred: {e}")
            if attempt < max_retries - 1:
                llm_manager.switch_to_next_config()
            # For other errors, we also switch config and retry
    
    # If all retries fail, raise the final error
    raise RateLimitError(f"All {max_retries} API key/model configurations failed after retries.", response=getattr(e, 'response', None), body=getattr(e, 'body', None)) 