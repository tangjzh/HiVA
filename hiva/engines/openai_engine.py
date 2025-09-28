import os
from typing import Dict, List, Optional
import time
from .base_engine import BaseLLMEngine

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False


class OpenAIEngine(BaseLLMEngine):
    """OpenAI LLM Engine"""
    
    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 cache_enabled: bool = True,
                 cache_dir: str = ".cache",
                 timeout: float = 30.0,
                 max_retries: int = 3):
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is not installed. Please install it with: pip install openai")
        
        super().__init__(model_name, temperature, max_tokens, cache_enabled, cache_dir)
        
        # Set API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Set base_url (for supporting other services compatible with OpenAI API)
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        # Create client
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        self.client = OpenAI(**client_kwargs)
        
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Validate if model is supported
        # self._validate_model()
    
    def _validate_model(self):
        """Validate if the model is supported"""
        supported_models = [
            "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
            "gpt-3.5-turbo", "gpt-3.5-turbo-16k",
            "gpt-4-1106-preview", "gpt-4-vision-preview"
        ]
        
        # Check if model is supported or starts with a supported prefix
        is_supported = any(
            self.model_name == model or self.model_name.startswith(model)
            for model in supported_models
        )
        
        if not is_supported:
            print(f"Warning: Model '{self.model_name}' may not be supported. Supported models: {supported_models}")
    
    def _generate_impl(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Actual OpenAI API call implementation"""
        
        # Prepare API parameters
        api_params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "timeout": self.timeout
        }
        
        if self.max_tokens:
            api_params["max_tokens"] = self.max_tokens
        
        # Add additional parameters
        api_params.update(kwargs)

        if self.temperature == 1:
            api_params["temperature"] = 1
            
        # Retry mechanism
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**api_params)
                
                if response.choices and response.choices[0].message:
                    return response.choices[0].message.content.strip()
                else:
                    raise ValueError("Empty response from OpenAI API")
                    
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"OpenAI API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"All OpenAI API call attempts failed. Last error: {e}")
        
        # If all retries failed, raise the last exception
        raise last_exception
    
    def set_api_key(self, api_key: str):
        """Set new API key"""
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)
    
    def set_base_url(self, base_url: str):
        """Set new base URL"""
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        } 