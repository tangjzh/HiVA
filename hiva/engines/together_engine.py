import os
import random
from typing import Dict, List, Optional, Union
import time
from .base_engine import BaseLLMEngine

try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    Together = None
    TOGETHER_AVAILABLE = False


class TogetherEngine(BaseLLMEngine):
    """Together AI LLM engine that supports API key pool and random sampling strategy"""
    
    def __init__(self,
                 model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                 api_keys: Optional[Union[str, List[str]]] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 cache_enabled: bool = True,
                 cache_dir: str = ".cache",
                 timeout: float = 30.0,
                 max_retries: int = 3,
                 retry_different_key: bool = True):
        
        if not TOGETHER_AVAILABLE:
            raise ImportError("Together package is not installed. Please install it with: pip install together")
        
        super().__init__(model_name, temperature, max_tokens, cache_enabled, cache_dir)
        
        # Set up API key pool
        self.api_keys = self._setup_api_keys(api_keys)
        if not self.api_keys:
            raise ValueError("No valid Together API keys found. Please set TOGETHER_API_KEY environment variable or pass api_keys parameter.")
        
        # Set base_url (for supporting other services compatible with Together API)
        self.base_url = base_url or os.getenv("TOGETHER_BASE_URL")
        
        # Create client pool
        self.clients = []
        self._create_clients()
        
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_different_key = retry_different_key
        
        # Track failed API keys
        self.failed_keys = set()
        
        # Validate if model is supported
        self._validate_model()
    
    def _setup_api_keys(self, api_keys: Optional[Union[str, List[str]]]) -> List[str]:
        """Set up API key pool"""
        keys = []
        
        if api_keys:
            if isinstance(api_keys, str):
                keys.append(api_keys)
            elif isinstance(api_keys, list):
                keys.extend(api_keys)
        
        # Get API key from environment variable
        env_key = os.getenv("TOGETHER_API_KEY")
        if env_key and env_key not in keys:
            keys.append(env_key)
        
        # Support multiple environment variables (TOGETHER_API_KEY_1, TOGETHER_API_KEY_2, ...)
        i = 1
        while True:
            env_key = os.getenv(f"TOGETHER_API_KEY_{i}")
            if env_key and env_key not in keys:
                keys.append(env_key)
                i += 1
            else:
                break
        
        return keys
    
    def _create_clients(self):
        """Create client pool"""
        self.clients = []
        for api_key in self.api_keys:
            client_kwargs = {"api_key": api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            
            try:
                client = Together(**client_kwargs)
                self.clients.append({
                    "client": client,
                    "api_key": api_key,
                    "active": True
                })
            except Exception as e:
                print(f"Warning: Failed to create client for API key {api_key[:8]}...: {e}")
    
    def _get_random_client(self):
        """Randomly select an available client"""
        active_clients = [c for c in self.clients if c["active"] and c["api_key"] not in self.failed_keys]
        
        if not active_clients:
            # If all clients failed, reset failure records and retry
            print("All API keys failed, resetting failed keys and retrying...")
            self.failed_keys.clear()
            active_clients = [c for c in self.clients if c["active"]]
        
        if not active_clients:
            raise ValueError("No active Together API clients available")
        
        return random.choice(active_clients)
    
    def _validate_model(self):
        """Validate if the model is supported"""
        supported_models = [
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "meta-llama/Llama-3-70b-chat-hf",
            "meta-llama/Llama-3-8b-chat-hf",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "microsoft/DialoGPT-medium",
            "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
            "togethercomputer/RedPajama-INCITE-7B-Chat",
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
            "deepseek-ai/deepseek-coder-33b-instruct",
            "codellama/CodeLlama-34b-Instruct-hf",
            "WizardLM/WizardCoder-Python-34B-V1.0"
        ]
        
        # Check if it's a supported model or starts with a supported prefix
        is_supported = any(
            self.model_name == model or 
            self.model_name.startswith(model.split('/')[0]) or
            any(prefix in self.model_name for prefix in ["meta-llama", "mistralai", "Qwen", "togethercomputer", "NousResearch", "deepseek-ai", "codellama", "WizardLM"])
            for model in supported_models
        )
        
        if not is_supported:
            print(f"Warning: Model '{self.model_name}' may not be supported. Supported models include: {supported_models[:5]}...")
    
    def _generate_impl(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Actual Together API call implementation"""
        
        # Prepare API parameters
        api_params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        
        if self.max_tokens:
            api_params["max_tokens"] = self.max_tokens
        
        # Add additional parameters
        api_params.update(kwargs)
        
        # Retry mechanism
        last_exception = None
        attempt = 0
        used_keys = set()
        
        while attempt < self.max_retries:
            try:
                # Get random client
                client_info = self._get_random_client()
                client = client_info["client"]
                api_key = client_info["api_key"]
                
                # Skip if retry with different key is enabled and this key has been used
                if self.retry_different_key and attempt > 0 and api_key in used_keys:
                    # Look for unused keys
                    available_clients = [c for c in self.clients 
                                       if c["active"] and 
                                       c["api_key"] not in self.failed_keys and 
                                       c["api_key"] not in used_keys]
                    if available_clients:
                        client_info = random.choice(available_clients)
                        client = client_info["client"]
                        api_key = client_info["api_key"]
                
                used_keys.add(api_key)
                
                print(f"Attempt {attempt + 1}/{self.max_retries}: Using API key {api_key[:8]}...")
                
                response = client.chat.completions.create(**api_params)
                
                if response.choices and response.choices[0].message:
                    return response.choices[0].message.content.strip()
                else:
                    raise ValueError("Empty response from Together API")
                    
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # Check if error is API key related
                if any(keyword in error_msg for keyword in ["api key", "authentication", "unauthorized", "invalid key", "quota", "rate limit"]):
                    print(f"API key related error with {api_key[:8]}...: {e}")
                    self.failed_keys.add(api_key)
                    # Mark client as inactive
                    for client_info in self.clients:
                        if client_info["api_key"] == api_key:
                            client_info["active"] = False
                            break
                
                attempt += 1
                if attempt < self.max_retries:
                    wait_time = 2 ** (attempt - 1)  # Exponential backoff
                    print(f"Together API call failed (attempt {attempt}/{self.max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"All Together API call attempts failed. Last error: {e}")
        
        # If all retries failed, raise the last exception
        raise last_exception
    
    def add_api_key(self, api_key: str):
        """Add a new API key"""
        if api_key not in self.api_keys:
            self.api_keys.append(api_key)
            
            client_kwargs = {"api_key": api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            
            try:
                client = Together(**client_kwargs)
                self.clients.append({
                    "client": client,
                    "api_key": api_key,
                    "active": True
                })
                print(f"Successfully added API key {api_key[:8]}...")
            except Exception as e:
                print(f"Failed to add API key {api_key[:8]}...: {e}")
    
    def remove_api_key(self, api_key: str):
        """Remove an API key"""
        if api_key in self.api_keys:
            self.api_keys.remove(api_key)
            self.clients = [c for c in self.clients if c["api_key"] != api_key]
            self.failed_keys.discard(api_key)
            print(f"Removed API key {api_key[:8]}...")
    
    def set_base_url(self, base_url: str):
        """Set a new base URL"""
        self.base_url = base_url
        self._create_clients()
    
    def reset_failed_keys(self):
        """Reset failed API keys"""
        self.failed_keys.clear()
        for client_info in self.clients:
            client_info["active"] = True
        print("Reset all failed API keys")
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_different_key": self.retry_different_key,
            "total_api_keys": len(self.api_keys),
            "active_api_keys": len([c for c in self.clients if c["active"]]),
            "failed_api_keys": len(self.failed_keys)
        }
    
    def get_api_key_status(self) -> Dict:
        """Get API key status"""
        status = {}
        for client_info in self.clients:
            key = client_info["api_key"]
            status[f"{key[:8]}..."] = {
                "active": client_info["active"],
                "failed": key in self.failed_keys
            }
        return status 