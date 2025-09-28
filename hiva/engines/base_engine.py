from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import time
import hashlib
import json
import os


class BaseLLMEngine(ABC):
    """Base LLM Engine Abstract Class"""
    
    def __init__(self, 
                 model_name: str,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 cache_enabled: bool = True,
                 cache_dir: str = ".cache"):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir
        
        if cache_enabled:
            os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate cache key"""
        cache_content = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs
        }
        cache_str = json.dumps(cache_content, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """Load result from cache"""
        if not self.cache_enabled:
            return None
        
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    return cache_data.get("response")
            except Exception:
                pass
        return None
    
    def _save_to_cache(self, cache_key: str, response: str):
        """Save result to cache"""
        if not self.cache_enabled:
            return
        
        cache_path = self._get_cache_path(cache_key)
        try:
            cache_data = {
                "response": response,
                "timestamp": time.time()
            }
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    @abstractmethod
    def _generate_impl(self, 
                      messages: List[Dict[str, str]], 
                      **kwargs) -> str:
        """Actual generation implementation (to be implemented by subclasses)"""
        pass
    
    def generate(self, 
                content: str, 
                system_prompt: Optional[str] = None,
                **kwargs) -> str:
        """Generate response"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": content})
        
        # Check cache
        cache_key = self._get_cache_key(messages, **kwargs)
        cached_response = self._load_from_cache(cache_key)
        if cached_response:
            return cached_response
        
        # Generate new response
        response = self._generate_impl(messages, **kwargs)
        
        # Save to cache
        self._save_to_cache(cache_key, response)
        
        return response
    
    def generate_with_messages(self, 
                              messages: List[Dict[str, str]], 
                              **kwargs) -> str:
        """Generate response using complete message list"""
        # Check cache
        cache_key = self._get_cache_key(messages, **kwargs)
        cached_response = self._load_from_cache(cache_key)
        if cached_response:
            return cached_response
        
        # Generate new response
        response = self._generate_impl(messages, **kwargs)
        
        # Save to cache
        self._save_to_cache(cache_key, response)
        
        return response
    
    def clear_cache(self):
        """Clear cache"""
        if self.cache_enabled and os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, filename))
    
    def set_temperature(self, temperature: float):
        """Set temperature parameter"""
        self.temperature = temperature
    
    def set_max_tokens(self, max_tokens: Optional[int]):
        """Set maximum number of tokens"""
        self.max_tokens = max_tokens
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.model_name})"
    
    def __repr__(self):
        return self.__str__()

    EVALUATION_PROMPT_TEMPLATE = """You are an expert evaluator. Analyze the given content and provide structured feedback.

Content to evaluate: {content}
Initial instruction: {context}

Evaluation criteria: {criteria}

Provide your evaluation in the following format:
<FEEDBACK>
[Your detailed analysis here. Identify specific strengths and weaknesses. Be precise and actionable.]
</FEEDBACK>"""

    GRADIENT_PROMPT_TEMPLATE = """You are a gradient engine providing textual gradients for optimization.

Variable to improve: {variable}
Role: {role}
Current feedback: {feedback}

Analyze the variable and feedback to provide specific improvement suggestions.

<FEEDBACK>
Based on the analysis, here are specific areas for improvement:
1. [Specific issue/weakness identified]
2. [Another specific issue if applicable]
3. [Additional improvement areas]

Recommendations:
- [Specific actionable suggestion]
- [Another concrete improvement]
- [Additional recommendations]
</FEEDBACK>"""

    IMPROVEMENT_PROMPT_TEMPLATE = """You are an optimizer that applies textual gradients to improve variables.

Current variable: {variable}
Role: {role}
Gradients/Feedback: {gradients}

Apply the gradients to improve the variable. Keep the same role and purpose but incorporate the feedback.

<IMPROVED_VARIABLE>
[Your improved version here. Apply the specific suggestions from the gradients while maintaining the variable's core purpose and format.]
</IMPROVED_VARIABLE>"""

    PREDECESSOR_FEEDBACK_TEMPLATE = """You are providing feedback to a predecessor variable based on successor feedback.

Predecessor variable: {predecessor}
Successor variable: {successor}
Successor feedback: {successor_feedback}
Connection context: {context}

Provide feedback to help improve the predecessor variable to better support the successor.

<FEEDBACK>
To improve the predecessor variable "{predecessor}" and better support the successor:

1. [Specific suggestion for predecessor improvement]
2. [How predecessor affects successor quality]
3. [Actionable changes to make]

The predecessor should be modified to: [clear directive]
</FEEDBACK>""" 