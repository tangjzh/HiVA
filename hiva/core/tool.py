"""
Tool System Module
Manages tool function definition, execution, update and validation for agents
"""

import inspect
import json
import re
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class ToolType(Enum):
    """Tool type enumeration"""
    PYTHON_FUNCTION = "python_function"
    API_CALL = "api_call"
    DATA_PROCESSING = "data_processing"
    TEXT_ANALYSIS = "text_analysis"
    MATHEMATICAL = "mathematical"
    FILE_OPERATION = "file_operation"
    CUSTOM = "custom"


@dataclass
class ToolExecutionResult:
    """Tool execution result"""
    success: bool
    result: Any
    error_message: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    metadata: Dict[str, Any] = None


@dataclass
class ToolDefinition:
    """Tool definition"""
    name: str
    description: str
    tool_type: ToolType
    function: Callable
    parameters: Dict[str, Any]
    return_type: str
    examples: List[str]
    safety_level: str = "safe"  # safe, moderate, restricted
    

class ToolValidator:
    """Tool validator"""
    
    @staticmethod
    def validate_python_code(code: str) -> bool:
        """Validate Python code safety"""
        # Check for dangerous imports and function calls
        dangerous_patterns = [
            r'import\s+os',
            r'import\s+sys',
            r'import\s+subprocess',
            r'exec\s*\(',
            r'eval\s*\(',
            r'__import__',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False
        
        # Try to compile code
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False
    
    @staticmethod
    def validate_function_signature(func: Callable, expected_params: List[str]) -> bool:
        """Validate function signature"""
        try:
            sig = inspect.signature(func)
            actual_params = list(sig.parameters.keys())
            return set(expected_params).issubset(set(actual_params))
        except Exception:
            return False


class ToolExecutor:
    """Tool executor"""
    
    def __init__(self, max_retries: int = 3, timeout: float = 30.0):
        self.max_retries = max_retries
        self.timeout = timeout
        self.execution_history = []
    
    def execute_tool(self, tool_def: ToolDefinition, input_data: str, **kwargs) -> ToolExecutionResult:
        """Execute tool function"""
        start_time = time.time()
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                # Preprocess input data
                processed_input = self._preprocess_input(input_data, tool_def)
                
                # Execute tool function
                if tool_def.tool_type == ToolType.PYTHON_FUNCTION:
                    result = self._execute_python_function(tool_def, processed_input, **kwargs)
                else:
                    result = self._execute_generic_function(tool_def, processed_input, **kwargs)
                
                execution_time = time.time() - start_time
                
                # Postprocess result
                processed_result = self._postprocess_result(result, tool_def)
                
                execution_result = ToolExecutionResult(
                    success=True,
                    result=processed_result,
                    execution_time=execution_time,
                    retry_count=retry_count,
                    metadata={
                        "tool_name": tool_def.name,
                        "tool_type": tool_def.tool_type.value,
                        "input_length": len(str(input_data))
                    }
                )
                
                self.execution_history.append(execution_result)
                return execution_result
                
            except Exception as e:
                retry_count += 1
                error_message = f"Tool execution error (attempt {retry_count}/{self.max_retries + 1}): {str(e)}"
                
                if retry_count > self.max_retries:
                    execution_time = time.time() - start_time
                    execution_result = ToolExecutionResult(
                        success=False,
                        result=None,
                        error_message=error_message,
                        execution_time=execution_time,
                        retry_count=retry_count - 1,
                        metadata={
                            "tool_name": tool_def.name,
                            "error_trace": traceback.format_exc()
                        }
                    )
                    
                    self.execution_history.append(execution_result)
                    return execution_result
                
                # Wait before retry
                time.sleep(0.5 * retry_count)
    
    def _preprocess_input(self, input_data: str, tool_def: ToolDefinition) -> Any:
        """Preprocess input data"""
        if tool_def.tool_type == ToolType.MATHEMATICAL:
            # Extract numbers and mathematical expressions
            numbers = re.findall(r'-?\d+\.?\d*', input_data)
            return {'numbers': [float(n) for n in numbers], 'text': input_data}
        elif tool_def.tool_type == ToolType.TEXT_ANALYSIS:
            return {'text': input_data, 'length': len(input_data)}
        else:
            return input_data
    
    def _execute_python_function(self, tool_def: ToolDefinition, input_data: Any, **kwargs) -> Any:
        """Execute Python function"""
        return tool_def.function(input_data, **kwargs)
    
    def _execute_generic_function(self, tool_def: ToolDefinition, input_data: Any, **kwargs) -> Any:
        """Execute generic function"""
        return tool_def.function(input_data, **kwargs)
    
    def _postprocess_result(self, result: Any, tool_def: ToolDefinition) -> Any:
        """Postprocess result"""
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)


class ToolGenerator:
    """Tool generator - Dynamically generate new tool functions"""
    
    def __init__(self, llm_engine):
        self.llm_engine = llm_engine
        self.validator = ToolValidator()
    
    def generate_tool_function(self, description: str, tool_type: ToolType, 
                             examples: List[str] = None) -> Optional[ToolDefinition]:
        """Generate new tool function based on description"""
        try:
            # Build generation prompt
            generation_prompt = self._build_generation_prompt(description, tool_type, examples)
            
            # Use LLM to generate tool code
            response = self.llm_engine.generate(
                content=generation_prompt,
                max_tokens=1000,
                temperature=0.3
            )
            
            # Parse generated code
            tool_def = self._parse_generated_tool(response, description, tool_type)
            
            # Validate tool function
            if tool_def and self._validate_generated_tool(tool_def):
                return tool_def
            else:
                print(f"⚠️ Generated tool function validation failed: {description}")
                return None
                
        except Exception as e:
            print(f"❌ Tool function generation error: {e}")
            return None
    
    def _build_generation_prompt(self, description: str, tool_type: ToolType, 
                               examples: List[str] = None) -> str:
        """Build tool generation prompt"""
        examples_text = "\n".join(examples) if examples else "No examples provided"
        
        return f"""Generate a Python tool function with the following requirements:

Description: {description}
Tool Type: {tool_type.value}
Examples: {examples_text}

Please generate a complete Python function in the following format:
```python
def tool_function(input_data, **kwargs):
    \"\"\"
    Tool function description
    
    Args:
        input_data: Input data
        **kwargs: Additional parameters
    
    Returns:
        Processing result
    \"\"\"
    # Implementation code
    pass
```

Requirements:
1. Function must accept input_data parameter
2. Must return meaningful results
3. Include appropriate error handling
4. Do not use dangerous system calls
5. Keep code concise and efficient

Please return only the function code, no additional content."""
    
    def _parse_generated_tool(self, response: str, description: str, 
                            tool_type: ToolType) -> Optional[ToolDefinition]:
        """Parse generated tool code"""
        try:
            # Extract code block
            code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
            if not code_match:
                code_match = re.search(r'```\n(.*?)\n```', response, re.DOTALL)
            
            if not code_match:
                # Try to parse entire response
                code = response.strip()
            else:
                code = code_match.group(1)
            
            # Create function object
            namespace = {}
            exec(code, namespace)
            
            # Find defined function
            tool_function = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith('_'):
                    tool_function = obj
                    break
            
            if not tool_function:
                return None
            
            # Create tool definition
            tool_def = ToolDefinition(
                name=f"generated_{tool_type.value}_{int(time.time())}",
                description=description,
                tool_type=tool_type,
                function=tool_function,
                parameters={},
                return_type="string",
                examples=[],
                safety_level="safe"
            )
            
            return tool_def
            
        except Exception as e:
            print(f"❌ Failed to parse generated tool function: {e}")
            return None
    
    def _validate_generated_tool(self, tool_def: ToolDefinition) -> bool:
        """Validate generated tool function"""
        try:
            # Validate function signature
            if not self.validator.validate_function_signature(tool_def.function, ['input_data']):
                return False
            
            # Test execution
            test_input = "Test input"
            result = tool_def.function(test_input)
            
            return result is not None
            
        except Exception:
            return False


class ToolUpdater:
    """Tool updater - Improve existing tool functions"""
    
    def __init__(self, llm_engine):
        self.llm_engine = llm_engine
        self.generator = ToolGenerator(llm_engine)
        self.validator = ToolValidator()
    
    def update_tool_function(self, current_tool: ToolDefinition, 
                           feedback: str, performance_data: Dict = None) -> Optional[ToolDefinition]:
        """Update tool function based on feedback"""
        try:
            # Build update prompt
            update_prompt = self._build_update_prompt(current_tool, feedback, performance_data)
            
            # Use LLM to generate improved tool code
            response = self.llm_engine.generate(
                content=update_prompt,
                max_tokens=1200,
                temperature=0.2
            )
            
            # Parse improved code
            updated_tool = self._parse_updated_tool(response, current_tool)
            
            # Validate updated tool
            if updated_tool and self._validate_updated_tool(updated_tool, current_tool):
                return updated_tool
            else:
                print(f"⚠️ Tool function update validation failed: {current_tool.name}")
                return current_tool  # Return original tool
                
        except Exception as e:
            print(f"❌ Tool function update error: {e}")
            return current_tool
    
    def _build_update_prompt(self, current_tool: ToolDefinition, 
                           feedback: str, performance_data: Dict = None) -> str:
        """Build tool update prompt"""
        
        # Get current function source code
        try:
            current_source = inspect.getsource(current_tool.function)
        except Exception:
            current_source = "# Unable to get source code"
        
        performance_info = ""
        if performance_data:
            performance_info = f"""
Performance Data:
- Execution Count: {performance_data.get('execution_count', 0)}
- Average Execution Time: {performance_data.get('avg_execution_time', 0):.3f}s
- Success Rate: {performance_data.get('success_rate', 0):.2%}
- Common Errors: {performance_data.get('common_errors', [])}
"""
        
        return f"""Please improve the following tool function:

Current Tool Function:
```python
{current_source}
```

Tool Description: {current_tool.description}
Tool Type: {current_tool.tool_type.value}

Feedback: {feedback}

{performance_info}

Please improve the tool function based on the feedback, requirements:
1. Keep the function signature unchanged (accept input_data parameter)
2. Optimize functionality and performance based on feedback
3. Enhance error handling and edge case processing
4. Maintain code security
5. Add necessary documentation

Please return the improved complete function code:

```python
def tool_function(input_data, **kwargs):
    # Improved implementation
    pass
```"""
    
    def _parse_updated_tool(self, response: str, current_tool: ToolDefinition) -> Optional[ToolDefinition]:
        """Parse updated tool code"""
        try:
            # Extract code block
            code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
            if not code_match:
                code_match = re.search(r'```\n(.*?)\n```', response, re.DOTALL)
            
            if not code_match:
                code = response.strip()
            else:
                code = code_match.group(1)
            
            # Create new function object
            namespace = {}
            exec(code, namespace)
            
            # Find updated function
            updated_function = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith('_'):
                    updated_function = obj
                    break
            
            if not updated_function:
                return None
            
            # Create updated tool definition
            updated_tool = ToolDefinition(
                name=current_tool.name + "_updated",
                description=current_tool.description,
                tool_type=current_tool.tool_type,
                function=updated_function,
                parameters=current_tool.parameters,
                return_type=current_tool.return_type,
                examples=current_tool.examples,
                safety_level=current_tool.safety_level
            )
            
            return updated_tool
            
        except Exception as e:
            print(f"❌ Failed to parse updated tool function: {e}")
            return None
    
    def _validate_updated_tool(self, updated_tool: ToolDefinition, 
                             original_tool: ToolDefinition) -> bool:
        """Validate updated tool function"""
        try:
            # Validate function signature
            if not self.validator.validate_function_signature(updated_tool.function, ['input_data']):
                return False
            
            # Compare performance - simple test
            test_inputs = ["Test input", "Another test", "Data analysis test"]
            
            for test_input in test_inputs:
                try:
                    result = updated_tool.function(test_input)
                    if result is None:
                        return False
                except Exception:
                    return False
            
            return True
            
        except Exception:
            return False


class ToolRegistry:
    """Tool registry - Manage all tool functions"""
    
    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self.executor = ToolExecutor()
        self.performance_stats: Dict[str, Dict] = {}
    
    def register_tool(self, tool_def: ToolDefinition) -> bool:
        """Register tool function"""
        try:
            self.tools[tool_def.name] = tool_def
            self.performance_stats[tool_def.name] = {
                'execution_count': 0,
                'success_count': 0,
                'total_execution_time': 0.0,
                'errors': []
            }
            return True
        except Exception as e:
            print(f"❌ Tool registration failed: {e}")
            return False
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get tool function"""
        return self.tools.get(name)
    
    def execute_tool(self, name: str, input_data: str, **kwargs) -> ToolExecutionResult:
        """Execute tool function"""
        tool_def = self.get_tool(name)
        if not tool_def:
            return ToolExecutionResult(
                success=False,
                result=None,
                error_message=f"Tool '{name}' not found"
            )
        
        # Execute tool
        result = self.executor.execute_tool(tool_def, input_data, **kwargs)
        
        # Update performance statistics
        self._update_performance_stats(name, result)
        
        return result
    
    def _update_performance_stats(self, tool_name: str, result: ToolExecutionResult):
        """Update performance statistics"""
        if tool_name not in self.performance_stats:
            self.performance_stats[tool_name] = {
                'execution_count': 0,
                'success_count': 0,
                'total_execution_time': 0.0,
                'errors': []
            }
        
        stats = self.performance_stats[tool_name]
        stats['execution_count'] += 1
        stats['total_execution_time'] += result.execution_time
        
        if result.success:
            stats['success_count'] += 1
        else:
            stats['errors'].append(result.error_message)
            # Only keep last 10 errors
            if len(stats['errors']) > 10:
                stats['errors'] = stats['errors'][-10:]
    
    def get_performance_report(self, tool_name: str) -> Dict:
        """Get tool performance report"""
        if tool_name not in self.performance_stats:
            return {}
        
        stats = self.performance_stats[tool_name]
        
        return {
            'execution_count': stats['execution_count'],
            'success_rate': stats['success_count'] / max(stats['execution_count'], 1),
            'avg_execution_time': stats['total_execution_time'] / max(stats['execution_count'], 1),
            'common_errors': list(set(stats['errors'])),
            'recent_errors': stats['errors'][-3:] if stats['errors'] else []
        }
    
    def list_tools(self) -> List[str]:
        """List all tools"""
        return list(self.tools.keys()) 
