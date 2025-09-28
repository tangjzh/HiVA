from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict
import json
import time
import threading
from sentence_transformers import SentenceTransformer
import numpy as np
from .tool import (
    ToolDefinition, ToolType, ToolRegistry, ToolGenerator, 
    ToolUpdater, ToolExecutionResult, ToolValidator
)
import torch

# Use thread-local storage to support parallel execution
_thread_local = threading.local()

def get_embedding_model():
    """Get thread-local embedding model instance"""
    if not hasattr(_thread_local, 'embedding_model'):
        try:
            # Initialize model
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            # Try to move to CUDA, fallback to CPU if failed
            if torch.cuda.is_available():
                try:
                    # Try normal to() method first
                    model = model.to('cuda')
                except RuntimeError as e:
                    if "meta tensor" in str(e).lower():
                        # If meta tensor error occurs, use to_empty()
                        print("Meta tensor detected, using to_empty() method...")
                        model = model.to_empty(device='cuda')
                    else:
                        # Other errors, keep on CPU
                        print(f"Cannot move to CUDA, keeping on CPU: {e}")
                        pass  # Model stays on CPU
            
            _thread_local.embedding_model = model
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            # If completely failed, try loading on CPU only
            try:
                _thread_local.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            except Exception as e2:
                print(f"Error loading embedding model on CPU: {e2}")
                _thread_local.embedding_model = None
    return _thread_local.embedding_model


class Agent:
    """Agent class: contains system prompt and tool functions"""
    
    def __init__(self, agent_id: str, system_prompt: str, tool_function: Callable = None):
        self.agent_id = agent_id
        self.system_prompt = system_prompt
        
        # Initialize tool system
        self.tool_registry = ToolRegistry()
        self.tool_generator = None  # Will be initialized when needed
        self.tool_updater = None   # Will be initialized when needed
        
        # Set default tool function
        if tool_function:
            self._register_default_tool(tool_function)
        else:
            self._create_default_tool()
        
        self.successors = []  # List of successor agents
        self.predecessors = []  # List of predecessor agents
        self.input_instruction = None  # Received instruction
        self.tool_result = None  # Tool execution result
        self.tool_execution_history = []  # Tool execution history
        self.visit_count = 0  # Number of times visited
        self.feedback = None  # Feedback during backpropagation
        self.performance_history = []  # Performance history records
        
    def _register_default_tool(self, tool_function: Callable):
        """Register default tool function"""
        try:
            # Analyze tool function type
            tool_type = self._analyze_tool_type(tool_function)
            
            # Create tool definition
            tool_def = ToolDefinition(
                name=f"{self.agent_id}_default_tool",
                description="Agent's default tool function",
                tool_type=tool_type,
                function=tool_function,
                parameters={},
                return_type="string",
                examples=[],
                safety_level="safe"
            )
            
            self.tool_registry.register_tool(tool_def)
            self.default_tool_name = tool_def.name
            
        except Exception as e:
            print(f"⚠️ Default tool registration failed: {e}")
            self._create_fallback_tool()
    
    def _create_default_tool(self):
        """Create default tool function"""
        def default_analysis_tool(input_data, **kwargs):
            """Default analysis tool"""
            return f"Analysis result: Basic processing and analysis performed on '{input_data}'."
        
        self._register_default_tool(default_analysis_tool)
    
    def _create_fallback_tool(self):
        """Create fallback tool function"""
        def fallback_tool(input_data, **kwargs):
            """Fallback tool function"""
            return f"Basic processing: {input_data}"
        
        tool_def = ToolDefinition(
            name=f"{self.agent_id}_fallback_tool",
            description="Fallback tool function",
            tool_type=ToolType.CUSTOM,
            function=fallback_tool,
            parameters={},
            return_type="string",
            examples=[],
            safety_level="safe"
        )
        
        self.tool_registry.register_tool(tool_def)
        self.default_tool_name = tool_def.name
    
    def _analyze_tool_type(self, tool_function: Callable) -> ToolType:
        """Analyze tool function type"""
        try:
            # Get function source code or docstring
            import inspect
            source = inspect.getsource(tool_function).lower()
            doc = (tool_function.__doc__ or "").lower()
            combined = source + " " + doc
            
            # Determine tool type based on keywords
            if any(keyword in combined for keyword in ['math', '数学', 'calculate', '计算']):
                return ToolType.MATHEMATICAL
            elif any(keyword in combined for keyword in ['text', '文本', 'analysis', '分析']):
                return ToolType.TEXT_ANALYSIS
            elif any(keyword in combined for keyword in ['data', '数据', 'process', '处理']):
                return ToolType.DATA_PROCESSING
            elif any(keyword in combined for keyword in ['api', 'request', '请求']):
                return ToolType.API_CALL
            else:
                return ToolType.PYTHON_FUNCTION
                
        except Exception:
            return ToolType.CUSTOM
    
    def execute_tool(self, instruction: str, tool_name: str = None) -> ToolExecutionResult:
        """Execute tool function (enhanced version)"""
        self.input_instruction = instruction
        
        # Select tool to execute
        if not tool_name:
            tool_name = self.default_tool_name
        
        # Execute tool
        result = self.tool_registry.execute_tool(tool_name, instruction)
        
        # Record result
        self.tool_result = result.result if result.success else result.error_message
        self.tool_execution_history.append(result)
        
        # If execution failed, try fallback strategy
        if not result.success:
            print(f"⚠️ Tool {tool_name} execution failed: {result.error_message}")
            result = self._execute_fallback_tool(instruction)
        
        return result
    
    def _execute_fallback_tool(self, instruction: str) -> ToolExecutionResult:
        """Execute fallback tool"""
        try:
            fallback_result = f"Fallback processing: {instruction}"
            self.tool_result = fallback_result
            
            return ToolExecutionResult(
                success=True,
                result=fallback_result,
                metadata={"fallback": True}
            )
        except Exception as e:
            return ToolExecutionResult(
                success=False,
                result=None,
                error_message=f"Fallback tool also failed: {e}"
            )
    
    def add_tool(self, tool_def: ToolDefinition) -> bool:
        """Add new tool"""
        return self.tool_registry.register_tool(tool_def)
    
    def generate_new_tool(self, description: str, tool_type: ToolType, 
                         examples: List[str] = None, llm_engine = None) -> Optional[str]:
        """Dynamically generate new tool function"""
        if not llm_engine:
            print("⚠️ LLM engine required to generate tool function")
            return None
        
        # Initialize tool generator
        if not self.tool_generator:
            self.tool_generator = ToolGenerator(llm_engine)
        
        # Generate new tool
        tool_def = self.tool_generator.generate_tool_function(description, tool_type, examples)
        
        if tool_def:
            # Register new tool
            if self.tool_registry.register_tool(tool_def):
                print(f"✅ Successfully generated and registered new tool: {tool_def.name}")
                return tool_def.name
            else:
                print(f"❌ Tool registration failed: {tool_def.name}")
        
        return None
    
    def update_tool(self, tool_name: str, feedback: str, llm_engine = None) -> bool:
        """Update existing tool function"""
        if not llm_engine:
            print("⚠️ LLM engine required to update tool function")
            return False
        
        # Initialize tool updater
        if not self.tool_updater:
            self.tool_updater = ToolUpdater(llm_engine)
        
        # Get current tool
        current_tool = self.tool_registry.get_tool(tool_name)
        if not current_tool:
            print(f"⚠️ Tool {tool_name} does not exist")
            return False
        
        # Get performance data
        performance_data = self.tool_registry.get_performance_report(tool_name)
        
        # Update tool
        updated_tool = self.tool_updater.update_tool_function(
            current_tool, feedback, performance_data
        )
        
        if updated_tool and updated_tool != current_tool:
            # Register updated tool
            if self.tool_registry.register_tool(updated_tool):
                print(f"✅ Successfully updated tool: {tool_name} -> {updated_tool.name}")
                
                # If it's the default tool, update reference
                if tool_name == self.default_tool_name:
                    self.default_tool_name = updated_tool.name
                
                return True
        
        return False
    
    def get_tool_performance_report(self) -> Dict[str, Any]:
        """Get performance report for all tools"""
        reports = {}
        for tool_name in self.tool_registry.list_tools():
            reports[tool_name] = self.tool_registry.get_performance_report(tool_name)
        
        return {
            "tool_reports": reports,
            "execution_history_count": len(self.tool_execution_history),
            "recent_executions": self.tool_execution_history[-5:] if self.tool_execution_history else []
        }
    
    def add_successor(self, agent: 'Agent'):
        """Add successor agent"""
        if agent not in self.successors:
            self.successors.append(agent)
            agent.predecessors.append(self)
    
    def remove_successor(self, agent: 'Agent'):
        """Remove successor agent"""
        if agent in self.successors:
            self.successors.remove(agent)
            agent.predecessors.remove(self)
    
    def calculate_semantic_similarity(self, other_agent: 'Agent') -> float:
        """Calculate semantic similarity with other agent (using sentence-transformers)"""
        try:
            
            # If no input instruction or system prompt, return 0
            if not self.input_instruction or not other_agent.system_prompt:
                return 0.0
                
            model = get_embedding_model()
            
            # Encode text
            instruction_embedding = model.encode(self.input_instruction)
            prompt_embedding = model.encode(other_agent.system_prompt)
            
            similarity = np.dot(instruction_embedding, prompt_embedding) / (np.linalg.norm(instruction_embedding) * np.linalg.norm(prompt_embedding))
            
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            # Fallback to simple word overlap calculation on error
            instruction_words = set(self.input_instruction.lower().split())
            prompt_words = set(other_agent.system_prompt.lower().split())
            
            if not instruction_words or not prompt_words:
                return 0.0
                
            intersection = instruction_words.intersection(prompt_words)
            union = instruction_words.union(prompt_words)
            
            return len(intersection) / len(union) if union else 0.0
    
    def select_top_k_successors(self, k: int) -> List['Agent']:
        """Select top-k successors with highest semantic similarity"""
        if not self.successors:
            return []
        
        similarities = []
        for successor in self.successors:
            similarity = self.calculate_semantic_similarity(successor)
            similarities.append((similarity, successor))
        
        # Sort by similarity and select top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [agent for _, agent in similarities[:k]]
    
    def generate_successor_instructions(self, llm_engine, top_k_successors: List['Agent']) -> Dict['Agent', str]:
        """Generate instructions for topk successors"""
        instructions = {}
        
        for successor in top_k_successors:
            # Build prompt to generate instructions
            system_prompt = """You are an instruction generator for multi-agent systems. Create clear, specific instructions that effectively transfer information between agents."""
            
            user_prompt = f"""Generate a clear instruction for the successor agent based on the current context:

CURRENT AGENT INPUT:
{self.input_instruction}

CURRENT AGENT TOOL RESULT:
{self.tool_result}

SUCCESSOR AGENT SYSTEM PROMPT:
{successor.system_prompt}

SUCCESSOR AGENT ID:
{successor.agent_id}

Requirements:
1. Create a specific, actionable instruction
2. Ensure the instruction aligns with the successor's capabilities
3. Transfer relevant context from the current agent's results
4. Keep the instruction concise but informative

Generate instruction for successor agent:"""
            
            try:
                instruction = llm_engine.generate(user_prompt, system_prompt)
                instructions[successor] = instruction.strip()
            except Exception as e:
                instructions[successor] = f"Process results from {self.agent_id}: {str(self.tool_result)[:100]}..."
        
        return instructions
    
    def backward_propagation(self, successor_feedbacks: List[Dict], visit_counts: Dict[str, int], 
                           llm_engine, network=None) -> Dict[str, Any]:
        """Backpropagation: improve agent based on successor feedback and decide network structure changes"""
        
        # Process feedback
        feedback = self.process_feedback(successor_feedbacks, visit_counts, llm_engine)
        improvements = {}
        
        if feedback:
            # Improve system prompt
            if 'system_prompt_feedback' in feedback:
                prompt_improvement_text = llm_engine.IMPROVEMENT_PROMPT_TEMPLATE.format(
                    variable=self.system_prompt,
                    role="AI agent system prompt",
                    gradients=feedback['system_prompt_feedback']
                )
                
                improved_prompt_response = llm_engine.generate(
                    content=prompt_improvement_text,
                    max_tokens=800,
                    temperature=0.3
                )
                
                # Extract improved system prompt using TextGrad format
                improved_prompt = self._extract_improved_variable(improved_prompt_response)
                if improved_prompt and improved_prompt != self.system_prompt:
                    self.system_prompt = improved_prompt
                    improvements['system_prompt'] = improved_prompt
                    self.performance_history.append({
                        'timestamp': time.time(),
                        'metric': 'system_prompt_updated',
                        'value': 1.0,
                        'feedback': feedback['system_prompt_feedback']
                    })
            
            # Optimize tool function
            if 'tool_feedback' in feedback:
                # Use new tool update system
                success = self.update_tool(
                    tool_name=self.default_tool_name,
                    feedback=feedback['tool_feedback'],
                    llm_engine=llm_engine
                )
                
                if success:
                    improvements['tool_function'] = "Tool function updated successfully"
                    self.performance_history.append({
                        'timestamp': time.time(),
                        'metric': 'tool_function_updated',
                        'value': 1.0,
                        'feedback': feedback['tool_feedback']
                    })
                else:
                    print(f"⚠️ Tool function update failed: {self.agent_id}")
                    improvements['tool_function'] = "Tool function update failed"
            
            # **Core addition: Network structure decision and changes**
            if network is not None:
                overall_feedback = feedback.get('overall_feedback', '')
                if overall_feedback:
                    structure_decision = self.decide_network_structure_change(overall_feedback, network, llm_engine)
                    
                    if structure_decision['action'] == 'add_successor':
                        # Choose different addition methods based on branch type
                        branch_type = structure_decision.get('branch_type', 'serial')
                        if branch_type == 'parallel':
                            success = network.execute_add_successor_parallel(self, structure_decision, llm_engine)
                        else:  # serial
                            success = network.execute_add_successor_serial(self, structure_decision, llm_engine)
                        
                        if success:
                            improvements['network_change'] = f"Added {branch_type} successor: {structure_decision.get('new_agent_description', 'New agent')}"
                        else:
                            improvements['network_change'] = f"Failed to add {branch_type} successor"
                        
                    elif structure_decision['action'] == 'remove_successor':
                        removed = network.execute_remove_successor(self, structure_decision)
                        if removed:
                            improvements['network_change'] = f"Removed successor: {structure_decision.get('target_info', 'Unknown')}"
                        else:
                            improvements['network_change'] = f"Failed to remove successor: {structure_decision.get('target_info', 'Unknown')}"
                        
                    else:
                        improvements['network_change'] = f"No change: {structure_decision.get('reason', 'Structure adequate')}"
                    
                    # Record decision reasoning process
                    if 'reasoning' in structure_decision:
                        improvements['structure_reasoning'] = structure_decision['reasoning']
        
        self.feedback = feedback
        return improvements
    
    def _extract_improved_variable(self, response: str) -> str:
        """Extract improved variable from structured response."""
        import re
        # Try to extract content from <IMPROVED_VARIABLE> tags
        improved_match = re.search(r'<IMPROVED_VARIABLE>\s*(.*?)\s*</IMPROVED_VARIABLE>', response, re.DOTALL)
        if improved_match:
            return improved_match.group(1).strip()
        
        # Fallback: return the whole response if no tags found
        return response.strip()
    
    def should_add_successor(self, llm_engine) -> bool:
        """Determine if a new successor should be added"""
        if not self.feedback:
            return False
        
        system_prompt = """You are a network topology optimizer. Analyze agent performance and determine if adding new successor agents would improve the system."""
        
        user_prompt = f"""Determine if this agent should add a new successor based on its performance:

AGENT FEEDBACK:
{self.feedback}

CURRENT SUCCESSOR COUNT:
{len(self.successors)}

AGENT ID:
{self.agent_id}

Analysis Criteria:
1. Does the feedback indicate capability gaps that a new agent could address?
2. Would specialization improve performance?
3. Is the current successor count manageable?
4. Are there clear benefits to adding complexity?

Response: Answer only 'True' or 'False'"""
        
        try:
            response = llm_engine.generate(user_prompt, system_prompt).strip().lower()
            return response == 'true'
        except Exception as e:
            return False
    
    def should_remove_successor(self, llm_engine) -> Optional['Agent']:
        """Determine if any successor should be removed"""
        if not self.successors or not self.feedback:
            return None
        
        system_prompt = """You are a network topology optimizer. Analyze agent performance and determine if any successor agents should be removed to improve efficiency."""
        
        user_prompt = f"""Determine if any successor agent should be removed based on performance feedback:

AGENT FEEDBACK:
{self.feedback}

SUCCESSOR AGENTS:
{[s.agent_id for s in self.successors]}

AGENT ID:
{self.agent_id}

Analysis Criteria:
1. Are any successors underperforming or redundant?
2. Would removing a successor simplify the network beneficially?
3. Is there evidence of inefficient information flow?
4. Would removal improve overall system performance?

Response: Provide the agent_id to remove, or 'None' if no removal is needed"""
        
        try:
            response = llm_engine.generate(user_prompt, system_prompt).strip()
            if response.lower() == 'none':
                return None
            
            for successor in self.successors:
                if successor.agent_id == response:
                    return successor
        except Exception as e:
            pass
        
        return None
    
    def decide_network_structure_change(self, feedback: str, network, llm_engine) -> Dict[str, Any]:
        """Decide network structure changes based on feedback
        
        Returns:
            Dict containing:
            - action: 'add_successor', 'remove_successor', 'no_change'
            - branch_type: 'parallel', 'serial' (if adding)
            - target_info: successor to remove (if removing)
            - new_agent_description: description for new agent (if adding)
            - reasoning: explanation of the decision
        """
        try:
            # First analyze task parallelizability
            parallelizability = self.analyze_task_parallelizability(feedback, llm_engine)
            
            # Construct enhanced decision prompt
            decision_prompt = f"""
            You are an intelligent network topology optimizer specializing in dynamic agent network evolution. Based on agent feedback, determine the optimal network structure modification strategy.

            Current Agent Role: {self.system_prompt[:200]}...
            
            Received Feedback: {feedback}
            
            Current Successor Count: {len(self.successors)}
            
            Task Parallelizability Analysis:
            - Can be parallelized: {parallelizability.get('can_parallelize', False)}
            - Subtask count: {parallelizability.get('subtasks_count', 0)}
            - Coordination required: {parallelizability.get('coordination_required', False)}
            - Parallel benefit: {parallelizability.get('parallel_benefit', 'None')}
            
            Network Structure Decision Options:
            
            Option A: ADD PARALLEL BRANCH (PARALLEL)
            - Use case: Task can be decomposed into independent subtasks requiring parallel processing
            - Structure change: A->B->C becomes A->(B,D)->C
            - Selection criteria: Feedback indicates need for specialization, parallel processing, or handling different aspects simultaneously
            
            Option B: ADD SERIAL NODE (SERIAL)
            - Use case: Need to insert preprocessing or intermediate processing steps into existing workflow
            - Structure change: A->B->C becomes A->D->B->C
            - Selection criteria: Feedback indicates need for additional processing steps, preprocessing, or quality control
            
            Option C: REMOVE SUCCESSOR (REMOVE)
            - Use case: Redundant or poorly performing nodes exist
            - Selection criteria: Feedback indicates redundancy, inefficiency, or simplification needs
            
            Option D: MAINTAIN CURRENT STRUCTURE (NO_CHANGE)
            - Use case: Current structure is adequate or changes would break connectivity
            
            Decision Rules:
            1. If feedback mentions "parallel", "concurrent", "simultaneously", "multiple aspects" -> Consider PARALLEL
            2. If feedback mentions "preprocessing", "intermediate steps", "quality check", "filtering" -> Consider SERIAL  
            3. If feedback mentions "redundant", "duplicate", "inefficient", "simplify" -> Consider REMOVE
            4. If current successor count >= 4, lean towards REMOVE or NO_CHANGE
            5. If task is parallelizable and current successor count < 3, lean towards PARALLEL
            
            Respond in EXACTLY one of these formats:
            
            ADD_PARALLEL: [Specific responsibility description for the new agent]
            
            ADD_SERIAL: [Specific responsibility description for the new agent]
            
            REMOVE_SUCCESSOR: [Index of successor to remove (0-based) or "least_visited"]
            
            NO_CHANGE: [Specific reason for maintaining current structure]
            """
            
            decision_response = llm_engine.generate(
                content=decision_prompt,
                max_tokens=400,
                temperature=0.3
            )
            
            # Parse decision
            decision_response = decision_response.strip()
            
            if decision_response.startswith("ADD_PARALLEL:"):
                new_agent_description = decision_response.split("ADD_PARALLEL:", 1)[1].strip()
                return {
                    'action': 'add_successor',
                    'branch_type': 'parallel',
                    'new_agent_description': new_agent_description,
                    'reasoning': f"Parallel branch: {new_agent_description}"
                }
            
            elif decision_response.startswith("ADD_SERIAL:"):
                new_agent_description = decision_response.split("ADD_SERIAL:", 1)[1].strip()
                return {
                    'action': 'add_successor',
                    'branch_type': 'serial',
                    'new_agent_description': new_agent_description,
                    'reasoning': f"Serial insertion: {new_agent_description}"
                }
            
            elif decision_response.startswith("REMOVE_SUCCESSOR:"):
                target_info = decision_response.split("REMOVE_SUCCESSOR:", 1)[1].strip()
                return {
                    'action': 'remove_successor',
                    'target_info': target_info,
                    'reasoning': f"Remove successor: {target_info}"
                }
            
            else:  # NO_CHANGE or unexpected response
                reason = decision_response.split("NO_CHANGE:", 1)[1].strip() if "NO_CHANGE:" in decision_response else decision_response
                return {
                    'action': 'no_change',
                    'reason': reason,
                    'reasoning': f"Maintain structure: {reason}"
                }
                
        except Exception as e:
            print(f"Error in network structure decision: {e}")
            return {
                'action': 'no_change', 
                'reason': 'Error in decision making',
                'reasoning': f"Decision error: {str(e)}"
            }

    def analyze_task_parallelizability(self, feedback: str, llm_engine) -> Dict[str, Any]:
        """Analyze task parallelization possibilities"""
        try:
            analysis_prompt = f"""
            Analyze the parallelization characteristics of the following task and feedback for multi-agent network optimization:

            Agent Role: {self.system_prompt[:150]}...
            Current Input: {getattr(self, 'input_instruction', 'N/A')[:150]}...
            Feedback Content: {feedback}

            Please analyze the following aspects:
            1. Can the task be decomposed into independent subtasks?
            2. Can these subtasks be executed in parallel?
            3. Do the subtasks require coordination or synchronization?
            4. Would parallel processing improve efficiency?

            Provide your analysis in the following structured format:
            CAN_PARALLELIZE: [True/False]
            SUBTASKS_COUNT: [Number from 0-5]
            COORDINATION_REQUIRED: [True/False]
            PARALLEL_BENEFIT: [High/Medium/Low/None]
            REASONING: [Brief explanation of your analysis]
            """
            
            response = llm_engine.generate(
                content=analysis_prompt,
                max_tokens=300,
                temperature=0.2
            )
            
            # Parse response
            result = {
                'can_parallelize': False,
                'subtasks_count': 0,
                'coordination_required': True,
                'parallel_benefit': 'None',
                'reasoning': ''
            }
            
            lines = response.strip().split('\n')
            for line in lines:
                if line.startswith('CAN_PARALLELIZE:'):
                    result['can_parallelize'] = 'True' in line
                elif line.startswith('SUBTASKS_COUNT:'):
                    try:
                        result['subtasks_count'] = int(line.split(':')[1].strip())
                    except:
                        result['subtasks_count'] = 1
                elif line.startswith('COORDINATION_REQUIRED:'):
                    result['coordination_required'] = 'True' in line
                elif line.startswith('PARALLEL_BENEFIT:'):
                    result['parallel_benefit'] = line.split(':')[1].strip()
                elif line.startswith('REASONING:'):
                    result['reasoning'] = line.split(':', 1)[1].strip()
            
            return result
            
        except Exception as e:
            print(f"Error in parallelizability analysis: {e}")
            return {
                'can_parallelize': False,
                'subtasks_count': 0,
                'coordination_required': True,
                'parallel_benefit': 'None',
                'reasoning': 'Analysis failed'
            }
    
    def create_new_successor_prompt(self, description: str, llm_engine) -> str:
        """Create system prompt for new successor"""
        try:
            prompt_creation = f"""
            You are an expert AI system architect. Create a focused, professional system prompt for a new AI agent based on the following specifications:
            
            Agent Description: {description}
            
            Context from Current Agent: {self.system_prompt[:300]}...
            
            Requirements:
            1. Create a specialized, complementary role that enhances the multi-agent network
            2. Ensure the role is focused, actionable, and clearly defined
            3. The agent should add distinct value without overlapping existing capabilities
            4. Maintain professional tone and clear operational guidelines
            5. Include specific competencies and expected behaviors
            
            Generate only the system prompt text:
            """
            
            new_prompt = llm_engine.generate(
                content=prompt_creation,
                max_tokens=400,
                temperature=0.5
            )
            
            return new_prompt.strip()
            
        except Exception as e:
            return f"You are a specialized AI assistant with expertise in: {description}. Your role is to provide focused, high-quality analysis and processing within your domain of specialization."
    
    def __str__(self):
        return f"Agent({self.agent_id})"
    
    def __repr__(self):
        return self.__str__()
    
    def process_feedback(self, successor_feedbacks: List[Dict], visit_counts: Dict[str, int], llm_engine) -> Dict[str, Any]:
        """Process feedback from successors"""
        if not successor_feedbacks:
            return {}
        
        try:
            # Aggregate all successor feedback
            feedback_summary = []
            for feedback_item in successor_feedbacks:
                agent_id = feedback_item.get('agent_id', 'unknown')
                feedback_text = feedback_item.get('feedback', '')
                visit_count = visit_counts.get(agent_id, 0)
                
                feedback_summary.append(f"Agent {agent_id} (visited {visit_count} times): {feedback_text}")
            
            combined_feedback = "\n\n".join(feedback_summary)
            
            # Use LLM to analyze feedback and generate improvement suggestions
            analysis_prompt = f"""
            Analyze the following feedback from successor agents and provide specific improvement suggestions for this agent.
            
            Current Agent Role: {self.system_prompt[:300]}...
            
            Successor Feedback:
            {combined_feedback}
            
            Based on this feedback, provide specific improvement suggestions in the following areas:
            
            1. SYSTEM_PROMPT_FEEDBACK: How to improve the agent's role and instructions
            2. TOOL_FEEDBACK: How to improve the agent's tools or methods  
            3. OVERALL_FEEDBACK: General assessment and strategic improvements
            
            Format your response as:
            SYSTEM_PROMPT_FEEDBACK: [specific suggestions for system prompt]
            TOOL_FEEDBACK: [specific suggestions for tools/methods]
            OVERALL_FEEDBACK: [general strategic feedback and recommendations]
            """
            
            analysis_response = llm_engine.generate(
                content=analysis_prompt,
                max_tokens=800,
                temperature=0.3
            )
            
            # Parse analysis results
            feedback_dict = {}
            lines = analysis_response.split('\n')
            current_section = None
            current_content = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('SYSTEM_PROMPT_FEEDBACK:'):
                    if current_section:
                        feedback_dict[current_section] = ' '.join(current_content)
                    current_section = 'system_prompt_feedback'
                    current_content = [line.split(':', 1)[1].strip()]
                elif line.startswith('TOOL_FEEDBACK:'):
                    if current_section:
                        feedback_dict[current_section] = ' '.join(current_content)
                    current_section = 'tool_feedback'
                    current_content = [line.split(':', 1)[1].strip()]
                elif line.startswith('OVERALL_FEEDBACK:'):
                    if current_section:
                        feedback_dict[current_section] = ' '.join(current_content)
                    current_section = 'overall_feedback'
                    current_content = [line.split(':', 1)[1].strip()]
                elif current_section and line:
                    current_content.append(line)
            
            # Add last section
            if current_section and current_content:
                feedback_dict[current_section] = ' '.join(current_content)
            
            return feedback_dict
            
        except Exception as e:
            print(f"Error processing feedback for agent {self.agent_id}: {e}")
            return {}
    
    @property
    def tool_function(self) -> Optional[Callable]:
        """Backward compatibility: get default tool function"""
        tool_def = self.tool_registry.get_tool(self.default_tool_name)
        return tool_def.function if tool_def else None
    
    @tool_function.setter
    def tool_function(self, func: Callable):
        """Backward compatibility: set tool function"""
        if func:
            self._register_default_tool(func) 