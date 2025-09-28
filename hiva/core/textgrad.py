from typing import Dict, Any, Callable, Optional, Union
import json
import re
from .agent import Agent
from .aggregator import Aggregator
from .network import AgentNetwork
from ..engines.base_engine import BaseLLMEngine
from ..engines.openai_engine import OpenAIEngine
from ..engines.mock_engine import MockEngine


class TextGrad:
    """Main class based on TextGrad algorithm framework, referencing Stanford TextGrad design"""
    
    def __init__(self, 
                 forward_engine: Union[BaseLLMEngine, Callable] = None,
                 backward_engine: Union[BaseLLMEngine, Callable] = None,
                 environment_function: Callable = None):
        """
        Initialize TextGrad
        
        Args:
            forward_engine: LLM engine or function used for forward propagation
            backward_engine: LLM engine or function used for backward propagation (if None, uses forward_engine)
            environment_function: Environment execution function, takes result as input and outputs environment feedback
        """
        # Set forward engine
        if forward_engine is None:
            forward_engine = MockEngine(smart_responses=True)
        elif callable(forward_engine) and not isinstance(forward_engine, BaseLLMEngine):
            # Wrap regular function as engine
            forward_engine = self._wrap_function_as_engine(forward_engine)
        
        # Set backward engine
        if backward_engine is None:
            backward_engine = forward_engine
        elif callable(backward_engine) and not isinstance(backward_engine, BaseLLMEngine):
            backward_engine = self._wrap_function_as_engine(backward_engine)
        
        self.forward_engine = forward_engine
        self.backward_engine = backward_engine
        self.environment_function = environment_function or self._default_environment_function
        self.network = AgentNetwork()
        self.iteration_history = []
    
    def _wrap_function_as_engine(self, func: Callable) -> BaseLLMEngine:
        """Wrap regular function as engine"""
        class FunctionEngine(BaseLLMEngine):
            def __init__(self, function):
                super().__init__("function-engine")
                self.function = function
            
            def _generate_impl(self, messages, **kwargs):
                # Convert messages to simple prompt
                prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                return self.function(prompt)
        
        return FunctionEngine(func)
    
    def _default_environment_function(self, result: str) -> str:
        """Default environment function"""
        return f"Environment feedback: Received result '{result[:100]}...'"
    
    def initialize_network(self, 
                          initial_system_prompt: str, 
                          initial_tool_function: Callable = None,
                          aggregator_id: str = "main_aggregator") -> bool:
        """
        Initialize agent network
        
        Args:
            initial_system_prompt: System prompt for source agent
            initial_tool_function: Tool function for source agent (optional)
            aggregator_id: Aggregator ID
        """
        try:
            # Create source agent
            source_agent = Agent(
                agent_id="source_agent",
                system_prompt=initial_system_prompt,
                tool_function=initial_tool_function
            )
            
            # Create aggregator
            aggregator = Aggregator(aggregator_id=aggregator_id)
            
            # Set network structure
            self.network.set_source_agent(source_agent)
            self.network.set_aggregator(aggregator)
            
            # Connect source agent to aggregator (initial structure: source -> aggregator)
            source_agent.add_successor(aggregator)
            aggregator.add_predecessor(source_agent)
            
            # Add aggregator to agents dict for unified management
            self.network.agents[aggregator_id] = aggregator
            
            # Set initial instruction storage
            self.initial_instruction = ""
            
            return True
            
        except Exception as e:
            print(f"Network initialization failed: {e}")
            return False
    
    def _extract_feedback(self, response: str) -> str:
        """Extract feedback from structured response."""
        # Try to extract content from <FEEDBACK> tags
        feedback_match = re.search(r'<FEEDBACK>\s*(.*?)\s*</FEEDBACK>', response, re.DOTALL)
        if feedback_match:
            return feedback_match.group(1).strip()
        
        # Fallback: return the whole response if no tags found
        return response.strip()
    
    def _extract_improved_variable(self, response: str) -> str:
        """Extract improved variable from structured response."""
        # Try to extract content from <IMPROVED_VARIABLE> tags
        improved_match = re.search(r'<IMPROVED_VARIABLE>\s*(.*?)\s*</IMPROVED_VARIABLE>', response, re.DOTALL)
        if improved_match:
            return improved_match.group(1).strip()
        
        # Fallback: return the whole response if no tags found
        return response.strip()
    
    def _compute_loss(self, result: str, criteria: str) -> str:
        """Compute loss using forward engine."""
        evaluation_prompt = self.forward_engine.EVALUATION_PROMPT_TEMPLATE.format(
            content=result,
            context=self.initial_instruction,
            criteria=criteria
        )
        
        loss_response = self.forward_engine.generate(
            content=evaluation_prompt,
            max_tokens=500,
            temperature=0.3
        )
        
        return self._extract_feedback(loss_response)
    
    def _compute_gradients(self, loss_text: str) -> str:
        """Compute gradients using backward engine."""
        gradient_prompt = self.backward_engine.GRADIENT_PROMPT_TEMPLATE.format(
            variable=f"Network result based on instruction: {self.initial_instruction}",
            role="system output to user instruction",
            feedback=loss_text
        )
        
        gradient_response = self.backward_engine.generate(
            content=gradient_prompt,
            max_tokens=600,
            temperature=0.3
        )
        
        return self._extract_feedback(gradient_response)
    
    def single_iteration(self, instruction: str, criteria: str = None, k: int = 3) -> Dict[str, Any]:
        """Execute single optimization iteration"""
        iteration_result = {
            "instruction": instruction,
            "forward_result": None,
            "environment_feedback": None,
            "loss": None,
            "loss_grad": None,
            "network_info_before": self.network.get_network_info(),
            "network_info_after": None,
            "backward_results": None,
            "topology_validation": None,
            "success": False
        }
        
        try:
            # Forward propagation
            forward_result = self.network.forward_propagation(instruction, self.forward_engine, k)
            iteration_result["forward_result"] = forward_result
            
            # Environment feedback
            environment_feedback = self.environment_function(forward_result)
            iteration_result["environment_feedback"] = environment_feedback
            
            # Compute loss
            if criteria is None:
                criteria = "Evaluate how well this result addresses the initial instruction. Consider accuracy, completeness, and effectiveness."
            loss = self._compute_loss(forward_result, criteria)
            iteration_result["loss"] = loss
            
            # Compute loss gradients
            loss_grad = self._compute_gradients(loss)
            iteration_result["loss_grad"] = loss_grad
            
            # Collect visit counts
            visit_counts = {}
            for agent_id, agent in self.network.agents.items():
                visit_counts[agent_id] = agent.visit_count
            
            # Backward propagation
            backward_results = self.network.backward_propagation(loss_grad, visit_counts, self.backward_engine)
            iteration_result["backward_results"] = backward_results
            
            # ⚠️ Validate and fix network topology (after each iteration)
            topology_validation = self.network.validate_and_fix_network_topology()
            iteration_result["topology_validation"] = topology_validation
            
            if topology_validation.get("fixes_applied"):
                print(f"🔧 Post-iteration topology fixes: {topology_validation['fixes_applied']}")
            
            iteration_result["network_info_after"] = self.network.get_network_info()
            iteration_result["success"] = True
            
        except Exception as e:
            iteration_result["error"] = str(e)
            print(f"Iteration execution error: {e}")
        
        return iteration_result
    
    def optimize(self, 
                initial_instruction: str, 
                max_iterations: int = 10, 
                k: int = 3,
                convergence_threshold: float = 0.1,
                criteria: str = None) -> Dict[str, Any]:
        """
        Execute complete optimization process
        
        Args:
            initial_instruction: Initial instruction
            max_iterations: Maximum number of iterations
            k: Number of top-k successors to select each time
            convergence_threshold: Convergence threshold (not used yet)
            criteria: Evaluation criteria
        """
        
        if not self.network.source_agent or not self.network.aggregator:
            return {
                "success": False,
                "error": "Network not properly initialized",
                "iterations": []
            }
        
        optimization_result = {
            "initial_instruction": initial_instruction,
            "max_iterations": max_iterations,
            "k": k,
            "iterations": [],
            "final_result": None,
            "final_network_info": None,
            "success": False
        }
        
        try:
            # Execute iterations
            for iteration in range(max_iterations):
                print(f"Executing iteration {iteration + 1}...")
                
                iteration_result = self.single_iteration(initial_instruction, criteria, k)
                optimization_result["iterations"].append(iteration_result)
                
                if not iteration_result["success"]:
                    print(f"Iteration {iteration + 1} failed")
                    break
                
                # Convergence check logic can be added here
                # if self._check_convergence(iteration_result):
                #     print(f"Converged after iteration {iteration + 1}")
                #     break
            
            # Get final results
            if optimization_result["iterations"]:
                last_iteration = optimization_result["iterations"][-1]
                optimization_result["final_result"] = last_iteration.get("forward_result")
                optimization_result["final_network_info"] = last_iteration.get("network_info_after")
                optimization_result["success"] = True
            
        except Exception as e:
            optimization_result["error"] = str(e)
            print(f"Optimization process error: {e}")
        
        return optimization_result
    
    def get_agent_by_id(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID"""
        return self.network.agents.get(agent_id)
    
    def add_custom_agent(self, agent: Agent) -> bool:
        """Add custom agent"""
        return self.network.add_agent(agent)
    
    def connect_agents(self, source_id: str, target_id: str) -> bool:
        """Connect two agents"""
        source = self.get_agent_by_id(source_id)
        target = self.get_agent_by_id(target_id)
        
        if source and target:
            source.add_successor(target)
            return True
        return False
    
    def export_network_state(self) -> Dict[str, Any]:
        """Export network state"""
        network_state = {
            "agents": {},
            "connections": [],
            "source_agent_id": self.network.source_agent.agent_id if self.network.source_agent else None,
            "aggregator_id": self.network.aggregator.aggregator_id if self.network.aggregator else None
        }
        
        # Export agent information
        for agent_id, agent in self.network.agents.items():
            network_state["agents"][agent_id] = {
                "system_prompt": agent.system_prompt,
                "has_tool_function": agent.tool_function is not None,
                "visit_count": agent.visit_count
            }
        
        # Export connection information
        for agent_id, agent in self.network.agents.items():
            for successor in agent.successors:
                if isinstance(successor, Agent):
                    network_state["connections"].append({
                        "source": agent_id,
                        "target": successor.agent_id,
                        "type": "agent_to_agent"
                    })
                else:  # Aggregator
                    network_state["connections"].append({
                        "source": agent_id,
                        "target": "aggregator",
                        "type": "agent_to_aggregator"
                    })
        
        return network_state
    
    def save_optimization_result(self, result: Dict[str, Any], filepath: str):
        """Save optimization result to file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Optimization result saved to: {filepath}")
        except Exception as e:
            print(f"Error saving optimization result: {e}")
    
    def __str__(self):
        return f"TextGrad(agents={len(self.network.agents)})"
    
    def __repr__(self):
        return self.__str__() 
    
    def __call__(self, instruction: str, k: int = 3):
        return self.network.forward_propagation(instruction, self.forward_engine, k)