from typing import List, Dict, Any, Callable
from .agent import Agent
import time

class Aggregator(Agent):
    """Aggregator class: Collects and integrates outputs from agents with out-degree 0, inherits from Agent for unified processing"""
    
    def __init__(self, aggregator_id: str = "aggregator"):
        # Initialize as Agent with aggregation-specific system prompt
        aggregator_system_prompt = """You are an expert output aggregator responsible for synthesizing multiple agent outputs into a coherent, comprehensive final result. Your goal is to create a unified response that captures the key insights from all inputs while maintaining accuracy and completeness."""
        
        super().__init__(
            agent_id=aggregator_id,
            system_prompt=aggregator_system_prompt,
            tool_function=self._aggregation_tool
        )
        
        self.aggregator_id = aggregator_id  # Maintain backward compatibility
        self.final_result = None
    
    def _aggregation_tool(self, instruction: str, agent_outputs: List[Dict[str, str]]) -> str:
        """Aggregation tool function - receives complete outputs including system prompts and responses"""
        if not agent_outputs:
            return "No agent outputs to aggregate"
        
        # Format agent outputs, including system prompts and responses
        formatted_outputs = []
        for i, output_info in enumerate(agent_outputs):
            agent_id = output_info.get('agent_id', f'Agent_{i+1}')
            system_prompt = output_info.get('system_prompt', '')
            response = output_info.get('response', '')
            
            formatted_output = f"""Agent {agent_id}:
Role/System Prompt: {system_prompt}
Response to instruction: {response}"""
            formatted_outputs.append(formatted_output)
        
        aggregation_instruction = f"""You are tasked with aggregating multiple agent outputs into a comprehensive final result.

Original instruction: {instruction}

Agent outputs to aggregate:
{chr(10).join(formatted_outputs)}

Please synthesize these outputs into a comprehensive final result that:
1. Integrates all key insights from each agent's response
2. Considers each agent's specialized role and capabilities
3. Resolves any conflicts between outputs
4. Directly answers the original instruction without unnecessary explanation

Final result:"""
        
        return aggregation_instruction
    
    def add_predecessor(self, agent: Agent):
        """Add predecessor agent - rewritten to add predecessor instead of successor"""
        if agent not in self.predecessors:
            self.predecessors.append(agent)
            agent.add_successor(self)
    
    def remove_predecessor(self, agent: Agent):
        """Remove predecessor agent"""
        if agent in self.predecessors:
            self.predecessors.remove(agent)
            agent.remove_successor(self)

    def backward_propagation(self, successor_feedbacks: List[Dict], visit_counts: Dict[str, int], 
                           llm_engine, network=None) -> Dict[str, Any]:
        """Backward propagation: Improve agent based on successor feedback and decide network structure changes"""
        
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
            
        self.feedback = feedback
        return improvements
    
    def aggregate_outputs(self, instruction: str, llm_engine) -> str:
        """Aggregate outputs from all agents with out-degree 0"""
        # Find all agents with out-degree 0 from the network (excluding connections to the aggregator)
        zero_out_degree_agents = self._find_zero_out_degree_agents()
        
        if not zero_out_degree_agents:
            return "No zero out-degree agents found to aggregate"
        
        # Collect complete outputs from all agents with out-degree 0
        agent_outputs = []
        for agent in zero_out_degree_agents:
            if hasattr(agent, 'tool_result') and agent.tool_result:
                agent_output = {
                    'agent_id': agent.agent_id,
                    'system_prompt': agent.system_prompt,
                    'response': agent.tool_result
                }
                agent_outputs.append(agent_output)
        
        if not agent_outputs:
            return "No agents produced output to aggregate"
        
        # Generate instruction using aggregation tool
        self.input_instruction = self._aggregation_tool(instruction, agent_outputs)
        
        # Generate final aggregated result using LLM
        self.tool_result = llm_engine.generate(
            content=self.input_instruction,
            system_prompt=self.system_prompt,
            max_tokens=1500,
            temperature=0.3
        )
        
        self.final_result = self.tool_result
        return self.final_result
    
    def _find_zero_out_degree_agents(self) -> List[Agent]:
        """Find all agents with out-degree 0 (excluding connections to the aggregator)"""
        zero_out_degree_agents = []
        
        # If network reference exists, use network's method directly
        if hasattr(self, 'network') and self.network:
            return self.network.find_zero_out_degree_agents()
        
        # Fallback method: get through network relationships of predecessors
        all_agents = set()
        
        # Starting from predecessors, traverse the entire network
        for predecessor in self.predecessors:
            all_agents.add(predecessor)
            self._collect_all_agents_recursive(predecessor, all_agents)
        
        # Check out-degree of each agent
        for agent in all_agents:
            if agent == self:  # Skip the aggregator itself
                continue
            
            # Check if it's an agent with out-degree 0 (excluding connections to aggregator)
            non_aggregator_successors = [s for s in agent.successors if s != self]
            if not non_aggregator_successors:
                zero_out_degree_agents.append(agent)
        
        return zero_out_degree_agents
    
    def _collect_all_agents_recursive(self, agent: Agent, all_agents: set):
        """Recursively collect all agents in the network"""
        for predecessor in getattr(agent, 'predecessors', []):
            if predecessor not in all_agents:
                all_agents.add(predecessor)
                self._collect_all_agents_recursive(predecessor, all_agents)
        
        for successor in getattr(agent, 'successors', []):
            if isinstance(successor, Agent) and successor not in all_agents:
                all_agents.add(successor)
                self._collect_all_agents_recursive(successor, all_agents)
    
    def generate_feedback(self, llm_engine, loss_grad: str) -> str:
        """Generate feedback for predecessor agents - using TextGrad standard format"""
        if not self.final_result:
            return "No final result available for feedback generation"
        
        feedback_prompt = llm_engine.PREDECESSOR_FEEDBACK_TEMPLATE.format(
            predecessor="predecessor agents",
            successor=f"Aggregator {self.aggregator_id}",
            successor_feedback=loss_grad,
            context=f"Final aggregated result: {self.final_result[:200]}..."
        )
        
        try:
            feedback_response = llm_engine.generate(
                content=feedback_prompt,
                max_tokens=600,
                temperature=0.3
            )
            
            # Extract structured feedback
            import re
            feedback_match = re.search(r'<FEEDBACK>\s*(.*?)\s*</FEEDBACK>', feedback_response, re.DOTALL)
            if feedback_match:
                return feedback_match.group(1).strip()
            
            return feedback_response.strip()
            
        except Exception as e:
            return f"Error generating feedback: {str(e)}"
    
    def __str__(self):
        return f"Aggregator({self.aggregator_id})"
    
    def __repr__(self):
        return self.__str__() 