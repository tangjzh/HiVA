from typing import List, Dict, Any, Optional, Callable, Union
from collections import defaultdict, deque
from .agent import Agent
from .aggregator import Aggregator
import time


class AgentNetwork:
    """Agent network flow graph: manages connections between agents"""
    
    def __init__(self):
        self.agents = {}  # agent_id -> Agent
        self.source_agent = None  # Source agent
        self.aggregator = None  # Sink aggregator
        self.visit_counts = defaultdict(int)  # Visit counts for agents
        
    def add_agent(self, agent: Agent) -> bool:
        """Add an agent to the network"""
        if agent.agent_id in self.agents:
            return False
        self.agents[agent.agent_id] = agent
        return True
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the network"""
        if agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        
        # Remove all connections
        for successor in agent.successors[:]:
            agent.remove_successor(successor)
        
        for predecessor in agent.predecessors[:]:
            predecessor.remove_successor(agent)
        
        del self.agents[agent_id]
        return True
    
    def set_source_agent(self, agent: Agent):
        """Set the source agent"""
        self.source_agent = agent
        if agent.agent_id not in self.agents:
            self.add_agent(agent)
    
    def set_aggregator(self, aggregator: Aggregator):
        """Set the sink aggregator"""
        self.aggregator = aggregator
        # Set network reference for aggregator to access all agents
        aggregator.network = self
    
    def get_topological_order(self) -> List[Agent]:
        """Calculate topological order of the network"""
        if not self.source_agent:
            return []
        
        # Use Kahn's algorithm to calculate topological order
        in_degree = defaultdict(int)
        nodes = list(self.agents.values())
        
        # Calculate in-degrees
        for agent in nodes:
            for successor in agent.successors:
                if isinstance(successor, Agent):  # Exclude aggregator
                    in_degree[successor.agent_id] += 1
        
        # Initialize queue (nodes with in-degree 0)
        queue = deque()
        if self.source_agent.agent_id in in_degree and in_degree[self.source_agent.agent_id] == 0:
            queue.append(self.source_agent)
        elif self.source_agent.agent_id not in in_degree:
            queue.append(self.source_agent)
        
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            # Update in-degrees of successor nodes
            for successor in current.successors:
                if isinstance(successor, Agent):
                    in_degree[successor.agent_id] -= 1
                    if in_degree[successor.agent_id] == 0:
                        queue.append(successor)
        
        return result
    
    def get_reverse_topological_order(self) -> List[Agent]:
        """Get reverse topological order"""
        return list(reversed(self.get_topological_order()))
    
    def find_zero_out_degree_agents(self) -> List[Agent]:
        """Find agents with out-degree 0 (excluding those connected to aggregator)"""
        zero_out_degree = []

        for agent in self.agents.values():
            # Skip aggregator itself
            if agent == self.aggregator:
                continue
            
            # Consider out-degree 0 if only connected to aggregator or no successors
            non_aggregator_successors = [s for s in agent.successors if isinstance(s, Agent) and s != self.aggregator]
            if not non_aggregator_successors:
                zero_out_degree.append(agent)
        return zero_out_degree
    
    def forward_propagation(self, instruction: str, llm_engine, k: int = 3) -> str:
        """Forward propagation process"""
        # Reset visit counts
        for agent in self.agents.values():
            agent.visit_count = 0
        
        # Get topological order
        topo_order = self.get_topological_order()
        
        for agent in topo_order:
            # Skip aggregator, it will be processed last
            if isinstance(agent, Aggregator):
                continue
            
            # Check if agent has received instruction
            if not hasattr(agent, 'input_instruction') or not agent.input_instruction:
                # Source agent receives initial instruction
                if agent == self.source_agent:
                    agent.input_instruction = instruction
                else:
                    # Skip other agents that haven't received instruction
                    continue
            
            # Execute agent
            try:
                # Use new tool execution system
                tool_result = agent.execute_tool(agent.input_instruction)
                
                if tool_result.success:
                    # Generate LLM response
                    content = f"""
                    Instruction: {agent.input_instruction}
                    Tooluse Result: {tool_result.result}
                    """
                    agent.tool_result = llm_engine.generate(
                        content=content,
                        system_prompt=agent.system_prompt,
                        max_tokens=1000,
                        temperature=0.7
                    )
                else:
                    # Tool execution failed, use LLM directly
                    content = f"""
                    Instruction: {agent.input_instruction}
                    Warning: Tool execution failed.
                    """
                    print(f"⚠️ Agent {agent.agent_id} tool execution failed, using LLM directly")
                    agent.tool_result = llm_engine.generate(
                        content=content,
                        system_prompt=agent.system_prompt,
                        max_tokens=1000,
                        temperature=0.7
                    )
                
                agent.visit_count += 1
                
                # Calculate semantic similarity and select successors
                successors = self.select_successors(agent, k)
                
                # Generate instructions for selected successors
                for successor in successors:
                    if isinstance(successor, Aggregator):
                        # Skip instruction generation for aggregator
                        continue
                    elif isinstance(successor, Agent):
                        successor.visit_count += 1
                        # Generate successor instruction
                        successor_instruction = self.generate_successor_instruction(
                            agent, successor, llm_engine
                        )
                        successor.input_instruction = successor_instruction
                
            except Exception as e:
                print(f"Error in agent {agent.agent_id}: {e}")
                agent.tool_result = f"Error: {str(e)}"
        
        # Process aggregator last
        if self.aggregator:
            try:
                final_result = self.aggregator.aggregate_outputs(instruction, llm_engine)
                return final_result
            except Exception as e:
                return f"Aggregation error: {str(e)}"
        else:
            # If no aggregator, return all agent outputs
            outputs = []
            for agent in self.agents.values():
                if hasattr(agent, 'tool_result') and agent.tool_result:
                    outputs.append(agent.tool_result)
            return "\n".join(outputs) if outputs else "No output"
    
    def backward_propagation(self, loss_grad: str, visit_counts: Dict[str, int], llm_engine) -> Dict[str, Any]:
        """Backward propagation: Improve agents and adjust network structure starting from aggregator in reverse topological order"""
        
        improvements = {}
        
        try:
            # Get reverse topological order
            reverse_order = self.get_reverse_topological_order()
            
            # Start backward propagation from aggregator
            for i, agent in enumerate(reverse_order):
                
                # Collect successor feedback
                successor_feedbacks = []
                
                if agent == self.aggregator:
                    # Aggregator uses loss_grad directly, but no network structure adjustment
                    successor_feedbacks = [{'agent_id': 'loss', 'feedback': loss_grad}]
                    
                    # ⚠️ Critical fix: Aggregator only improves itself, no network reference passed
                    try:
                        agent_improvements = agent.backward_propagation(
                            successor_feedbacks=successor_feedbacks,
                            visit_counts=visit_counts,
                            llm_engine=llm_engine,
                            network=None  # Don't pass network, prevent structure changes
                        )
                        
                        if agent_improvements:
                            improvements[agent.agent_id] = agent_improvements
                            
                    except Exception as e:
                        print(f"Error in backward propagation for aggregator {agent.agent_id}: {e}")
                        improvements[agent.agent_id] = {'error': str(e)}
                        
                else:
                    # Regular agents collect feedback from all successors
                    for successor in agent.successors:
                        if hasattr(successor, 'feedback') and successor.feedback:
                            feedback_text = successor.feedback.get('overall_feedback', '')
                            if feedback_text:
                                successor_feedbacks.append({
                                    'agent_id': successor.agent_id,
                                    'feedback': feedback_text
                                })
                    
                    # Regular agents can adjust network structure
                    if successor_feedbacks:
                        try:
                            agent_improvements = agent.backward_propagation(
                                successor_feedbacks=successor_feedbacks,
                                visit_counts=visit_counts,
                                llm_engine=llm_engine,
                                network=self  # Pass network reference to support structure changes
                            )
                            
                            if agent_improvements:
                                improvements[agent.agent_id] = agent_improvements
                                
                                # Record network structure changes
                                if 'network_change' in agent_improvements:
                                    print(f"🔄 Network change for {agent.agent_id}: {agent_improvements['network_change']}")
                                    
                        except Exception as e:
                            print(f"Error in backward propagation for agent {agent.agent_id}: {e}")
                            improvements[agent.agent_id] = {'error': str(e)}
                        
        except Exception as e:
            print(f"Error in backward propagation: {e}")
            improvements['error'] = str(e)
        
        return improvements
    
    def _generate_agent_feedback(self, agent: Agent, successor_feedback: List[Dict], 
                               visit_counts: Dict[str, int], llm_engine) -> str:
        """Generate feedback for an agent"""
        if not successor_feedback:
            return "No specific feedback from successors."
        
        # Build feedback context
        context_parts = []
        for feedback_item in successor_feedback:
            context_parts.append(f"Successor {feedback_item['successor_id']} "
                               f"(visited {feedback_item['visit_count']} times): {feedback_item['feedback']}")
        
        context = "; ".join(context_parts)
        
        # Use TextGrad standard format for feedback generation
        feedback_prompt = llm_engine.PREDECESSOR_FEEDBACK_TEMPLATE.format(
            predecessor=f"Agent {agent.agent_id}: {agent.system_prompt[:200]}...",
            successor="successor agents in network",
            successor_feedback=context,
            context=f"Agent visit count: {visit_counts.get(agent.agent_id, 0)}"
        )
        
        try:
            feedback_response = llm_engine.generate(
                content=feedback_prompt,
                max_tokens=600,
                temperature=0.3
            )
            
            # Extract feedback using TextGrad format
            return self._extract_feedback(feedback_response)
            
        except Exception as e:
            return f"Error generating feedback: {str(e)}"
    
    def _extract_feedback(self, response: str) -> str:
        """Extract feedback from structured response."""
        import re
        # Try to extract content from <FEEDBACK> tags
        feedback_match = re.search(r'<FEEDBACK>\s*(.*?)\s*</FEEDBACK>', response, re.DOTALL)
        if feedback_match:
            return feedback_match.group(1).strip()
        
        # Fallback: return the whole response if no tags found
        return response.strip()
    
    def improve_agent(self, agent: Agent, llm_engine):
        """Improve an agent"""
        # 1. Improve system prompt
        agent.improve_system_prompt(llm_engine)
        
        # 2. Improve tool functions (simplified)
        # More complex tool function improvement logic would be needed in practice
        
        # 3. Network structure adjustment
        self.adjust_network_structure(agent, llm_engine)
    
    def adjust_network_structure(self, agent: Agent, llm_engine):
        """Adjust network structure"""
        # Decide whether to add new successor
        if agent.should_add_successor(llm_engine):
            self.add_new_successor(agent, llm_engine)
        
        # Decide whether to remove successor
        successor_to_remove = agent.should_remove_successor(llm_engine)
        if successor_to_remove and self.can_remove_successor(agent, successor_to_remove):
            self.remove_successor_safely(agent, successor_to_remove)
    
    def add_new_successor(self, agent: Agent, llm_engine):
        """Add new successor for an agent"""
        # Generate new agent ID and system prompt
        new_agent_id = f"agent_{len(self.agents) + 1}"
        
        system_prompt = """You are an expert system designer creating specialized agent prompts. Create focused, role-specific system prompts that complement existing agents."""
        
        user_prompt = f"""Create a system prompt for a new successor agent based on the following context:

PREDECESSOR AGENT FEEDBACK:
{agent.feedback}

PREDECESSOR AGENT SYSTEM PROMPT:
{agent.system_prompt}

Requirements:
1. The new agent should complement the predecessor's capabilities
2. Focus on addressing gaps identified in the feedback
3. Create a clear, specific role definition
4. Ensure the prompt is actionable and well-structured

Generate a concise system prompt for the new successor agent:"""
        
        try:
            new_system_prompt = llm_engine.generate(user_prompt, system_prompt).strip()
            new_agent = Agent(new_agent_id, new_system_prompt)
            
            # Add to network
            self.add_agent(new_agent)
            
            # Connection: agent -> new_agent -> agent's original successors
            original_successors = agent.successors[:]
            
            # Remove original connections
            for successor in original_successors:
                agent.remove_successor(successor)
            
            # Create new connections
            agent.add_successor(new_agent)
            for successor in original_successors:
                new_agent.add_successor(successor)
                
        except Exception as e:
            print(f"Error adding new successor: {e}")
    
    def can_remove_successor(self, agent: Agent, successor: Agent) -> bool:
        """Check if successor can be safely removed (won't disconnect sink)"""
        # Simplified: can remove if it won't disconnect network
        if successor == self.aggregator:
            return False
        
        # Check if there are other paths to sink
        return len(agent.successors) > 1 or len(successor.successors) > 0
    
    def remove_successor_safely(self, agent: Agent, successor: Agent):
        """Safely remove a successor"""
        # Connect agent directly to successor's successors
        successor_successors = successor.successors[:]
        
        # Remove connection
        agent.remove_successor(successor)
        
        # Create new connections
        for s in successor_successors:
            agent.add_successor(s)
        
        # Remove successor from network
        self.remove_agent(successor.agent_id)
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get network information"""
        return {
            "total_agents": len(self.agents),
            "source_agent": self.source_agent.agent_id if self.source_agent else None,
            "has_aggregator": self.aggregator is not None,
            "connections": {
                agent_id: [s.agent_id if isinstance(s, Agent) else "aggregator" 
                          for s in agent.successors]
                for agent_id, agent in self.agents.items()
            }
        }
    
    def __str__(self):
        return f"AgentNetwork(agents={len(self.agents)}, source={self.source_agent})"
    
    def __repr__(self):
        return self.__str__()
    
    def _optimize_network_structure(self, agent: Agent, feedback: str, llm_engine) -> Dict[str, Any]:
        """Optimize network structure"""
        structure_changes = {
            'agents_added': [],
            'agents_removed': [],
            'connections_added': [],
            'connections_removed': []
        }
        
        try:
            # Decide structure changes based on feedback
            # This is a simplified implementation that can be extended
            if "add successor" in feedback.lower() or "new agent" in feedback.lower():
                # Consider adding new successor agent
                if len(agent.successors) < 3:  # Limit max successors
                    new_agent_id = f"auto_agent_{len(self.agents) + 1}"
                    new_prompt = f"You are a specialized agent created to improve the system based on feedback: {feedback[:200]}"
                    
                    new_agent = Agent(new_agent_id, new_prompt)
                    if self.add_agent(new_agent):
                        agent.add_successor(new_agent)
                        # Connect new agent to aggregator
                        if self.aggregator:
                            new_agent.add_successor(self.aggregator)
                        
                        structure_changes['agents_added'].append(new_agent_id)
                        structure_changes['connections_added'].append((agent.agent_id, new_agent_id))
            
            elif "remove" in feedback.lower() and "redundant" in feedback.lower():
                # Consider removing redundant connections
                if len(agent.successors) > 1:
                    # Simple strategy: remove least visited successor
                    min_visits = float('inf')
                    target_successor = None
                    
                    for successor in agent.successors:
                        if isinstance(successor, Agent) and successor.visit_count < min_visits:
                            min_visits = successor.visit_count
                            target_successor = successor
                    
                    if target_successor and min_visits == 0:  # Only remove never visited
                        agent.remove_successor(target_successor)
                        structure_changes['connections_removed'].append((agent.agent_id, target_successor.agent_id))
        
        except Exception as e:
            print(f"Warning: Error in network structure optimization: {e}")
        
        return structure_changes 
    
    def select_successors(self, agent: Agent, k: int) -> List[Agent]:
        """Intelligently select successor agents
        
        Uses semantic similarity and parallel branching strategy to select most suitable successors
        """
        if not agent.successors:
            return []
        
        # Return all successors if count <= k
        if len(agent.successors) <= k:
            return agent.successors
        
        # Use semantic similarity for intelligent selection
        try:
            # Calculate comprehensive score for each successor
            successor_scores = []
            
            for successor in agent.successors:
                if isinstance(successor, Agent):
                    # Semantic similarity score (0-1)
                    semantic_score = agent.calculate_semantic_similarity(successor)
                    
                    # Visit frequency score (lower visits = higher score, encourage exploration)
                    max_visits = max([s.visit_count for s in agent.successors if isinstance(s, Agent)] + [1])
                    visit_score = 1.0 - (successor.visit_count / max_visits) if max_visits > 0 else 1.0
                    
                    # Task matching score (based on system prompt and current task match)
                    task_match_score = self._calculate_task_match_score(agent, successor)
                    
                    # Comprehensive score (adjustable weights)
                    total_score = (
                        semantic_score * 0.4 +      # Semantic similarity weight
                        visit_score * 0.3 +         # Visit frequency weight
                        task_match_score * 0.3      # Task matching weight
                    )
                    
                    successor_scores.append((total_score, successor))
                else:
                    # Non-Agent nodes like aggregators get fixed score
                    successor_scores.append((0.5, successor))
            
            # Sort by score and select top-k
            successor_scores.sort(key=lambda x: x[0], reverse=True)
            selected = [successor for _, successor in successor_scores[:k]]
            
            # Debug info
            if len(successor_scores) > k:
                print(f"🎯 Agent {agent.agent_id} intelligently selected {k}/{len(agent.successors)} successors:")
                for i, (score, successor) in enumerate(successor_scores[:k]):
                    print(f"  {i+1}. {successor.agent_id if isinstance(successor, Agent) else 'aggregator'} (score: {score:.3f})")
            
            return selected
            
        except Exception as e:
            print(f"⚠️ Intelligent successor selection failed, using fallback strategy: {e}")
            # Fallback to simple strategy
            return agent.successors[:k]
    
    def _calculate_task_match_score(self, agent: Agent, successor: Agent) -> float:
        """Calculate task matching score"""
        try:
            if not hasattr(agent, 'input_instruction') or not agent.input_instruction:
                return 0.5  # Default medium match
            
            # Extract task keywords
            task_keywords = self._extract_task_keywords(agent.input_instruction)
            successor_keywords = self._extract_capability_keywords(successor.system_prompt)
            
            if not task_keywords or not successor_keywords:
                return 0.5
            
            # Calculate keyword matching
            common_keywords = task_keywords.intersection(successor_keywords)
            total_keywords = task_keywords.union(successor_keywords)
            
            match_score = len(common_keywords) / len(total_keywords) if total_keywords else 0.0
            
            # Consider special keyword weighting
            priority_keywords = {'analyze', 'process', 'optimize', 'check', 'verify', 'generate', 'classify', 'predict'}
            priority_matches = common_keywords.intersection(priority_keywords)
            
            if priority_matches:
                match_score += 0.2 * len(priority_matches)  # Bonus for priority keywords
            
            return min(match_score, 1.0)  # Ensure not exceeding 1.0
            
        except Exception as e:
            return 0.5  # Return medium score on error