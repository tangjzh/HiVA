import time
import random
from typing import Dict, List, Optional, Any
from .base_engine import BaseLLMEngine


class MockEngine(BaseLLMEngine):
    """Mock LLM engine for testing and demonstration"""
    
    def __init__(self,
                 model_name: str = "mock-gpt",
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 cache_enabled: bool = True,
                 cache_dir: str = ".cache",
                 response_delay: float = 0.1,
                 smart_responses: bool = True):
        
        super().__init__(model_name, temperature, max_tokens, cache_enabled, cache_dir)
        
        self.response_delay = response_delay
        self.smart_responses = smart_responses
        
        # Predefined smart response templates
        self.response_templates = {
            # Optimization related
            "optimize": [
                "To improve this output, consider adding more specific details and concrete examples.",
                "The response could be enhanced by providing step-by-step reasoning and clearer structure.",
                "Consider incorporating domain-specific terminology and more precise language.",
                "Adding quantitative metrics and measurable outcomes would strengthen this response."
            ],
            
            # Evaluation related
            "evaluate": [
                "This response demonstrates good understanding but lacks depth in analysis.",
                "The output shows partial accuracy with room for improvement in clarity and completeness.",
                "Overall quality is acceptable, though the reasoning could be more systematic.",
                "The response addresses the main points but could benefit from more detailed explanations."
            ],
            
            # Feedback related
            "feedback": [
                "The current approach is on the right track but needs refinement in execution.",
                "Strong foundational elements present, but implementation details require enhancement.",
                "Good conceptual framework, though practical application could be more robust.",
                "Solid base understanding evident, but depth and nuance need development."
            ],
            
            # Instruction related
            "instruction": [
                "Analyze the following data comprehensively and provide actionable insights.",
                "Process this information systematically and deliver well-structured conclusions.",
                "Examine the given content thoroughly and generate detailed recommendations.",
                "Review this material carefully and produce comprehensive analysis with supporting evidence."
            ],
            
            # System prompt related
            "system_prompt": [
                "You are an expert analyst with deep domain knowledge and strong analytical capabilities.",
                "You are a systematic problem-solver who provides thorough, evidence-based responses.",
                "You are a detail-oriented professional who delivers comprehensive and accurate analysis.",
                "You are a strategic thinker capable of providing insightful and actionable recommendations."
            ],
            
            # Aggregation related
            "aggregate": [
                "Synthesizing multiple analytical perspectives reveals several key patterns and insights.",
                "Integration of diverse analytical outputs demonstrates convergent findings across multiple dimensions.",
                "Comprehensive review of analytical components indicates robust consensus on primary conclusions.",
                "Unified analysis of multiple expert inputs yields coherent and actionable strategic recommendations."
            ],
            
            # Default responses
            "default": [
                "Based on the provided context, here is a comprehensive analytical response.",
                "After careful consideration of the input parameters, the following conclusions emerge.",
                "Through systematic analysis of the given information, several key insights can be identified.",
                "Detailed examination of the presented data yields the following structured recommendations."
            ]
        }
    
    def _detect_intent(self, messages: List[Dict[str, str]]) -> str:
        """Detect user intent"""
        content = " ".join([msg.get("content", "") for msg in messages]).lower()
        
        # Check keywords
        if any(word in content for word in ["optimize", "improve", "enhance", "better"]):
            return "optimize"
        elif any(word in content for word in ["evaluate", "assess", "quality", "good", "bad"]):
            return "evaluate"
        elif any(word in content for word in ["feedback", "comment", "suggest", "recommend"]):
            return "feedback"
        elif any(word in content for word in ["instruction", "task", "do", "process"]):
            return "instruction"
        elif any(word in content for word in ["system prompt", "role", "you are"]):
            return "system_prompt"
        elif any(word in content for word in ["aggregate", "combine", "integrate", "merge"]):
            return "aggregate"
        else:
            return "default"
    
    def _generate_smart_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate smart response"""
        intent = self._detect_intent(messages)
        
        # Choose appropriate response template
        templates = self.response_templates.get(intent, self.response_templates["default"])
        base_response = random.choice(templates)
        
        # Add context-related content
        content = " ".join([msg.get("content", "") for msg in messages])
        
        # Adjust response based on content length and complexity
        if len(content) > 200:
            base_response += " The complexity of the input requires detailed multi-faceted analysis."
        elif len(content) < 50:
            base_response += " Given the concise input, a focused response is most appropriate."
        
        # Add dynamic elements
        if "数据" in content or "data" in content.lower():
            base_response += " Data-driven insights indicate significant potential for improvement."
        
        if "分析" in content or "analysis" in content.lower():
            base_response += " Analytical depth and methodological rigor are essential for optimal outcomes."
        
        return base_response
    
    def _generate_impl(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Mock response generation"""
        # Get the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        # Generate different types of responses based on message content
        if "decide_network_structure_change" in user_message or "decide what network structure change" in user_message:
            return self._generate_structure_decision(user_message)
        elif "Create a focused system prompt" in user_message:
            return self._generate_system_prompt(user_message)
        elif "EVALUATION_PROMPT_TEMPLATE" in user_message or "Evaluate how well" in user_message:
            return self._generate_evaluation_feedback(user_message)
        elif "GRADIENT_PROMPT_TEMPLATE" in user_message or "improvement suggestions" in user_message:
            return self._generate_gradient_feedback(user_message)
        elif "IMPROVEMENT_PROMPT_TEMPLATE" in user_message or "Apply the gradients" in user_message:
            return self._generate_improved_variable(user_message)
        elif "analyze feedback" in user_message.lower() or "improvement suggestions" in user_message.lower():
            return self._generate_analysis_feedback(user_message)
        else:
            return self._generate_default_response(user_message)
    
    def _generate_structure_decision(self, message: str) -> str:
        """Generate network structure decision"""
        import random
        
        # Decide structure changes based on feedback
        if "专门" in message or "专业" in message or "深入" in message or "添加" in message:
            specializations = [
                "专门进行深度数据分析和统计建模",
                "专注于用户体验设计和界面优化", 
                "专门处理市场分析和竞争情报",
                "专注于技术架构和系统设计",
                "专门进行内容创作和文案优化"
            ]
            return f"ADD_SUCCESSOR: {random.choice(specializations)}"
        
        elif "冗余" in message or "重复" in message or "简化" in message:
            return "REMOVE_SUCCESSOR: 0"
        
        else:
            reasons = [
                "当前结构已经能够满足需求",
                "现有智能体协作良好，无需调整",
                "结构变化可能影响整体性能"
            ]
            return f"NO_CHANGE: {random.choice(reasons)}"
    
    def _generate_system_prompt(self, message: str) -> str:
        """Generate system prompt for new agent"""
        if "数据分析" in message:
            return "你是一个专业的数据分析师，专门负责深度数据挖掘、统计分析和预测建模。你擅长从复杂数据中提取有价值的洞察和趋势。"
        elif "用户体验" in message or "UI" in message:
            return "你是一个用户体验设计专家，专注于界面设计、用户交互和可用性优化。你能够从用户角度思考并设计直观易用的解决方案。"
        elif "市场" in message:
            return "你是一个市场研究专家，专门分析市场趋势、竞争格局和消费者行为。你能够提供深入的市场洞察和战略建议。"
        else:
            return "你是一个专业的领域专家，负责提供专业的分析和建议。你有丰富的经验和深入的专业知识。"
    
    def _generate_evaluation_feedback(self, message: str) -> str:
        """Generate evaluation feedback"""
        evaluations = [
            "<FEEDBACK>\n结果基本满足要求，但缺乏深度分析。建议增加更多专业观点和详细数据支撑。结果的结构清晰，但在某些专业领域需要更深入的探讨。\n</FEEDBACK>",
            "<FEEDBACK>\n结果内容丰富，涵盖了主要方面。但在用户体验和技术实现方面需要更专业的分析。建议添加专门的专家视角来完善结果。\n</FEEDBACK>",
            "<FEEDBACK>\n结果质量良好，但过于宽泛。需要更加专业化的分析，特别是在数据分析和市场洞察方面。建议引入专门的分析模块。\n</FEEDBACK>"
        ]
        import random
        return random.choice(evaluations)
    
    def _generate_gradient_feedback(self, message: str) -> str:
        """Generate gradient feedback"""
        gradients = [
            "<FEEDBACK>\n基于分析，需要在以下方面改进：\n1. 增加专业化的深度分析模块\n2. 提供更具体的数据支撑和案例\n3. 完善用户体验设计考虑\n\n建议：\n- 添加专门的数据分析智能体\n- 引入用户体验专家角色\n- 加强不同模块间的协作\n</FEEDBACK>",
            "<FEEDBACK>\n分析显示当前方案存在以下改进空间：\n1. 缺乏深度的专业洞察\n2. 需要更全面的多角度分析\n3. 建议增加专业化分工\n\n推荐改进：\n- 建立专业化的分析团队\n- 加强各领域专家的参与\n- 优化分析流程和方法\n</FEEDBACK>"
        ]
        import random
        return random.choice(gradients)
    
    def _generate_improved_variable(self, message: str) -> str:
        """Generate improved variable"""
        if "system prompt" in message.lower():
            improved_prompts = [
                "<IMPROVED_VARIABLE>\n你是一个高级综合分析师，专门负责复杂问题的多维度分析。你擅长整合不同专业领域的观点，提供全面而深入的解决方案。你会主动寻求专业化的支持，确保分析的准确性和完整性。\n</IMPROVED_VARIABLE>",
                "<IMPROVED_VARIABLE>\n你是一个专业的问题解决专家，具备跨领域的知识和丰富的实践经验。你能够识别问题的核心，协调不同专业的资源，提供系统性的解决方案。你重视数据驱动的决策和用户导向的思维。\n</IMPROVED_VARIABLE>"
            ]
        else:
            improved_prompts = [
                "<IMPROVED_VARIABLE>\n改进后的工具函数将提供更专业的分析能力，包括数据处理、用户体验评估和市场洞察功能。\n</IMPROVED_VARIABLE>"
            ]
        
        import random
        return random.choice(improved_prompts)
    
    def _generate_analysis_feedback(self, message: str) -> str:
        """Generate analysis feedback"""
        return """SYSTEM_PROMPT_FEEDBACK: 当前角色定义较为宽泛，建议增加更具体的专业领域指导，明确分析方法和标准
TOOL_FEEDBACK: 工具功能基础，建议增加专业化的分析工具和数据处理能力
OVERALL_FEEDBACK: 整体表现良好但缺乏专业深度，建议添加专门的分析模块来增强专业能力和分析质量"""
    
    def _generate_default_response(self, message: str) -> str:
        """Generate default response"""
        default_responses = [
            "这是一个模拟的分析结果。需要深入的数据分析和多角度的专业评估来提供更准确的结论。",
            "基于当前信息的分析表明，这个问题需要综合考虑技术、用户体验和市场等多个维度。",
            "分析显示这是一个复杂的问题，需要专业化的处理和更详细的研究。建议引入专门的分析工具和专家意见。"
        ]
        import random
        return random.choice(default_responses)
    
    def set_response_delay(self, delay: float):
        """Set response delay"""
        self.response_delay = delay
    
    def set_smart_responses(self, enabled: bool):
        """Enable/disable smart responses"""
        self.smart_responses = enabled
    
    def add_response_template(self, intent: str, templates: List[str]):
        """Add new response templates"""
        if intent not in self.response_templates:
            self.response_templates[intent] = []
        self.response_templates[intent].extend(templates)
    
    def get_available_intents(self) -> List[str]:
        """Get available intent types"""
        return list(self.response_templates.keys())
    
    def _generate_mock_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate contextually appropriate mock responses using TextGrad standard format"""
        prompt_lower = prompt.lower()
        
        # Detect TextGrad template usage and respond with appropriate format
        if "evaluate" in prompt_lower and "feedback" in prompt_lower:
            return self._generate_evaluation_feedback(prompt)
        elif "gradient" in prompt_lower and "improvement" in prompt_lower:
            return self._generate_gradient_feedback(prompt)
        elif "improved_variable" in prompt_lower or "apply the gradients" in prompt_lower:
            return self._generate_improved_variable(prompt)
        elif "predecessor" in prompt_lower and "successor" in prompt_lower:
            return self._generate_predecessor_feedback(prompt)
        
        # Business analysis specific responses
        if "business analysis" in prompt_lower and "finance app" in prompt_lower:
            return self._generate_business_analysis()
        
        # Code optimization responses
        if "code" in prompt_lower and ("optimization" in prompt_lower or "improve" in prompt_lower):
            return self._generate_code_optimization()
        
        # General instruction following
        if any(word in prompt_lower for word in ["create", "analyze", "develop", "design"]):
            return self._generate_general_analysis(prompt)
        
        # Default fallback
        return "I understand your request. As a mock engine, I'm providing a simulated response that would typically come from an advanced LLM."
    
    def _generate_predecessor_feedback(self, prompt: str) -> str:
        """Generate predecessor feedback in TextGrad format"""
        return """Feedback for predecessor agent optimization has been generated.

<FEEDBACK>
To improve the predecessor agent and enhance successor performance:

1. Instruction clarity - Provide more specific and structured prompts to successors
2. Context enhancement - Include relevant background information and constraints
3. Output formatting - Specify desired response structure and detail level
4. Quality control - Implement validation criteria for successor outputs

The predecessor should be modified to: Provide more structured and detailed instructions that include specific requirements, format expectations, and evaluation criteria to enable successors to generate higher-quality, more targeted responses.
</FEEDBACK>"""

    def _generate_business_analysis(self) -> str:
        """Generate business analysis response"""
        return """# AI-Powered Personal Finance App: Business Analysis

## Market Overview
The personal finance app market is experiencing significant growth, driven by increased financial awareness and smartphone adoption. Our target market focuses on young professionals aged 25-40 seeking intelligent financial guidance.

## Competitive Landscape
Key competitors include Mint (budget tracking), YNAB (zero-based budgeting), and Personal Capital (investment focus). Our differentiation lies in AI-powered predictive insights and personalized coaching.

## Revenue Model
- Freemium model: Basic features free, premium AI coaching $9.99/month
- Target: 50,000 paying subscribers by year 2
- Customer acquisition through digital marketing and referral programs

## Risk Assessment
- Market competition from established players
- Regulatory compliance for financial data
- Technical challenges in AI model accuracy
- User adoption and retention rates"""
    
    def _generate_code_optimization(self) -> str:
        """Generate code optimization response"""
        return """# Code Optimization Analysis

The provided code shows good structure but can be improved in several areas:

## Performance Optimizations
- Implement caching for frequently accessed data
- Use more efficient algorithms for data processing
- Optimize database queries with proper indexing

## Code Quality
- Add comprehensive error handling
- Improve variable naming for clarity
- Implement proper logging and monitoring

## Security Enhancements
- Validate all user inputs
- Implement proper authentication and authorization
- Use secure communication protocols"""
    
    def _generate_general_analysis(self, prompt: str) -> str:
        """Generate general analysis response"""
        return f"""# Analysis Report

Based on your request, I've analyzed the following areas:

## Key Findings
- The subject matter requires comprehensive evaluation across multiple dimensions
- Several optimization opportunities have been identified
- Implementation should follow best practices and industry standards

## Recommendations
- Conduct thorough research on market conditions
- Implement robust testing and validation procedures
- Consider scalability and future growth requirements
- Ensure compliance with relevant regulations and standards

## Next Steps
- Develop detailed implementation plan
- Allocate appropriate resources for execution
- Establish monitoring and evaluation metrics
- Create contingency plans for risk mitigation

This analysis provides a foundation for informed decision-making and strategic planning."""

    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 1000, 
                temperature: float = 0.7, **kwargs) -> str:
        """Generate mock response with simulated delay"""
        # Simulate API call delay
        time.sleep(random.uniform(0.1, 0.5))
        
        context = {
            'system_prompt': system_prompt,
            'max_tokens': max_tokens,
            'temperature': temperature,
            **kwargs
        }
        
        # Use the new structured response generator
        return self._generate_mock_response(prompt, context) 