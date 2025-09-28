"""
工具函数模块
提供一些常用的工具函数和示例LLM函数
"""

import time
import random
from typing import Any, Dict, List


def mock_llm_function(prompt: str) -> str:
    """
    模拟LLM函数，用于测试
    在实际使用中，应该替换为真实的LLM API调用
    """
    # 模拟API调用延迟
    time.sleep(0.1)
    
    # 简单的规则响应，实际应该使用真实的LLM
    prompt_lower = prompt.lower()
    
    if "生成指令" in prompt or "生成一个" in prompt:
        return "请处理以下任务并提供详细分析"
    elif "改进" in prompt or "优化" in prompt:
        return "通过增加更多细节和上下文来改进输出质量"
    elif "整合" in prompt or "聚合" in prompt:
        return "综合所有输入信息，提供统一的结论"
    elif "反馈" in prompt:
        return "输出质量良好，建议在细节方面进一步完善"
    elif "评估" in prompt:
        return "系统输出基本满足要求，但可以在准确性方面进一步提升"
    elif "true" in prompt.lower() or "false" in prompt.lower():
        return random.choice(["True", "False"])
    elif "系统提示词" in prompt:
        return "你是一个专业的助手，负责分析和处理用户的请求，提供准确和有用的信息。"
    else:
        return f"基于输入 '{prompt[:50]}...' 的响应"


def simple_tool_function(instruction: str) -> str:
    """简单的工具函数示例"""
    return f"工具执行结果：已处理指令 '{instruction}'"


def math_tool_function(instruction: str) -> str:
    """数学计算工具函数示例"""
    try:
        # 尝试提取数学表达式并计算
        if "+" in instruction:
            parts = instruction.split("+")
            if len(parts) == 2:
                a, b = float(parts[0].strip()), float(parts[1].strip())
                return f"计算结果: {a} + {b} = {a + b}"
        elif "计算" in instruction or "数学" in instruction:
            return "数学工具已准备就绪，请提供具体的计算表达式"
        else:
            return f"数学工具收到指令: {instruction}"
    except:
        return f"数学工具处理指令: {instruction}"


def text_analysis_tool_function(instruction: str) -> str:
    """文本分析工具函数示例"""
    word_count = len(instruction.split())
    char_count = len(instruction)
    return f"文本分析结果: 字符数={char_count}, 词数={word_count}, 内容='{instruction[:30]}...'"


def create_environment_function(expected_keywords: List[str] = None) -> callable:
    """创建环境函数"""
    expected_keywords = expected_keywords or ["分析", "处理", "结果"]
    
    def environment_function(result: str) -> str:
        score = 0
        feedback_parts = []
        
        # 检查关键词
        for keyword in expected_keywords:
            if keyword in result:
                score += 1
                feedback_parts.append(f"包含关键词'{keyword}'")
        
        # 检查长度
        if len(result) > 50:
            score += 1
            feedback_parts.append("输出长度充足")
        
        # 生成反馈
        if score >= len(expected_keywords):
            feedback = f"优秀: {', '.join(feedback_parts)}"
        elif score >= len(expected_keywords) // 2:
            feedback = f"良好: {', '.join(feedback_parts)}"
        else:
            feedback = f"需要改进: 缺少关键元素"
        
        return f"环境反馈: {feedback} (得分: {score}/{len(expected_keywords) + 1})"
    
    return environment_function


def print_network_visualization(textgrad_instance):
    """打印网络可视化"""
    print("\n=== 智能体网络结构 ===")
    network_info = textgrad_instance.network.get_network_info()
    
    print(f"总智能体数: {network_info['total_agents']}")
    print(f"源点智能体: {network_info['source_agent']}")
    print(f"包含聚合器: {network_info['has_aggregator']}")
    
    print("\n连接关系:")
    for agent_id, successors in network_info['connections'].items():
        if successors:
            for successor in successors:
                print(f"  {agent_id} -> {successor}")
        else:
            print(f"  {agent_id} (无后继)")
    print("=" * 30)


def print_iteration_summary(iteration_result: Dict[str, Any], iteration_num: int):
    """打印迭代摘要"""
    print(f"\n=== 第 {iteration_num} 次迭代摘要 ===")
    print(f"指令: {iteration_result['instruction']}")
    print(f"前向结果: {iteration_result['forward_result']}")
    print(f"环境反馈: {iteration_result['environment_feedback']}")
    print(f"损失评估: {iteration_result['loss']}")
    print(f"成功: {iteration_result['success']}")
    
    # 网络变化
    before = iteration_result['network_info_before']
    after = iteration_result.get('network_info_after', {})
    
    if before and after:
        if before['total_agents'] != after.get('total_agents', 0):
            print(f"网络变化: 智能体数量从 {before['total_agents']} 变为 {after.get('total_agents', 0)}")
    
    print("=" * 40)


def validate_network_connectivity(network):
    """验证网络连通性"""
    if not network.source_agent:
        return False, "缺少源点智能体"
    
    if not network.aggregator:
        return False, "缺少聚合器"
    
    # 检查是否存在从源点到聚合器的路径
    visited = set()
    
    def dfs(agent):
        if agent in visited:
            return False
        visited.add(agent)
        
        for successor in agent.successors:
            if successor == network.aggregator:
                return True
            if isinstance(successor, type(agent)) and dfs(successor):
                return True
        return False
    
    if dfs(network.source_agent):
        return True, "网络连通性良好"
    else:
        return False, "从源点无法到达聚合器" 