"""
Predefined tools module
Provides common tool functions for agents
"""

import re
import json
import math
from typing import Any, Dict, List, Union
from .tool import ToolDefinition, ToolType


def create_data_analysis_tool() -> ToolDefinition:
    """Create data analysis tool"""
    
    def data_analysis_tool(input_data, **kwargs):
        """
        Data analysis tool - analyze numerical data in text
        
        Args:
            input_data: Text containing data
            **kwargs: Additional parameters
        
        Returns:
            Analysis result string
        """
        try:
            text = str(input_data)
            
            # Extract numbers
            numbers = re.findall(r'-?\d+\.?\d*', text)
            numeric_data = [float(n) for n in numbers if n]
            
            if not numeric_data:
                return f"Data Analysis Result: No numerical data found in text '{text[:50]}...'"
            
            # Calculate statistics
            total = sum(numeric_data)
            count = len(numeric_data)
            avg = total / count
            max_val = max(numeric_data)
            min_val = min(numeric_data)
            
            # Calculate growth rate (if multiple data points)
            growth_rate = 0
            if count >= 2:
                growth_rate = ((numeric_data[-1] - numeric_data[0]) / abs(numeric_data[0])) * 100
            
            result = f"""Data Analysis Report:
📊 Data Overview:
- Data points found: {count}
- Data range: {min_val} - {max_val}
- Total: {total:.2f}
- Average: {avg:.2f}
- Growth rate: {growth_rate:.1f}%

📈 Insights:
- Maximum value: {max_val} (proportion {(max_val/total*100):.1f}%)
- Minimum value: {min_val}
- Data dispersion: {'High' if (max_val - min_val) > avg else 'Low'}

Raw data: {numeric_data}
"""
            return result
            
        except Exception as e:
            return f"Data analysis error: {str(e)}"
    
    return ToolDefinition(
        name="data_analysis_tool",
        description="Analyze numerical data in text, calculate statistics and trends",
        tool_type=ToolType.DATA_PROCESSING,
        function=data_analysis_tool,
        parameters={"input_data": "str"},
        return_type="string",
        examples=[
            "Analyze sales data: Q1 sales 1M, Q2 sales 1.2M, Q3 sales 1.5M",
            "User growth: January 1000 users, February 1200 users, March 1500 users"
        ],
        safety_level="safe"
    )


def create_text_analysis_tool() -> ToolDefinition:
    """Create text analysis tool"""
    
    def text_analysis_tool(input_data, **kwargs):
        """
        Text analysis tool - analyze text content and structure
        
        Args:
            input_data: Text to analyze
            **kwargs: Additional parameters
        
        Returns:
            Text analysis result
        """
        try:
            text = str(input_data)
            
            # Basic statistics
            char_count = len(text)
            word_count = len(text.split())
            sentence_count = len(re.findall(r'[.!?]+', text))
            paragraph_count = len([p for p in text.split('\n') if p.strip()])
            
            # Keyword extraction (simple implementation)
            words = re.findall(r'\b\w+\b', text.lower())
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Only count words with length > 3
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get high-frequency words
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Sentiment analysis (simple implementation)
            positive_words = ['good', 'excellent', 'success', 'growth', 'improve', 'satisfied', 'great', 'amazing', 'perfect']
            negative_words = ['bad', 'failure', 'decrease', 'problem', 'error', 'poor', 'terrible', 'awful', 'worst']
            
            positive_count = sum(1 for word in positive_words if word in text.lower())
            negative_count = sum(1 for word in negative_words if word in text.lower())
            
            sentiment = "Neutral"
            if positive_count > negative_count:
                sentiment = "Positive"
            elif negative_count > positive_count:
                sentiment = "Negative"
            
            result = f"""Text Analysis Report:
📝 Basic Statistics:
- Character count: {char_count}
- Word count: {word_count}
- Sentence count: {sentence_count}
- Paragraph count: {paragraph_count}

🔍 Content Analysis:
- Average sentence length: {word_count/max(sentence_count, 1):.1f} words
- Content density: {'High' if word_count > 100 else 'Medium' if word_count > 50 else 'Low'}
- Sentiment: {sentiment}

🏷️ Top Keywords (frequency):
{chr(10).join([f'- {word}: {count} times' for word, count in top_words[:3]])}

📊 Readability: {'Complex' if word_count/max(sentence_count, 1) > 15 else 'Moderate'}
"""
            return result
            
        except Exception as e:
            return f"Text analysis error: {str(e)}"
    
    return ToolDefinition(
        name="text_analysis_tool",
        description="Analyze text content, structure, keywords and sentiment",
        tool_type=ToolType.TEXT_ANALYSIS,
        function=text_analysis_tool,
        parameters={"input_data": "str"},
        return_type="string",
        examples=[
            "Analyze user feedback text",
            "Analyze product descriptions and features"
        ],
        safety_level="safe"
    )


def create_mathematical_tool() -> ToolDefinition:
    """Create mathematical calculation tool"""
    
    def mathematical_tool(input_data, **kwargs):
        """
        Mathematical calculation tool - perform various mathematical operations
        
        Args:
            input_data: Text containing mathematical expressions or data
            **kwargs: Additional parameters
        
        Returns:
            Calculation results
        """
        try:
            text = str(input_data)
            
            # Extract numbers
            numbers = re.findall(r'-?\d+\.?\d*', text)
            numeric_data = [float(n) for n in numbers if n]
            
            if not numeric_data:
                return f"Mathematical calculation: No numerical values found in '{text[:50]}...'"
            
            # Perform basic calculations
            results = []
            
            if len(numeric_data) >= 2:
                # Basic operations
                results.append(f"Addition: {numeric_data[0]} + {numeric_data[1]} = {numeric_data[0] + numeric_data[1]}")
                results.append(f"Subtraction: {numeric_data[0]} - {numeric_data[1]} = {numeric_data[0] - numeric_data[1]}")
                results.append(f"Multiplication: {numeric_data[0]} × {numeric_data[1]} = {numeric_data[0] * numeric_data[1]}")
                
                if numeric_data[1] != 0:
                    results.append(f"Division: {numeric_data[0]} ÷ {numeric_data[1]} = {numeric_data[0] / numeric_data[1]:.4f}")
                
                # Percentage change
                if numeric_data[0] != 0:
                    change = ((numeric_data[1] - numeric_data[0]) / numeric_data[0]) * 100
                    results.append(f"Change rate: {change:.2f}%")
            
            # Statistical functions
            if len(numeric_data) >= 1:
                results.append(f"Sum: {sum(numeric_data)}")
                results.append(f"Average: {sum(numeric_data) / len(numeric_data):.4f}")
                results.append(f"Maximum: {max(numeric_data)}")
                results.append(f"Minimum: {min(numeric_data)}")
                
                # Standard deviation
                mean = sum(numeric_data) / len(numeric_data)
                variance = sum((x - mean) ** 2 for x in numeric_data) / len(numeric_data)
                std_dev = math.sqrt(variance)
                results.append(f"Standard deviation: {std_dev:.4f}")
            
            # Compound interest calculation (if 3 numbers, assume principal, rate, time)
            if len(numeric_data) >= 3:
                principal, rate, time = numeric_data[0], numeric_data[1]/100, numeric_data[2]
                compound = principal * (1 + rate) ** time
                results.append(f"Compound interest: Principal {principal}, Rate {rate*100}%, Time {time} = {compound:.2f}")
            
            return f"Mathematical Calculation Results:\n🔢 Input data: {numeric_data}\n\n📊 Calculation results:\n" + "\n".join(results)
            
        except Exception as e:
            return f"Mathematical calculation error: {str(e)}"
    
    return ToolDefinition(
        name="mathematical_tool",
        description="Perform mathematical operations including basic arithmetic, statistics and financial calculations",
        tool_type=ToolType.MATHEMATICAL,
        function=mathematical_tool,
        parameters={"input_data": "str"},
        return_type="string",
        examples=[
            "Calculate relationship between 100 and 120",
            "Analyze data sequence: 10, 15, 20, 25, 30"
        ],
        safety_level="safe"
    )


def create_business_analysis_tool() -> ToolDefinition:
    """Create business analysis tool"""
    
    def business_analysis_tool(input_data, **kwargs):
        """
        Business analysis tool - analyze business data and trends
        
        Args:
            input_data: Business data or description
            **kwargs: Additional parameters
        
        Returns:
            Business analysis result
        """
        try:
            text = str(input_data).lower()
            
            # Extract numbers
            numbers = re.findall(r'-?\d+\.?\d*', str(input_data))
            numeric_data = [float(n) for n in numbers if n]
            
            # Identify business metric keywords
            metrics = {
                'revenue': ['revenue', 'sales', 'income', 'earnings'],
                'profit': ['profit', 'margin', 'earnings', 'gains'],
                'users': ['user', 'customer', 'client', 'subscriber'],
                'growth': ['growth', 'increase', 'expansion', 'rise'],
                'cost': ['cost', 'expense', 'spending', 'investment'],
                'market': ['market', 'share', 'competition', 'segment']
            }
            
            identified_metrics = []
            for metric, keywords in metrics.items():
                if any(keyword in text for keyword in keywords):
                    identified_metrics.append(metric)
            
            # Analysis results
            analysis_results = []
            
            # Numerical analysis
            if numeric_data:
                if len(numeric_data) >= 2:
                    change = ((numeric_data[-1] - numeric_data[0]) / abs(numeric_data[0])) * 100
                    trend = "Upward" if change > 0 else "Downward" if change < 0 else "Stable"
                    analysis_results.append(f"📈 Trend Analysis: {trend} ({change:+.1f}%)")
                
                analysis_results.append(f"📊 Data Range: {min(numeric_data):.2f} - {max(numeric_data):.2f}")
                
                # ROI estimation
                if 'profit' in identified_metrics and 'cost' in identified_metrics and len(numeric_data) >= 2:
                    roi = ((numeric_data[1] - numeric_data[0]) / numeric_data[0]) * 100
                    analysis_results.append(f"💰 ROI Estimate: {roi:.1f}%")
            
            # Business insights
            insights = []
            if 'growth' in identified_metrics:
                insights.append("🚀 Focus on growth metrics, recommend developing specific growth strategies")
            if 'users' in identified_metrics:
                insights.append("👥 User-related data, recommend analyzing user retention and acquisition costs")
            if 'revenue' in identified_metrics:
                insights.append("💵 Revenue data, recommend monitoring revenue diversification and seasonality")
            if 'market' in identified_metrics:
                insights.append("🎯 Market data, recommend competitive analysis")
            
            # Recommendations
            recommendations = []
            if numeric_data and len(numeric_data) >= 2:
                if change > 20:
                    recommendations.append("High growth momentum, recommend increasing investment and optimizing operations")
                elif change > 5:
                    recommendations.append("Stable growth, recommend maintaining current strategy and seeking new opportunities")
                elif change < -10:
                    recommendations.append("Declining trend, recommend urgent evaluation and strategy adjustment")
                else:
                    recommendations.append("Stable trend, recommend analyzing market opportunities")
            
            result = f"""Business Analysis Report:
🎯 Identified Business Metrics: {', '.join(identified_metrics) if identified_metrics else 'No specific metrics'}

📈 Quantitative Analysis:
{chr(10).join(analysis_results) if analysis_results else '- Insufficient data for quantitative analysis'}

💡 Business Insights:
{chr(10).join(insights) if insights else '- Recommend providing more specific business data'}

🎯 Recommended Actions:
{chr(10).join(recommendations) if recommendations else '- Need more data to provide specific recommendations'}

📊 Raw Data: {numeric_data if numeric_data else 'No numerical data'}
"""
            return result
            
        except Exception as e:
            return f"Business analysis error: {str(e)}"
    
    return ToolDefinition(
        name="business_analysis_tool",
        description="Analyze business data, identify trends and provide business insights and recommendations",
        tool_type=ToolType.DATA_PROCESSING,
        function=business_analysis_tool,
        parameters={"input_data": "str"},
        return_type="string",
        examples=[
            "Analyze Q1 to Q4 revenue data: 1M, 1.2M, 1.5M, 1.8M",
            "User growth: January 1000, February 1200, March 1800"
        ],
        safety_level="safe"
    )


# Tool factory functions
def get_predefined_tools() -> Dict[str, ToolDefinition]:
    """Get all predefined tools"""
    return {
        "data_analysis": create_data_analysis_tool(),
        "text_analysis": create_text_analysis_tool(),
        "mathematical": create_mathematical_tool(),
        "business_analysis": create_business_analysis_tool()
    }


def create_tool_by_type(tool_type: ToolType) -> ToolDefinition:
    """Create tool by type"""
    tools_map = {
        ToolType.DATA_PROCESSING: create_data_analysis_tool(),
        ToolType.TEXT_ANALYSIS: create_text_analysis_tool(),
        ToolType.MATHEMATICAL: create_mathematical_tool(),
        ToolType.CUSTOM: create_business_analysis_tool()
    }
    
    return tools_map.get(tool_type, create_data_analysis_tool()) 