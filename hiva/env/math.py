from heaven.engines.openai_engine import OpenAIEngine
from functools import partial

engine = OpenAIEngine(model_name="gpt-4o-mini", base_url="https://c-z0-api-01.hash070.com/v1")

def math_env_function(result, problem, solution):
    # 使用engine校验answer是否正确并提供反馈
    result = engine.generate(
        content=f"Problem: {problem}\nSolution: {solution}\nAnswer: {result}",
        system_prompt=f"You are a math checker. You are given a problem, a correct solution, and an answer from a student. You need to check if the answer is correct. You should give some useful feedback about the answer, do not directly give the correct answer.",
        max_tokens=1000,
        temperature=1
    )
    return result

def make_math_env(problem, solution):
    return partial(math_env_function, problem=problem, solution=solution)