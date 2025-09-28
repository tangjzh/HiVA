from heaven.engines.openai_engine import OpenAIEngine
from functools import partial

engine = OpenAIEngine(model_name="gpt-4o-mini", base_url="https://c-z0-api-01.hash070.com/v1")

def env_function(result, question, answer):
    # 使用engine校验answer是否正确并提供反馈
    
    result = engine.generate(
        content=f"Question: {question}\nCorrect Answer: {answer}\nStudent Answer: {result}",
        system_prompt=f"You are a strict teacher. You are given a question, a correct answer, and a student answer. You need to check if the student answer is correct. You should give some useful feedback about the answer, do not directly give the correct answer.",
        max_tokens=512,
        temperature=1
    )
    return result

def make_env(question, answer):
    return partial(env_function, question=question, answer=answer)