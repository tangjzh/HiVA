from datasets import load_dataset
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import random
import json
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from hiva.core.textgrad import TextGrad
from hiva.engines.openai_engine import OpenAIEngine

dataset_name = "EleutherAI/hendrycks_math"
dataset = load_dataset(dataset_name, 'algebra', split="test")

forward_engine = OpenAIEngine(model_name="gpt-4o-mini", temperature=1)
backward_engine = OpenAIEngine(model_name="gpt-4o-mini")

def math_env_function(result, problem, solution, engine):
    result = engine.generate(
        content=f"Problem: {problem}\nSolution: {solution}\nAnswer: {result}",
        system_prompt=f"You are a math teacher. You are given a problem, a correct solution, and an answer from a student. You need to check if the answer is correct. You should give some useful feedback about the answer, DONNOT directly give the correct answer and do not show the correct answer.",
        max_tokens=1000,
        temperature=1
    )
    return result

def verify_answer(result, problem, solution, engine):
    response = engine.generate(
        content=f"Correct Solution: {solution}\nStudent Answer: {result}",
        system_prompt="You are a math checker. Compare the student's answer with the correct solution and check if the student's answer is correct. Respond with ONLY 'correct' or 'wrong'.",
        max_tokens=10,
        temperature=1
    )
    return "correct" in response.lower()

def process_example(example):
    try:
        answer_type = example['type']
        
        criteria = f"Evaluate how well the content addresses the initial instruction, consider the correctness, completeness, and effectiveness. If there seems to be some uncetrainties, try to fix them using new tools and workflows. The answer type should be {answer_type}."
        initial_instruction = f"{example['problem']}\n### Requirement: You should only provide the final {answer_type} answer:"
        initial_system_prompt = f"You are a math expert. You are given a question and an answer. You need to try to solve problems based on the given instruction. You can use tools or build a team to solve the problem."

        env_func = partial(math_env_function, engine=forward_engine, problem=example['problem'], solution=example['solution'])
        textgrad = TextGrad(forward_engine=forward_engine, backward_engine=backward_engine, environment_function=env_func)
        textgrad.initialize_network(initial_system_prompt=initial_system_prompt)
        textgrad.optimize(initial_instruction=initial_instruction, criteria=criteria, k=2, max_iterations=3)

        result = textgrad(initial_instruction, k=2)
        
        is_correct = verify_answer(result, example['problem'], example['solution'], backward_engine)
        
        return {
            'problem': example['problem'],
            'generated_answer': result,
            'correct_solution': example['solution'],
            'is_correct': is_correct
        }
    except Exception as e:
        print(f"Error processing example: {str(e)}")
        return None

examples = dataset.shuffle(seed=random.randint(0, 1000000)).select(range(50))
results = []
correct_count = 0

with ProcessPoolExecutor(max_workers=32) as executor:
    future_to_example = {executor.submit(process_example, example): example for example in examples}
    
    for future in as_completed(future_to_example):
        result = future.result()
        if result:
            results.append(result)
            if result['is_correct']:
                correct_count += 1

accuracy = correct_count / len(results) if results else 0
print(f"Accuracy: {accuracy * 100:.2f}%")

output_dir = "output/math_4omini_algebra"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "results.json")

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump({
        'accuracy': accuracy,
        'total_samples': len(results),
        'correct_samples': correct_count,
        'detailed_results': results
    }, f, ensure_ascii=False, indent=2)

print(f"Results saved to: {output_path}")