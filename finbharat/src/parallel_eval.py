#!/usr/bin/env python3
"""Parallel FinBharat evaluation - runs all models concurrently."""
import sys, json, time, os, glob, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

KEY = sys.argv[1]
MODELS = ['deepseek/deepseek-v3.2', 'qwen/qwen3.5-397b-a17b',
          'mistralai/mistral-large-2512', 'meta-llama/llama-3.3-70b-instruct:free']
LANGS = ['en', 'hi', 'hinglish', 'te', 'bn', 'ta']

def query(prompt, model):
    r = requests.post('https://openrouter.ai/api/v1/chat/completions',
        headers={'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json'},
        json={'model': model, 'messages': [{'role': 'user', 'content': prompt}],
              'temperature': 0, 'max_tokens': 400},
        timeout=(10, 60))
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']

def eval_one(task):
    """Evaluate one question for one model in one language."""
    model, lang, q = task
    qk = f'question_{lang}'
    if qk not in q:
        return None

    prompt = f"You are a financial advisor for Indian markets. Give a specific answer with exact numbers and Indian regulations.\n\nQuestion: {q[qk]}\n\nAnswer concisely:"
    if q.get('system_prompt'):
        prompt = f"{q['system_prompt']}\n\nQuestion: {q[qk]}\n\nAnswer:"

    mn = model.split('/')[-1][:25]
    try:
        resp = query(prompt, model)
        return {'qid': q['id'], 'cat': q['category'], 'model': mn,
                'lang': lang, 'danger': q.get('danger_level', ''), 'response': resp}
    except Exception as e:
        return {'qid': q['id'], 'model': mn, 'lang': lang, 'error': str(e)[:80]}

# Load questions
questions = []
for f in sorted(glob.glob('data/questions_*.json')):
    questions.extend(json.load(open(f)))
print(f'Loaded {len(questions)} questions', flush=True)

# Build task list
tasks = []
for model in MODELS:
    for lang in LANGS:
        for q in questions:
            tasks.append((model, lang, q))

total = len(tasks)
print(f'Total tasks: {total} ({len(MODELS)} models x {len(LANGS)} langs x {len(questions)} questions)', flush=True)
print(f'Running with 8 parallel threads...', flush=True)

results = []
errors = 0
done = 0

os.makedirs('results', exist_ok=True)

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(eval_one, t): t for t in tasks}

    for future in as_completed(futures):
        done += 1
        result = future.result()
        if result is None:
            continue
        if 'error' in result:
            errors += 1
        results.append(result)

        if done % 50 == 0:
            print(f'  [{done}/{total}] done, {errors} errors, {len(results)} results', flush=True)
            # Save incrementally
            with open('results/responses.json', 'w') as f:
                json.dump(results, f, ensure_ascii=False)

# Final save
with open('results/responses.json', 'w') as f:
    json.dump(results, f, ensure_ascii=False)

print(f'\nDONE: {len(results)} results, {errors} errors', flush=True)
