from datasets import load_dataset
from collections import defaultdict
import pandas as pd, random, json, ast

SEED = 448
TARGET_MODELS = {
    "GPT": "gpt-4o-2024-05-13",
    "Claude": "claude-3-5-sonnet-20240620",
    "Gemini": "gemini-1.5-pro-api-0514",
}
TARGET_PER_MODEL_DOMAIN = 100

def normalize_obj(x):
    if isinstance(x, dict): return x
    if isinstance(x, str):
        for fn in (json.loads, ast.literal_eval):
            try:
                y = fn(x)
                if isinstance(y, dict): return y
            except Exception: pass
    return {}

def is_math(row):
    tag = normalize_obj(row.get("category_tag"))
    math_tag = tag.get("math_v0.1", {})
    return isinstance(math_tag, dict) and math_tag.get("math") is True

def get_domain(row):
    if row.get("is_code") is True: return "code"
    if is_math(row): return "math"
    return "general"

def extract_prompt_response(conversation):
    if isinstance(conversation, str):
        for fn in (json.loads, ast.literal_eval):
            try:
                conversation = fn(conversation); break
            except Exception: pass
    if not isinstance(conversation, list): return None, None
    prompt = response = None
    for msg in conversation:
        if not isinstance(msg, dict): continue
        role, content = str(msg.get("role","")).lower(), msg.get("content")
        if not isinstance(content, str): continue
        if role == "user" and prompt is None: prompt = content.strip()
        elif role == "assistant" and prompt is not None and response is None:
            response = content.strip(); break
    return prompt, response

def main():
    random.seed(SEED)
    ds = load_dataset("lmarena-ai/arena-human-preference-100k", split="train", streaming=True)
    buckets = defaultdict(list)
    wanted = {(f,d) for f in TARGET_MODELS for d in ["general","code","math"]}
    for row in ds:
        if str(row.get("language","")).lower() != "english": continue
        if row.get("is_refusal") is True: continue
        if int(row.get("turn",1) or 1) != 1: continue
        domain = get_domain(row)
        for side in ("a","b"):
            model = row.get(f"model_{side}")
            family = next((f for f,m in TARGET_MODELS.items() if model == m), None)
            if family is None or len(buckets[(family,domain)]) >= 400: continue
            prompt, response = extract_prompt_response(row.get(f"conversation_{side}"))
            if not prompt or not response or len(prompt)<5 or len(response)<20: continue
            buckets[(family,domain)].append({
                "question_id":row.get("question_id"), "domain":domain,
                "llm_family":family, "model_name":model,
                "llm_input":prompt, "llm_output":response})
        if all(len(buckets[k]) >= 400 for k in wanted): break

    selected=[]
    for family in TARGET_MODELS:
        for domain in ["general","code","math"]:
            pool=buckets[(family,domain)]
            if len(pool) < TARGET_PER_MODEL_DOMAIN:
                raise RuntimeError(f"Not enough examples for {(family,domain)}")
            random.Random(SEED + sum(map(ord,family+domain))).shuffle(pool)
            selected.extend(pool[:TARGET_PER_MODEL_DOMAIN])
    df=pd.DataFrame(selected).drop_duplicates(["model_name","llm_input","llm_output"])
    df.to_csv("cmpsc448_llm_fingerprint_dataset.csv", index=False)
    print(df.groupby(["llm_family","domain"]).size())

if __name__ == "__main__":
    main()
