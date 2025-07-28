import os
import json
from dataclasses import dataclass
from typing import List, Dict
from loguru import logger
from openai import OpenAI
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy
import concurrent
from threading import Lock, Semaphore
from time import sleep
import uuid
# -------- Config --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# USER_MODEL = "gpt-4o-mini"
# ASSISTANT_MODELS = {
#     "gpt-4o-mini": ("gpt-4o-mini", False),
#     "gpt-4o-mini-cot": ("gpt-4o-mini", True),
#     "gpt-4o": ("gpt-4o", False),
#     "gpt-4o-cot": ("gpt-4o", True),
#     "gpt-4.1": ("gpt-4.1", False),
#     "gpt-4.1-cot": ("gpt-4.1", True),
#     "o1": ("o1", False),
#     "o1-cot": ("o1", True),
#     "o3": ("o3", False),
#     "o3-cot": ("o3", True),
# }

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
USER_MODEL = "qwen-turbo"
ASSISTANT_MODELS = {
    "deepseek-v3": ("deepseek-v3", False),
    "deepseek-v3-cot": ("deepseek-v3", True),
    "qwen3-235b-a22b": ("qwen3-235b-a22b", False),
    "qwen3-235b-a22b-cot": ("qwen3-235b-a22b", True),
    "qwen3-8b": ("qwen3-8b", False),
    "qwen3-8b-cot": ("qwen3-8b", True),
    "qwen-turbo": ("qwen-turbo", False),
    "qwen-turbo-cot": ("qwen-turbo", True),
    "qwen-plus": ("qwen-plus", False),
    "qwen-plus-cot": ("qwen-plus", True),
    "qwen-max": ("qwen-max", False),
    "qwen-max-cot": ("qwen-max", True),
    "deepseek-r1": ("deepseek-r1", False),
    "deepseek-r1-cot": ("deepseek-r1", True),
}
N_ROUNDS = 10
TEMPERATURE_USER = 0.7
TEMPERATURE_ASSISTANT = 0.7
MAX_RETRY = 10
PERSONA_PROMPT = (
    "You are an assistant who is highly Open, Conscientious, Extraverted, "
    "Agreeable, and Emotionally Stable."
)
COT_SUFFIX = (
    " Think step by step to ensure your response reflect this personality, "
    "but only output the final answer naturally; do not expose your chain-of-thought."
)
FORMATSUFFUX = '''
Output your reply in Json format. You need to output a json object wrapped in backticks in Markdown format. The format is as follows:
```json
{
    response:"your reply content"
}
```
'''

STARTERS = [
    "I'm planning a long backpacking trip—what kind of places should I prioritize if I love unexpected culture shocks?",
    "What’s your favorite way to spend a lazy Sunday afternoon?",
    "If I want to try new cuisines every week, how should I plan it for a year?",
    "Tell me about a time when traveling changed your perspective on life.",
    "What kind of music do you think best matches a road trip through the mountains?",
    "How do you keep friendships strong when everyone is super busy?",
    "If I want to be more spontaneous in life, where should I start?",
    "What’s a small habit that dramatically improved your well-being?",
    "How do you politely decline social invitations without hurting relationships?",
    "What would be an ideal city to live in if you like art, nature, and late-night conversations?"
]

SCENE = {
    "neutral": "I just had a regular day at work, nothing special happened.",
    "baseline": "",
    "positive": "I just finished my first marathon! I feel amazing.",
    "negative": "I just got laid off from my job. Feeling pretty lost right now.",

}

client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_ENDPOINT)
LOCK = Lock()
SYMAPHORE = Semaphore(16)
@dataclass
class Turn:
    role: str
    content: str

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def gen_user_dialogue_script(client: OpenAI, starter: str, n_rounds: int) -> List[str]:
    prompt = (
        "You are a user who is engaging in a casual conversation with an assistant.\n"
        "Generate exactly {n} user utterances, one per line, continuing this conversation.\n\n"
        "Initial user message:\n{starter}\n\n"
        "Format your output as follows:\n"
        "- The 1st user utterance you will generate.\n"
        "- The 2nd user utterance you will generate.\n"
        "- The 3rd user utterance you will generate.\n"
        "...\n"
        "- The {n}th user utterance you havwille generate.\n\n"
        "Do not include any additional text, index, number or explanations.\n"
        "Make sure each utterance is natural and flows from the previous one.\n"
        "Your output:\n"
    ).format(n=n_rounds, starter=starter)

    resp = client.chat.completions.create(
        model=USER_MODEL,
        # temperature=TEMPERATURE_USER,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for generating user messages."},
            {"role": "user", "content": prompt}
        ],
    )
    lines = [l.strip("- ").strip() for l in resp.choices[0].message.content.strip().split("\n") if l.strip()]
    while len(lines) < n_rounds:
        lines.append(lines[-1])
    return lines[:n_rounds]

def run_one_assistant(client: OpenAI, cot: bool, model: str,
                      persona_prompt: str, user_msgs: List[str]):
    sys_prompt = persona_prompt + (COT_SUFFIX if cot else "")
    history = [{"role": "system", "content": sys_prompt}]
    for user_msg in user_msgs:
        history.append({"role": "user", "content": user_msg})
        resp = client.chat.completions.create(
            model=model,
            # temperature=TEMPERATURE_ASSISTANT,
            messages=history,
            extra_body={"enable_thinking": False},
        )
        assistant_text = resp.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": assistant_text})
    return history
def save_incrementally(incremental_summary_path, summary_records):
    with LOCK:
        # Check if the file exists and has content, or create it if it doesn't
        if os.path.exists(incremental_summary_path):
            os.remove(incremental_summary_path)

        existing_data = summary_records  # If no existing file, use the current records
        with open(incremental_summary_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

def run_assistant_for_script(assistant_key, client, cot, model_name, persona_prompt, user_utterances, background, description, starter):
    config = {
        "assistant": assistant_key,
        "starter": starter,
        "background": background,
        "description": description,
    }
    with SYMAPHORE:
        for i in range(MAX_RETRY):
            try:
                outputs = run_one_assistant(client, cot, model_name, persona_prompt, user_utterances)
                break
            except Exception as e:
                logger.error(f"Config: {json.dumps(config, ensure_ascii=False, indent=2)} \nFail to run assistant script at attempt {i}: {e}")
                sleep(5)
                continue
        return {
            "uuid": uuid.uuid4(),
            "assistant": assistant_key,
            "starter": starter,
            "background": background,
            "description": description,
            "content": outputs
        }

def main():
    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"runs/{exp_id}"
    ensure_dir(out_dir)

    logger.info("Generating user dialogue scripts...")
    all_user_scripts = []
    for starter in tqdm(STARTERS, desc="User scripts"):
        all_user_scripts.append({
            "starter": starter,
            "user_utterances": gen_user_dialogue_script(client, starter, N_ROUNDS)
        })
    for s in all_user_scripts:
        logger.info(f"Starter: {s['starter']}\nUser Utterances: {s['user_utterances']}")

    logger.info("Running assistants...")
    summary_records = []
    final_summary_path = os.path.join(out_dir, "summary.json")

    for background, description in SCENE.items():
        logger.info(f"Running assistants in {background} scene with description: {description}")
        _all_user_scripts = deepcopy(all_user_scripts)
        for s in _all_user_scripts:
            s['user_utterances'][0] = description + " " + s['user_utterances'][0]

        # Initialize the record for this scene
        record = {
            "background": background,
            "description": description,
            "user_scripts": _all_user_scripts,
            "data": []
        }
        summary_records.append(record)

        # Process each assistant's replies
        for assistant_key, (model_name, cot) in tqdm(ASSISTANT_MODELS.items(), desc="Assistants"):
            # Use ThreadPoolExecutor to parallelize the innermost loop
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for s in tqdm(_all_user_scripts, desc=f"{assistant_key}", leave=False):
                    # Submit tasks to the thread pool
                    futures.append(executor.submit(run_assistant_for_script, assistant_key, client, cot, model_name, PERSONA_PROMPT, s["user_utterances"], background, description, s['user_utterances'][0]))

                # Collect all task results
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    logger.debug(f"Assistant {result['assistant']} - {result['background']} completed for starter: {result['starter']}")
                    # Append the response data to the record
                    summary_records[-1]["data"].append(result)
                    # Save the results incrementally for this scene
                    save_incrementally(final_summary_path, summary_records)

    logger.info(f"All done. Final summary saved to {final_summary_path}")

if __name__ == "__main__":
    main()