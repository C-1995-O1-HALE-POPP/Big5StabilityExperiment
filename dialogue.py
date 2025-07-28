import os
import json
from loguru import logger
from prompt import (
    STARTER_QUESTIONS, ASSISTANT_PROMPT, ASSISTANT_COT, USER_PROMPT
)
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
N_ROUNDS = 10

USER_MODEL = "gpt-4o-mini"
ASSISTANTS = {
    "gpt-4o-mini":      ["gpt-4o-mini", False],
    "gpt-4o-mini-cot":  ["gpt-4o-mini", True],
    # "gpt-4o":           ["gpt-4o", False],
    # "gpt-4o-cot":       ["gpt-4o", False]
}

proxies = {
    "http://": "http://127.0.0.1:10227",
    "https://": "http://127.0.0.1:01227"
}

client = OpenAI(api_key=OPENAI_API_KEY,proxies=proxies)

def simulate_dialogue(
        user_model: str, assistant_model: str, 
        rounds: int = N_ROUNDS, question: str = "",
        cot: bool = False):
    

    user_prefix = [{"role": "system", "content": USER_PROMPT}]
    assistant_prefix = [{"role": "system", "content": ASSISTANT_PROMPT + (ASSISTANT_COT if cot else "")}]

    # 第一轮用户消息
    dialogue = [{"role": "user", "content": question}]

    for i in range(rounds):
        # 生成助手回复
        full_context = assistant_prefix + dialogue
        resp_assistant = client.chat.completions.create(
            model=assistant_model,
            messages=full_context,
            temperature=0.7
        )
        assistant_reply = resp_assistant.choices[0].message.content.strip()
        dialogue.append({"role": "assistant", "content": assistant_reply})

        # 下一轮用户提问
        resp_user = client.chat.completions.create(
            model=user_model,
            messages=user_prefix + dialogue,
            temperature=0.7
        )
        user_msg = resp_user.choices[0].message.content.strip()
        dialogue.append({"role": "user", "content": user_msg})

    return dialogue

def main():
    os.makedirs("multi_agent_runs", exist_ok=True)
    result = []
    for key, config in ASSISTANTS.items():
        model, cot = config
        logger.info(f"Simulating dialogue with assistant={key}")
        dialogue = []
        for question in STARTER_QUESTIONS:
            dialogue.append({"question": question, "dialogue": simulate_dialogue(USER_MODEL, model, key, question, cot)})
        path = f"multi_agent_runs/dialogue_{key}.jsonl"
        result.append({"config": key, "data": dialogue})
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(dialogue, ensure_ascii=False))
    logger.info(f"Saved dialogue to {path}")

if __name__ == "__main__":
    main()
