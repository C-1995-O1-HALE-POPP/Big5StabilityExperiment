from openai import OpenAI
from httpx import Client, HTTPTransport
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

proxy_mounts = {
    "http://": HTTPTransport(proxy="http://127.0.0.1:10227"),
    "https://": HTTPTransport(proxy="http://127.0.0.1:10227"),
}

# 使用 httpx.Client(proxies=proxies)
http_client = Client(proxy="http://127.0.0.1:10227")

# 将 http_client 传入 OpenAI
client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)

try:
    models = client.models.list()
    print("OpenAI API OK, total models:", len(models.data))
except Exception as e:
    print("OpenAI API error:", e)
