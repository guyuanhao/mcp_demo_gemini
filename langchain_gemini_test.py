import getpass
import os
import time
import sys
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

model = init_chat_model("gemini-2.5-pro", model_provider="google_genai")

messages = [
    SystemMessage(content="You are a story writer"),
    HumanMessage(content="Write me a song about goldfish on the moon, try more than 500 tokens"),
]

# res = model.invoke(messages)
# print(res.content)
# 流式方式：用户立即看到回答开始生成
for token in model.stream(messages):
    print(token.content, end="|", flush=True)
    time.sleep(0.2)  # 模拟逐字显示效果

# for i, token in enumerate(model.stream(messages)):
#     print(f"Token {i}: '{token.content}'", flush=True)
#     sys.stdout.flush()  # 强制刷新输出缓冲区
#     time.sleep(1)  # 模拟逐字显示效果