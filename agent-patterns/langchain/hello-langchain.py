import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 加载 .env 文件中的环境变量

def load_config():
  load_dotenv()

  api_key = os.getenv("DASHSCOPE_API_KEY")
  base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

  if not api_key:
    print("❌ 错误：未找到 DASHSCOPE_API_KEY")
    sys.exit(1)  # 退出程序，返回错误状态码 1

  print(f"✅ API Key 已加载")
  print(f"✅ API Base URL：{base_url}")

  return api_key, base_url
  
# 创建 LLM 对象

def create_llm(api_key: str, base_url: str) -> ChatOpenAI:
  
  if "dashscope" in base_url:
    model_name = "qwen-plus-2025-07-28"
  else:
    print("请检查是否正确配置 DASHSCOPE_BASE_URL")
    sys.exit(1)
  
  llm = ChatOpenAI(
      model=model_name,
      api_key=api_key,
      base_url=base_url,
      temperature=0.7,
      streaming=True,
  )

  print(f"✅ LLM 初始化完成（模型：{model_name}）\n")
  return llm


# 构建消息列表
SYSTEM_PROMPT = """你是一位帮助前端工程师转型 AI Agent 工程师的学习伙伴。

你的回答风格：
- 对于概念解释，总是先用前端开发者熟悉的类比（React、TypeScript、Node.js 等）来建立直觉
- 然后再说清楚技术本质
- 给出可以直接运行的 Python 代码示例（而不是伪代码）
- 最后指出学习这个知识点的常见踩坑

回答要简洁而有深度，不要废话。用中文回答。"""



# 主对话循环
def run_chat(llm: ChatOpenAI):
  print("=" * 50)
  print("🤖 AI 学习伙伴已就绪！")
  print("输入你的问题，按 Enter 发送，输入 'exit' 退出")
  print("=" * 50)
  
  # 用于记录本次会话总 token 消耗（了解成本）
  total_tokens = 0  
    
  while True:
    try:
      user_input = input("\nUSER：").strip()
    except KeyboardInterrupt:
      # 处理 Ctrl+C，优雅退出而不是崩溃报错
      print("\n\n👋 已退出，再见！")
      break
    
    # 跳过空输入，继续等待
    if not user_input:
        continue
      
    # 检查退出命令
    if user_input.lower() in ["exit", "quit", "退出"]:
        print(f"\n👋 再见！本次会话共消耗 {total_tokens} tokens")
        break
      
    messages = [
      SystemMessage(content=SYSTEM_PROMPT),
      HumanMessage(content=user_input)
    ]
    
    # 调用 LLM，使用流式输出实时显示内容
    print("\nAI：", end="", flush=True)
    
    # 用于收集完整响应（计算 token 需要）
    full_response = ""
    
    try:
      for chunk in llm.stream(messages):
        content = chunk.content
        if content:
          print(content, end="", flush=True)
          full_response += content
          
    except Exception as e:
      print(f"\n❌ 调用出错：{e}")
      print("请检查网络连接和 API Key 是否有效")
      continue
    
    estimated_tokens = len(full_response) // 2 # 粗略估算：中文约 2 字符 = 1 token
    total_tokens += estimated_tokens
    print(f"\n\n[约 {estimated_tokens} tokens | 累计 {total_tokens} tokens]")
    


if __name__ == "__main__":
  api_key, base_url = load_config()
  llm = create_llm(api_key, base_url)
  run_chat(llm)