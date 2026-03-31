# 功能：演示 LangChain invoke/stream/batch 三种调用方式

import os
import time
import sys
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

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

# ============================================
# 工具函数
# ============================================
def print_separator(title: str, width: int = 50) -> None:
  """打印带标题的分隔线，让输出更易读"""
  print(f"\n{'=' * width}")
  print(f"  {title}")
  print(f"{'=' * width}")
  
def print_stats(label: str, elapsed: float, char_count: int = 0) -> None:
    """打印性能统计信息"""
    if char_count > 0:
        speed = char_count / elapsed if elapsed > 0 else 0
        print(f"\n📊 {label}: 耗时 {elapsed:.2f}s | 输出 {char_count} 字 | 速度 {speed:.1f} 字/秒")
    else:
        print(f"\n📊 {label}: 耗时 {elapsed:.2f}s")
        
# ============================================
# 构建核心 Chain
# ============================================

def build_writing_chain(api_key: str, base_url: str, model_name: str = "qwen-plus-2025-07-28") -> StrOutputParser:
  """
  构建写作助手 Chain：Prompt → ChatModel → OutputParser
  
  这个 Chain 本身也是 Runnable，支持 invoke/stream/batch
  """
  
  prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位{style}风格的作家，擅长创作简洁有力的短文。所有回答控制在 80 字以内。"),
    ("human", "请以「{topic}」为主题，创作一段文字。")
  ])
  
  model = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
    temperature=0.8,
    max_tokens=200,
  )
  
  output_parser = StrOutputParser()
  
  chain = prompt | model | output_parser
  
  return chain

# ============================================
# 模式一：invoke —— 同步调用
# ============================================

def demo_invoke(chain, api_key, base_url) -> None:
  """演示 invoke：同步等待完整响应"""
  print_separator("模式一：invoke（同步调用）")
  print("💡 特点：等待全部生成完成后一次性返回，适合后台处理")
  print("⏳ 正在等待模型响应...\n")
  
  start = time.time()
  
  result = chain.invoke({
    "style": "海明威",
    "topic": "第一次看见大海"
  })
  
  elapsed = time.time() - start
  print("📝 生成结果：")
  print(result)
  print_stats("invoke 耗时", elapsed, len(result))
  
  # 对比：直接用 ChatModel（不经过 chain），展示 AIMessage 结构
  print("\n--- 附：直接调用 ChatModel（不经过 OutputParser）---")
  model = ChatOpenAI(model="qwen-plus-2025-07-28",api_key=api_key,base_url=base_url, temperature=0)
  raw_response = model.invoke([
      SystemMessage(content="你是简洁的助手"),
      HumanMessage(content="一句话：今天天气怎么样？")
  ])
  # 展示 AIMessage 的完整结构
  print(f"类型: {type(raw_response).__name__}")
  print(f"内容: {raw_response.content}")
  print(f"Token 用量: 输入={raw_response.usage_metadata['input_tokens']}, "
        f"输出={raw_response.usage_metadata['output_tokens']}")
  
# ============================================
# 模式二：stream —— 流式输出（打字机效果）
# ============================================

def demo_stream(chain) -> None:
  
  """演示 stream：逐 token 输出，实现打字机效果"""
  print_separator("模式二：stream（流式输出）")
  print("💡 特点：生成一个字就显示一个字，用户体验好，适合交互式场景")
  print("⌨️  开始流式生成：\n")
  
  start = time.time()
  char_count = 0
  
  try:
    for chunk in chain.stream({
      "style": "村上春树",
      "topic": "一个人喝咖啡的下午"
    }):
      print(chunk, end="", flush=True)
      char_count += len(chunk)

  except KeyboardInterrupt:
    print("\n\n⚠️  用户中断了流式输出")
  except Exception as e:
    print(f"\n\n❌ 流式输出出错: {e}")
    
  elapsed = time.time() - start
  print()
  print_stats("stream 耗时", elapsed, char_count)
  
 
# ============================================
# 模式三：batch —— 批量并发调用
# ============================================
def demo_batch(chain) -> None:
  """演示 batch：并发处理多个请求，对比串行速度"""
  print_separator("模式三：batch（批量并发）")
  print("💡 特点：并发处理多个请求，适合批量数据处理任务\n")
  
  writing_tasks = [
    {"style": "余华", "topic": "下雨天"},
    {"style": "张爱玲", "topic": "城市的夜晚"},
    {"style": "老舍", "topic": "胡同里的猫"},
    {"style": "钱钟书", "topic": "会议"},
    {"style": "汪曾祺", "topic": "早餐"},
  ]
  
  # --- 方式一：串行调用（for 循环）---
  print("🐢 串行调用（for 循环 + invoke）：")
  serial_start = time.time()
  serial_results = []
  for i, task in enumerate(writing_tasks):
    result = chain.invoke(task)
    serial_results.append(result)
    print(f"  ✓ 任务 {i+1} 完成", end="\r")
  
  serial_elapsed = time.time() - serial_start
  print(f"  ✓ 全部完成（{len(writing_tasks)} 个任务）")
  print_stats("串行总耗时", serial_elapsed)

  # --- 方式二：batch 并发调用 ---
  print(f"\n🚀 batch 并发调用（max_concurrency=5）：")
  batch_start = time.time()
  
  batch_results = chain.batch(
    writing_tasks,
    config = {
      "max_concurrency": 5
    }
  )
  
  batch_elapsed = time.time() - batch_start
  print(f"  ✓ 全部完成（{len(writing_tasks)} 个任务）")
  print_stats("batch 总耗时", batch_elapsed)
  
  # 展示速度提升
  speedup = serial_elapsed / batch_elapsed if batch_elapsed > 0 else 1
  print(f"⚡ 速度提升: {speedup:.1f}x")

  # 展示结果（batch 的输出顺序与输入顺序一致）
  print("\n📚 批量生成结果：")
  styles = [task["style"] for task in writing_tasks]
  for i, (style, result) in enumerate(zip(styles, batch_results)):
      print(f"\n[{style}风格 - {writing_tasks[i]['topic']}]")
      # 只显示前 50 字避免输出太长
      preview = result[:50] + "..." if len(result) > 50 else result
      print(f"  {preview}")
    
def main():
  print("🤖 LangChain 调用模式演示：invoke / stream / batch")
  print("📦 使用模型：qwen-plus-2025-07-28（ChatModels 接口）")
  
  api_key, base_url = load_config()
  
  # 构建共用的写作助手 Chain
  chain = build_writing_chain(api_key, base_url)
  
  # 依次演示三种调用模式
  demo_invoke(chain, api_key, base_url)
  
  input("\n按 Enter 继续演示 stream 模式...")
  demo_stream(chain)
  
  input("\n按 Enter 继续演示 batch 模式...")
  demo_batch(chain)

if __name__ == "__main__":
  main()