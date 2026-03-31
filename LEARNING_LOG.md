## [2026-03-28]

**今日主题**：搭建 LangChain 对话学习样例与基础 Python 项目配置

**完成内容**：
- 新增基于 DashScope 兼容接口的 LangChain 命令行对话样例，支持环境变量加载、模型初始化和流式回复
- 补充 Python 项目依赖与忽略规则，建立 `pyproject.toml`、`.gitignore` 和 `.env.example` 的基础开发配置
- 同步创建学习日志条目，沉淀本次 Agent 学习项目初始化的目标与产出

## [2026-03-31]

**今日主题**：实现 LangChain 三种调用模式的统一演示脚本

**完成内容**：
- 新增 `streaming_demo.py`，在同一示例中串联 `invoke`、`stream` 与 `batch` 三种调用方式
- 封装写作助手 Chain，组合 Prompt、`ChatOpenAI` 与 `StrOutputParser` 并复用于多种执行模式
- 增加批量并发与串行耗时对比、流式输出与基础错误处理，强化调用模式差异的可观测性
