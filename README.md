# CGM 智能客服 Agent Demo

这是一个独立于主项目的 LangChain + LangGraph + Qdrant Demo，用来模拟动态血糖仪 CGM 客服 Agent。它刻意按生产系统可对照的方式分层：知识入库流水线、感知层、编排层、执行层、评估材料。

## 三条运行命令

本机当前推荐先进入 conda 环境：

```bash
conda activate si-l-z-sync
export PYTHONNOUSERSITE=1
```

```bash
cd customer_agent_demo
docker compose up -d qdrant
```

```bash
cd ..
python -m customer_agent_demo.ingest.run
```

```bash
python -m customer_agent_demo.agent.cli
```

Web UI 启动命令：

```bash
python -m customer_agent_demo.web --host 127.0.0.1 --port 7860
```

一键启动脚本：

```bash
bash customer_agent_demo/scripts/start_demo.sh
```

本机环境提示：作业目标是 Python 3.13 + Docker Qdrant。当前 WSL 里如果只有 Python 3.12 或没有 Docker，需要先安装 Python 3.13，并在 Docker Desktop 中开启 WSL integration。

## 安装依赖

```bash
python -m pip install -r customer_agent_demo/requirements-demo.txt
cp customer_agent_demo/.env.example customer_agent_demo/.env
```

在 `.env` 中填入 Qwen OpenAI-compatible 配置：

```bash
QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_API_KEY=你的 key
QWEN_LLM_MODEL=qwen3.5-plus
QWEN_EMBEDDING_MODEL=qwen3-vl-embedding
EMBEDDING_VECTOR_SIZE=1024
```

说明：当前 RAG 入库和查询统一使用 `qwen3-vl-embedding`，通过 DashScope 多模态 embedding 原生接口调用，并固定 `dimension=1024`。

## 代码地图

- `ingest/pipeline.py`：知识入库流水线。`load_sources -> clean_documents -> split_documents -> upsert_to_qdrant` 四段明确拆开。
- `agent/perception.py`：感知层。Qwen 配置完整时走 `with_structured_output(PerceptionResult)`；未配置时走本地启发式兜底，方便测试。
- `agent/graph.py`：LangGraph 编排层。入口是 `perceive`，再进入 swarm 执行层；执行层拆成产品咨询、售后流程、情绪安抚三个专职 Agent，并通过 `active_agent` 和 `Command(goto=...)` 交接控制权。
- `agent/rag.py`：产品咨询 Agent 使用的 C1 RAG 防线。链路为问题改写、检索、文档 grader、生成、幻觉自检；只基于 grader 通过的证据回答，末尾拼接引用。
- `prompts/`：Prompt 模板独立存放，方便版本对比和迭代。
- `data/hallucination_eval.md`：幻觉评估表，重点看参数化幻觉。
- `docs/graph.mmd`：由 `graph.get_graph().draw_mermaid()` 生成的图结构。

## 关键设计决定

入库重跑策略选择“删除并重建 collection”。理由是 Demo 需要强可复现，且 embedding 模型维度在 Qdrant collection 创建后固定；换模型时重建比增量兼容更直观。

切分策略由 `DEMO_CHUNKING_STRATEGY` 控制：

- `recursive`：原始基线，使用 `RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)`。
- `structural`：推荐默认值，复用主项目结构化切分，按标题、章节、目录和跨页续段保留语义边界。
- `parent-child`：embedding 使用 child chunk，回答上下文返回 parent context，适合短问题命中细粒度片段但回答需要完整段落的客服场景。

客服问题通常短，但答案需要保留一小段上下文；chunk 太小会丢条件和限制，太大会混入多个主题，增加幻觉风险。现场演示时可以先用 `structural` 入库，再切到 `parent-child` 对比召回上下文。

距离度量使用 cosine。Qwen embedding 表达的是语义方向，相比向量长度，方向相似度更适合 FAQ/说明书类语义召回。

默认 `top_k=4`。调小会降低噪声但容易漏掉关键限制条件；调大会提高召回覆盖，但把无关片段放进 context 后会增加生成幻觉和答案跑偏的概率。

默认检索策略是 `hybrid`。真实 Qdrant/Qwen embedding 可用时，dense 召回来自 Qdrant；本地 sparse 召回来自 `data/cgm_sources.json` 的轻量关键词索引。两路结果使用 alpha 融合：`AGENT_FUSION_ALPHA` 越大越偏向语义向量，越小越偏向关键词命中。客服 Demo 里保留 sparse 通道，是为了让防水等级、佩戴天数、IP28 等参数型问题更容易命中精确来源，同时也能演示“向量相似不等于事实相关”。

## C1 RAG 防线分层

产品咨询 Agent 的 RAG 链路拆成五个可观察节点：

1. `rewrite_question`：把“它/这个”等追问补全为当前产品问题。
2. `retrieve`：沿用 dense/hybrid 检索，并保留 candidate hits。
3. `grade_documents`：对每个候选文档输出 `binary_score=yes|no`、原因和失败类型。
4. `generate`：只使用 grader 通过的文档生成答案。
5. `hallucination_check`：检查答案是否有引用、数字是否被证据支撑。

Web UI 的 “C1 防线” 面板会展示每个节点状态，以及被 grader 拦下的候选文档。推荐演示问题是 `连接码是几位数？`：它可能召回包含 `14 天`、`IP28` 的高分参数片段，但 grader 会判定这不是连接码证据，失败类型为 `retrieval_mismatch`。

四种失败类型用于对应修法：

- `knowledge_missing`：知识库没有，需要补知识源后重新入库。
- `retrieval_mismatch`：召回不准，需要调 chunk、hybrid/rerank 或 query rewrite。
- `hallucination`：答案未被证据支撑，需要收紧 prompt、生成后校验并拒答。
- `format_unstable`：引用或结构不稳，需要结构化输出或后处理校验。

## Swarm 执行层

Demo 采用 swarm 模式而不是 supervisor 模式：没有中心主管 Agent 反复决定下一步，感知层只选择本轮入口，执行层 Agent 可以直接把控制权交给另一个 Agent。

- `product_consultant`：产品咨询和使用问题 Agent，挂载 RAG 检索与基于证据回答；同一对话里 RAG 连续两次证据不足时主动交给 `after_sales`。
- `after_sales`：售后流程 Agent，处理人工、退款、换货、订单、投诉等诉求，并生成坐席交接摘要。
- `empathy_agent`：情绪安抚 Agent，先响应强负面情绪；如果用户仍需要售后或人工，使用 handoff 交给 `after_sales`，如果只是产品/使用问题则回到 `product_consultant`。

这个实现对齐 `langgraph-swarm` 的三个核心概念：

- `create_swarm`：整体图记录 `active_agent`，多轮对话知道上一次由哪个 Agent 接手。
- `add_active_agent_router`：`perceive` 后的 `_active_agent_router` 按当前状态路由到活跃 Agent。
- `create_handoff_tool`：本 Demo 的 handoff 是确定性业务节点，不让模型调用工具；底层仍使用 LangGraph `Command(goto=...)` 更新 `active_agent` 并跳转到目标 Agent。

对照 `langgraph-supervisor create_supervisor`：supervisor 更适合“中心调度员统一分派任务”的层级结构；这里的客服场景更需要连续多轮由同一个专职 Agent 接住，并允许安抚、产品、售后之间直接交接，所以选择 swarm。

## 路由规则

- 愤怒用户先到 `empathy_agent`，安抚后如果仍需人工或售后则到 `after_sales`。
- 主动要求人工或售后诉求到 `after_sales`。
- 产品咨询和使用问题到 `product_consultant`，并由该 Agent 挂 RAG。
- 闲聊到 `smalltalk`。
- 同一对话里 RAG 连续两次检索不到足够依据时，由 `product_consultant` 主动转交 `after_sales`，生成坐席交接摘要。

## 转人工摘要

`after_sales` Agent 输出给坐席看的摘要：

- 用户最近问题
- 当前意图和情绪
- 已尝试回答
- 命中的知识来源
- 未解决原因
- 建议坐席下一步

## 感知稳定性实验

```bash
python -m customer_agent_demo.agent.perception_experiment
```

配置 Qwen 后，该脚本会把同一批输入分别用 `temperature=0` 和 `0.7` 跑 10 次，记录分类输出分布。未配置 Qwen 时会走本地启发式，输出稳定但不能代表真实模型波动。

## 幻觉评估

```bash
python -m customer_agent_demo.agent.evaluate_hallucination
```

评估 case 位于 `data/hallucination_eval_cases.json`，包含 10 个问题，其中至少 3 个是知识库外问题。脚本会检查是否拒答、是否带引用、是否命中期望来源，以及知识库外问题是否编造了高危数字。报告默认写入 `data/runs/`。

## 运行日志

默认开启 JSONL 运行日志：

```bash
AGENT_RUN_LOG_ENABLED=true
AGENT_RUN_LOG_DIR=customer_agent_demo/data/runs
```

每轮记录 `thread_id`、用户消息、感知结果、活跃 Agent、当前主题、回答状态、检索文档、debug trace、转人工原因和耗时。日志会过滤 API key、token、secret、password 等敏感字段。

## 从 Baklib 获取原始知识数据

当前入库流水线读取 `customer_agent_demo/data/cgm_sources.json`。如果要把授权 Baklib 站点内容作为客服知识源，先用登录后的 Cookie 抓取页面并生成同款 JSON，再运行 ingest。

```bash
export BAKLIB_COOKIE='从浏览器复制的 Cookie 请求头'
python3 -m customer_agent_demo.tools.baklib_crawler \
  --start-url https://sibionics.demo.baklib.vip/ \
  --output customer_agent_demo/data/cgm_sources.json \
  --max-pages 120
```

如果想先保留当前手工样例，改输出到临时文件检查：

```bash
python3 -m customer_agent_demo.tools.baklib_crawler \
  --output customer_agent_demo/data/baklib_sources.json \
  --max-pages 20
```

Cookie 获取方式：在浏览器登录 Baklib 后打开开发者工具 Network，刷新页面，点开主文档请求，在 Request Headers 中复制完整 `Cookie` 值。不要把 Cookie 写进 `.env` 或提交到 git；当前工具只从环境变量读取。

生成 JSON 后重建向量库：

```bash
cd customer_agent_demo
docker compose up -d qdrant
cd ..
python3 -m customer_agent_demo.ingest.run
```

## Demo 对话

见 `data/demo_conversations.md`。

推荐现场演示三段：

1. `Dexcom G7 可以戴着洗澡吗？`
2. `我的订单为什么还没发货？`
3. `你们这个传感器太差了，刚贴上就坏了，我要投诉，马上给我人工！`
