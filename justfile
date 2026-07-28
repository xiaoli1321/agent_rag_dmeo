# 开发模式运行
dev:
    concurrently --names "backend" --prefix-colors "yellow" \
        "PYTHONPATH=/home/admin/workspace python -m agent_rag_dmeo.web --host 127.0.0.1 --port 7860"

# 格式化代码
format:
    ruff format .
