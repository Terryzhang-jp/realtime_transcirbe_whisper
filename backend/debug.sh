#!/bin/bash
export DEBUG=1
export LOG_LEVEL=DEBUG
export PYTHONPATH=$(pwd)

# 创建日志目录
mkdir -p logs

# 直接运行应用
python -m app.main
