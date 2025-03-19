import os
from typing import Optional, Dict, Any, List

# 基本配置 - 开启调试模式
DEBUG = os.getenv("DEBUG", "1") == "1"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# CORS配置
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# 音频处理配置
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_SIZE = 1024

# 转写模型配置
DEFAULT_MODEL_TYPE = "tiny"
DEFAULT_LANGUAGE = "zh"
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]
AVAILABLE_LANGUAGES = ["zh", "en", "ja", "ko", "fr", "de", "ru"]

# 模型文件缓存目录
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")

# 日志配置 - 设置为DEBUG级别
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
LOG_FILE = os.getenv("LOG_FILE", "realtimestt.log")

# 音频处理配置 - 降低阈值使VAD更敏感
VAD_SILENCE_THRESHOLD = int(os.getenv("VAD_SILENCE_THRESHOLD", "5"))  # 进一步降低静音帧数阈值，原为10
MIN_SPEECH_FRAMES = int(os.getenv("MIN_SPEECH_FRAMES", "2"))  # 进一步降低最小语音帧数，原为3 