from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import logging
import os
import uvicorn
from app.api import websocket
from app.audio.audio_input import AudioInput
from app import config
from app.api.test import router as test_router  # 导入新的测试路由

# 定义服务名称
SERVICE_NAME = "实时语音转写系统"

# 配置日志
logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=config.LOG_FILE,
    filemode="a"
)

logger = logging.getLogger(SERVICE_NAME)

# 创建FastAPI应用
app = FastAPI(
    title=SERVICE_NAME,
    description="实时语音转写服务",
    version="0.1.0",
    debug=config.DEBUG
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册WebSocket路由
app.include_router(websocket.router)
app.include_router(test_router)  # 注册测试路由

@app.get("/")
async def root():
    """健康检查端点"""
    return {"status": "ok", "service": SERVICE_NAME}

@app.get("/api/devices")
async def get_devices() -> List[Dict[str, Any]]:
    """获取可用的音频输入设备"""
    audio_input = AudioInput(debug_mode=config.DEBUG)
    devices = audio_input.list_devices()
    return devices

@app.get("/api/config")
async def get_config() -> Dict[str, Any]:
    """获取服务配置信息"""
    return {
        "available_models": config.AVAILABLE_MODELS,
        "available_languages": config.AVAILABLE_LANGUAGES,
        "default_model": config.DEFAULT_MODEL_TYPE,
        "default_language": config.DEFAULT_LANGUAGE
    }

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    await websocket.transcription_service.cleanup()
    logging.info("应用已关闭，资源已清理")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """日志中间件，记录所有请求"""
    logger.debug(f"请求: {request.method} {request.url}")
    response: Response = await call_next(request)
    logger.debug(f"响应: {response.status_code}")
    return response

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", 
        host=config.HOST, 
        port=config.PORT, 
        reload=config.DEBUG
    ) 