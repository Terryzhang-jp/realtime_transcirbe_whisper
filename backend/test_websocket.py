import asyncio
import websockets
import json
import uuid
import time
import logging
import sys
import os

# 设置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("WebSocketTest")

# 测试音频数据（模拟1秒44.1kHz的16位PCM数据）
# 这里使用空白噪音作为测试数据
TEST_AUDIO_DATA = bytes([0, 0] * 4000)  # 简化的测试数据

async def test_websocket():
    """测试WebSocket连接和音频处理"""
    client_id = str(uuid.uuid4())
    logger.info(f"测试开始，使用客户端ID: {client_id}")
    
    # 连接到WebSocket
    url = f"ws://localhost:8000/ws/transcribe/{client_id}"
    logger.info(f"正在连接到: {url}")
    
    try:
        async with websockets.connect(url) as websocket:
            logger.info("WebSocket连接已建立")
            
            # 接收连接成功消息
            response = await websocket.recv()
            response_data = json.loads(response)
            logger.info(f"收到连接响应: {response_data}")
            
            # 发送配置
            config = {
                "event": "config",
                "config": {
                    "language": "zh",
                    "model": "tiny"
                }
            }
            logger.info(f"发送配置: {config}")
            await websocket.send(json.dumps(config))
            
            # 接收配置更新响应
            response = await websocket.recv()
            response_data = json.loads(response)
            logger.info(f"收到配置更新响应: {response_data}")
            
            # 发送测试音频数据
            logger.info(f"开始发送音频数据...")
            for i in range(10):
                logger.info(f"发送音频数据块 {i+1}/10")
                await websocket.send(TEST_AUDIO_DATA)
                await asyncio.sleep(0.2)  # 每200ms发送一次
            
            # 等待可能的转写结果
            logger.info("等待转写结果（10秒）...")
            for i in range(5):
                try:
                    # 设置超时等待
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    response_data = json.loads(response)
                    logger.info(f"收到消息: {response_data}")
                    
                    # 检查是否是转写结果
                    if response_data.get("event") == "transcription":
                        logger.info(f"收到转写结果: {response_data.get('text')}")
                except asyncio.TimeoutError:
                    logger.warning(f"等待转写结果超时，继续等待...")
                    continue
            
            logger.info("测试完成")
            
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_websocket()) 