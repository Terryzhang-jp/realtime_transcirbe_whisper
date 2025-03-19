import asyncio
import numpy as np
import os
import logging
import sys
from app.audio.audio_processor import AudioProcessor

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AudioProcessorTest")

# 创建简单的测试音频数据（1秒钟的440Hz正弦波）
def create_test_audio():
    sample_rate = 16000
    duration = 1.0  # 秒
    frequency = 440  # Hz
    
    # 生成时间数组
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # 生成正弦波
    audio = np.sin(2 * np.pi * frequency * t)
    
    # 转换为16位整数
    audio = (audio * 32767).astype(np.int16)
    
    # 转换为字节
    audio_bytes = audio.tobytes()
    
    logger.info(f"创建了测试音频: {len(audio_bytes)} 字节, {len(audio)/sample_rate:.2f}秒")
    return audio_bytes

# 回调函数
async def transcription_callback(text):
    logger.info("="*80)
    logger.info(f"【回调函数】收到转写结果: {text}")
    logger.info("="*80)

async def test_audio_processor():
    logger.info("开始测试AudioProcessor")
    
    # 创建AudioProcessor实例
    logger.info("创建AudioProcessor实例...")
    processor = AudioProcessor(
        language="zh",
        model_type="tiny",
        callback=transcription_callback,
        debug_mode=True
    )
    
    # 启动处理器
    logger.info("启动AudioProcessor...")
    await processor.start()
    
    # 创建测试音频
    audio_data = create_test_audio()
    
    # 发送音频数据
    logger.info("发送测试音频数据...")
    for i in range(5):
        logger.info(f"发送第 {i+1}/5 块音频数据")
        await processor.process_audio(audio_data)
        await asyncio.sleep(0.2)  # 每200ms发送一次
    
    # 确保有足够的时间进行转写
    logger.info("等待转写结果（10秒）...")
    await asyncio.sleep(10)
    
    # 停止处理器
    logger.info("停止AudioProcessor...")
    await processor.stop()
    
    logger.info("测试完成")

if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_audio_processor()) 