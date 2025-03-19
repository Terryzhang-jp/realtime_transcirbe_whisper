from fastapi import APIRouter, HTTPException
import numpy as np
import logging
from app.audio.audio_processor import AudioProcessor
import os
import time
import traceback
from pydub import AudioSegment
from app.services.transcription import TranscriptionService
import asyncio
import json
from fastapi.responses import HTMLResponse
import datetime

router = APIRouter(prefix="/test", tags=["测试"])

logger = logging.getLogger("test_api")

@router.get("/transcribe")
async def test_transcribe():
    """测试转写功能，使用模拟音频数据"""
    logger.info("开始转写测试")
    
    # 创建一个临时的音频处理器
    try:
        processor = AudioProcessor(
            client_id="test_client",
            debug_mode=True
        )
        
        # 设置一个简单的回调函数来记录结果
        results = []
        
        def callback(client_id, text):
            logger.info(f"转写结果: {text}")
            results.append(text)
        
        processor.set_transcription_callback(callback)
        
        # 生成一个简单的测试音频数据 (正弦波)
        sample_rate = 16000
        duration_seconds = 3
        frequency = 440  # 440 Hz, A4音符
        
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
        audio_data = np.sin(2 * np.pi * frequency * t) * 32767 / 2  # 半音量的正弦波
        audio_data = audio_data.astype(np.int16)
        
        logger.info(f"生成了 {len(audio_data)/sample_rate:.2f} 秒的测试音频数据")
        
        # 直接使用音频处理器的转写功能
        try:
            logger.info("开始直接转写测试音频")
            audio_float32 = audio_data.astype(np.float32) / 32768.0
            
            segments, info = processor.model.transcribe(
                audio_float32,
                language=processor.language,
                beam_size=5
            )
            
            segments_list = list(segments)
            transcription = " ".join([segment.text for segment in segments_list]).strip()
            
            if transcription:
                logger.info(f"直接调用模型转写结果: {transcription}")
            else:
                logger.warning("直接调用模型转写结果为空")
                
            logger.info("测试转写完成")
            
            return {
                "success": True,
                "message": "测试转写完成",
                "transcription": transcription,
                "audio_info": {
                    "duration": float(duration_seconds),
                    "sample_rate": int(sample_rate),
                    "max_amplitude": float(np.max(np.abs(audio_data))),
                    "samples": int(len(audio_data))
                }
            }
            
        except Exception as e:
            logger.error(f"直接转写测试时出错: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"转写测试失败: {str(e)}")
        
    except Exception as e:
        logger.error(f"测试初始化失败: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"测试初始化失败: {str(e)}")


@router.get("/transcribe-file")
async def test_transcribe_file(file_path: str = None):
    """测试从文件转写功能"""
    if not file_path:
        # 使用示例音频文件
        file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tests/test_voice.wav")
        if not os.path.exists(file_path):
            file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tests/test-87834.mp3")
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"默认测试音频文件不存在")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"指定的音频文件不存在: {file_path}")
    
    logger.info(f"开始转写文件测试: {file_path}")
    
    try:
        # 加载音频文件
        try:
            audio = AudioSegment.from_file(file_path)
            # 转换为16kHz, 16位, 单声道
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            
            # 转换为numpy数组
            audio_data = np.array(audio.get_array_of_samples(), dtype=np.int16)
            max_amplitude = float(np.max(np.abs(audio_data)))
            logger.info(f"加载了 {len(audio_data)/16000:.2f} 秒的音频数据，最大振幅: {max_amplitude}")
        except Exception as e:
            logger.error(f"加载音频文件失败: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"加载音频文件失败: {str(e)}")
        
        # 创建音频处理器
        processor = AudioProcessor(
            client_id="test_file_client",
            debug_mode=True
        )
        
        # 直接转写
        try:
            logger.info("开始转写音频文件")
            audio_float32 = audio_data.astype(np.float32) / 32768.0
            
            start_time = time.time()
            segments, info = processor.model.transcribe(
                audio_float32,
                language=processor.language,
                beam_size=5
            )
            elapsed_time = time.time() - start_time
            
            segments_list = list(segments)
            transcription = " ".join([segment.text for segment in segments_list]).strip()
            
            if transcription:
                logger.info(f"文件转写结果: {transcription}")
            else:
                logger.warning("文件转写结果为空")
                
            logger.info(f"文件转写完成，耗时: {elapsed_time:.2f}秒")
            
            return {
                "success": True,
                "message": "文件转写完成",
                "transcription": transcription,
                "elapsed_time": float(elapsed_time),
                "audio_info": {
                    "duration": float(len(audio_data) / 16000),
                    "sample_rate": 16000,
                    "max_amplitude": float(max_amplitude),
                    "samples": int(len(audio_data)),
                    "file_path": file_path
                }
            }
            
        except Exception as e:
            logger.error(f"文件转写失败: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"文件转写失败: {str(e)}")
        
    except Exception as e:
        logger.error(f"文件转写测试失败: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"文件转写测试失败: {str(e)}")


@router.get("/test-service")
async def test_transcription_service(client_id: str = "test_service_client", file_name: str = "test-87834.mp3"):
    """测试转写服务的回调功能"""
    logger.info(f"开始测试转写服务，客户端ID: {client_id}, 文件: {file_name}")
    
    try:
        # 创建转写服务
        service = TranscriptionService()
        
        # 定义回调函数来捕获转写结果
        results = []
        
        def transcription_callback(client_id: str, text: str):
            logger.info(f"收到转写结果: client_id={client_id}, text={text}")
            results.append(text)
        
        # 确保转写服务为该客户端初始化
        await service.register_client(
            client_id=client_id, 
            language="zh", 
            model_type="base",
            callback=transcription_callback,
            debug_mode=True
        )
        
        # 尝试加载并发送真实人声文件
        file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), f"tests/{file_name}")
        if os.path.exists(file_path):
            try:
                logger.info(f"加载测试音频文件: {file_path}")
                audio = AudioSegment.from_file(file_path)
                # 转换为16kHz, 16位, 单声道
                audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                
                # 转换为numpy数组
                audio_data = np.array(audio.get_array_of_samples(), dtype=np.int16)
                max_amplitude = float(np.max(np.abs(audio_data)))
                logger.info(f"加载了 {len(audio_data)/16000:.2f} 秒的音频数据，最大振幅: {max_amplitude}")
                
                # 将音频数据分成多个小块，模拟实时流式传输
                chunk_size = 16000  # 1秒的音频数据
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:min(i+chunk_size, len(audio_data))]
                    audio_bytes = chunk.tobytes()
                    
                    # 发送音频数据
                    logger.info(f"发送第 {i//chunk_size + 1} 块音频数据，大小: {len(audio_bytes)} 字节")
                    receive_result = await service.process_audio(client_id, audio_bytes)
                    logger.info(f"音频数据接收结果: {receive_result}")
                    
                    # 短暂等待，模拟实时流
                    await asyncio.sleep(0.2)
                
                # 等待音频处理完成
                logger.info(f"已发送所有音频数据，等待处理完成...")
                waiting_time = min(10, len(audio_data) / 16000 / 5)  # 等待时间与音频长度相关，但最多10秒
                logger.info(f"等待 {waiting_time:.2f} 秒让处理完成...")
                await asyncio.sleep(waiting_time)
            except Exception as e:
                logger.error(f"处理音频文件时出错: {e}")
                logger.error(traceback.format_exc())
        else:
            # 使用合成的音频数据
            logger.info(f"未找到测试音频文件: {file_path}，使用合成音频")
            
            # 模拟接收一些音频数据
            sample_rate = 16000
            duration_seconds = 2
            
            # 创建一些测试音频数据 (正弦波)
            t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
            audio_data = np.sin(2 * np.pi * 440 * t) * 32767 / 2  # 440Hz正弦波
            audio_bytes = audio_data.astype(np.int16).tobytes()
            
            # 发送一些音频数据
            logger.info(f"发送 {len(audio_bytes)} 字节的测试音频数据")
            receive_result = await service.process_audio(client_id, audio_bytes)
            logger.info(f"音频数据接收结果: {receive_result}")
            
            # 等待一下让音频处理完成
            logger.info("等待3秒让音频处理完成...")
            await asyncio.sleep(3)
        
        # 检查处理器状态
        processor = service.clients.get(client_id)
        if processor:
            logger.info(f"处理器状态: 运行中={processor.running}")
        else:
            logger.warning(f"未找到客户端 {client_id} 的处理器")
        
        # 清理资源
        await service.unregister_client(client_id)
        
        return {
            "success": True,
            "message": "测试完成",
            "client_id": client_id,
            "file_name": file_name,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"测试转写服务失败: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"测试转写服务失败: {str(e)}")


@router.get("/direct-transcribe")
async def test_direct_transcribe():
    """测试直接使用音频处理器进行转写的功能"""
    from app.audio.audio_processor import AudioProcessor
    import time
    
    # 创建一个简单的回调函数用于记录结果
    results = []
    
    def sync_callback(client_id, text):
        results.append(text)
        logger.info(f"回调收到转写结果: {text}")
    
    # 初始化音频处理器
    processor = AudioProcessor(
        client_id="direct_test",
        model_type="tiny",
        language="zh",
        debug_mode=True
    )
    
    # 设置回调函数
    processor.set_transcription_callback(sync_callback)
    
    # 启动处理
    processor.start_processing()
    
    try:
        # 打开测试音频文件
        import soundfile as sf
        import numpy as np
        import os
        
        # 先尝试在tests目录下找测试文件
        test_file = "/Users/yichuanzhang/Desktop/real_time_transcribe/backend/tests/test-87834.mp3"
        if not os.path.exists(test_file):
            # 如果找不到，使用其他测试文件
            test_file = "/Users/yichuanzhang/Desktop/real_time_transcribe/backend/tests/long_voice_test.mp3"
        
        logger.info(f"使用测试文件: {test_file}")
        
        # 读取音频文件
        audio_data, sample_rate = sf.read(test_file, dtype='float32')
        
        # 如果是立体声，转换为单声道
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = audio_data.mean(axis=1)
        
        # 确保采样率为16000Hz
        if sample_rate != 16000:
            logger.warning(f"文件采样率 {sample_rate}Hz 不是预期的 16000Hz，需要重采样")
            # TODO: 添加重采样逻辑
        
        # 转换为16位整数
        audio_int16 = (audio_data * 32768.0).astype(np.int16)
        
        # 将音频分成多个1024样本的块
        chunk_size = 1024
        total_chunks = len(audio_int16) // chunk_size
        
        # 添加一些块到处理队列
        logger.info(f"开始发送 {total_chunks} 个音频块...")
        
        # 每次发送一个块并等待一小段时间模拟实时性
        for i in range(0, total_chunks, 10):  # 每10个块发送一次，加快处理
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size * 10, len(audio_int16))
            chunk = audio_int16[start_idx:end_idx].tobytes()
            
            # 发送块到处理器
            processor.add_audio_chunk(chunk)
            
            # 等待一小段时间
            time.sleep(0.1)
        
        # 等待处理完成
        logger.info("所有音频块已发送，等待处理完成...")
        time.sleep(3)  # 给足够的时间让处理器处理所有数据
        
        # 返回结果
        return {
            "success": True,
            "message": "直接转写测试完成",
            "results": results
        }
    except Exception as e:
        logger.error(f"测试过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "message": f"测试失败: {str(e)}"
        }
    finally:
        # 清理资源
        processor.stop_processing()
        processor.cleanup()


@router.get("/ws-status")
async def websocket_status():
    """显示当前WebSocket客户端状态和音频统计信息"""
    from app.api.websocket import active_connections, audio_stats, transcription_service
    
    # 收集状态信息
    results = {
        "active_connections": [],
        "audio_stats": {},
        "transcription_clients": list(transcription_service.clients.keys()),
    }
    
    # 收集活跃连接信息，确保完全避免序列化问题
    for client_id, websocket in active_connections.items():
        connection_info = {
            "client_id": client_id,
            "state": "unknown",
            "connected": False
        }
        
        try:
            # 仅获取需要的字符串信息，避免序列化整个对象
            connection_info["state"] = str(websocket.client_state)
            
            # 安全地检查连接状态
            try:
                is_connected = bool(websocket.client_state.CONNECTED)
                connection_info["connected"] = is_connected
            except Exception as connect_err:
                connection_info["connected"] = False
                connection_info["state_error"] = str(connect_err)
        except Exception as e:
            connection_info["state"] = f"错误: {str(e)}"
            connection_info["connected"] = False
            
        # 将安全处理后的信息添加到结果中
        results["active_connections"].append(connection_info)
    
    # 收集音频统计信息
    for client_id, stats in audio_stats.items():
        clean_stats = {}
        for key, value in stats.items():
            if key in ["first_chunk_time", "last_chunk_time"] and value is not None:
                clean_stats[key] = str(datetime.datetime.fromtimestamp(value))
            else:
                # 确保所有值都是可序列化的
                try:
                    json.dumps({key: value})  # 测试是否可序列化
                    clean_stats[key] = value
                except (TypeError, OverflowError):
                    # 如果不能序列化，转换为字符串
                    clean_stats[key] = str(value)
        
        # 计算一些额外的统计数据
        if stats.get("first_chunk_time") is not None and stats.get("last_chunk_time") is not None:
            try:
                duration = stats["last_chunk_time"] - stats["first_chunk_time"]
                clean_stats["duration"] = f"{duration:.2f} 秒"
                
                if stats.get("total_chunks", 0) > 0:
                    clean_stats["chunks_per_second"] = f"{stats['total_chunks'] / duration:.2f}"
                    clean_stats["bytes_per_second"] = f"{stats['total_bytes'] / duration:.2f}"
            except Exception as e:
                clean_stats["duration_error"] = str(e)
        
        results["audio_stats"][client_id] = clean_stats
        
    # 添加转写客户端状态详情
    results["transcription_client_details"] = {}
    for client_id, processor in transcription_service.clients.items():
        client_info = {
            "running": getattr(processor, "running", False)
        }
        
        # 安全地获取其他属性
        for attr in ["language", "model_type", "debug_mode"]:
            if hasattr(processor, attr):
                client_info[attr] = getattr(processor, attr)
            else:
                client_info[attr] = "未知"
        
        # 添加一些安全的额外信息
        try:
            if hasattr(processor, "vad"):
                client_info["vad_enabled"] = processor.vad is not None
            if hasattr(processor, "audio_buffer"):
                client_info["buffer_size"] = len(processor.audio_buffer) if processor.audio_buffer else 0
        except Exception as e:
            client_info["info_error"] = str(e)
            
        results["transcription_client_details"][client_id] = client_info
    
    # 生成HTML前验证所有数据是否可序列化
    try:
        # 先测试一下是否所有数据都可以序列化
        connections_json = json.dumps(results["active_connections"], indent=2, ensure_ascii=False)
        stats_json = json.dumps(results["audio_stats"], indent=2, ensure_ascii=False)
        clients_json = json.dumps(results["transcription_clients"], indent=2, ensure_ascii=False)
        client_details_json = json.dumps(results["transcription_client_details"], indent=2, ensure_ascii=False)
    except Exception as json_err:
        # 如果出现序列化错误，返回错误信息
        return HTMLResponse(content=f"""
        <html>
            <head><title>序列化错误</title></head>
            <body>
                <h1>序列化错误</h1>
                <p>无法序列化WebSocket状态数据: {str(json_err)}</p>
                <button onclick="location.reload()">重试</button>
            </body>
        </html>
        """)
    
    # 返回结果，设置为HTML格式
    html_content = f"""
    <html>
        <head>
            <title>WebSocket状态</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                pre {{ background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                .container {{ margin-bottom: 20px; }}
                .refresh-btn {{ padding: 10px 15px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }}
                .refresh-btn:hover {{ background: #45a049; }}
            </style>
        </head>
        <body>
            <h1>WebSocket状态面板</h1>
            <div class="container">
                <h2>活跃WebSocket连接</h2>
                <pre>{connections_json}</pre>
            </div>
            <div class="container">
                <h2>音频处理统计</h2>
                <pre>{stats_json}</pre>
            </div>
            <div class="container">
                <h2>转写客户端列表</h2>
                <pre>{clients_json}</pre>
            </div>
            <div class="container">
                <h2>转写客户端详情</h2>
                <pre>{client_details_json}</pre>
            </div>
            <button class="refresh-btn" onclick="location.reload()">刷新</button>
        </body>
    </html>
    """
    
    return HTMLResponse(content=html_content) 