import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import BatchedInferencePipeline
import torch
import webrtcvad
import logging
import threading
import queue
import multiprocessing as mp
from typing import Optional, Callable, List, Dict, Any, Union
from app import config
import traceback
import time
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
import re

# 检查是否安装了pvporcupine
try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False

class AudioProcessor:
    def __init__(
        self,
        language: str = config.DEFAULT_LANGUAGE,
        model_type: str = config.DEFAULT_MODEL_TYPE,
        callback: Optional[Callable] = None,
        realtime_callback: Optional[Callable] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        compute_type: str = "float16" if torch.cuda.is_available() else "int8",
        debug_mode: bool = config.DEBUG,
        # 添加实时转录相关参数
        enable_realtime_transcription: bool = False,
        use_main_model_for_realtime: bool = True,
        realtime_model_type: str = "tiny",
        realtime_processing_pause: float = 0.2,
        # 添加文本稳定化相关参数
        stabilization_window: int = 2,
        match_threshold: int = 10,
        # 添加断句相关参数
        vad_speech_pad_ms: int = 300,  # 语音前后填充的毫秒数
        vad_min_speech_duration_ms: int = 250,  # 最小语音持续时间
        vad_max_speech_duration_ms: int = 30000,  # 最大语音持续时间
        sentence_punctuation_boost: bool = True,  # 加强句子标点处理
        initial_prompt: Optional[str] = None,  # 转写初始提示
    ):
        self.language = language
        self.model_type = model_type
        self.callback = callback
        self.realtime_callback = realtime_callback
        self.sample_rate = config.DEFAULT_SAMPLE_RATE  # 固定采样率
        self.vad_model = webrtcvad.Vad(3)  # 使用最高灵敏度
        self.debug_mode = debug_mode
        self.device = device  # 保存设备信息
        self.compute_type = compute_type  # 保存计算类型
        
        # VAD设置
        self.bypass_vad = False  # 启用VAD检测
        self.frame_duration = 30  # 每帧的持续时间（毫秒）
        self.vad_threshold = 0.8  # VAD检测阈值
        self.use_silero_vad = False  # 是否使用Silero VAD
        self.silero_vad_model = None  # Silero VAD模型
        self.porcupine = None  # 唤醒词检测模型
        
        # 音频缓冲相关设置
        self.audio_buffer = []
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.max_silence_frames = config.VAD_SILENCE_THRESHOLD  # 静音检测阈值
        self.min_speech_frames = config.MIN_SPEECH_FRAMES  # 最小语音帧数
        self.force_vad_trigger = False  # 强制触发VAD的标志
        self.last_transcription_time = 0  # 上次转写时间戳
        
        # 音频数据缓存
        self.cumulative_audio = np.array([], dtype=np.int16)
        
        # 运行状态
        self.running = False
        
        # 多进程设置
        self.use_multiprocessing = False  # 默认不使用多进程
        self.transcript_queue = None
        self.transcript_process = None
        
        # 配置日志
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(f"AudioProcessor-{language}-{model_type}")
        self.logger.setLevel(logging.DEBUG)
        
        # 日志配置信息
        self.logger.info(f"初始化AudioProcessor: language={language}, model_type={model_type}")
        self.logger.info(f"设备: {device}, 计算类型: {compute_type}")
        self.logger.info(f"VAD配置: {'绕过VAD' if self.bypass_vad else '使用VAD'}")
        
        # 初始化Whisper模型
        self.logger.info(f"正在初始化Whisper模型: {model_type}...")
        try:
            model_path = None
            if config.MODEL_CACHE_DIR:
                model_path = os.path.join(config.MODEL_CACHE_DIR, model_type)
                os.makedirs(config.MODEL_CACHE_DIR, exist_ok=True)
                
            self.model = WhisperModel(
                model_size_or_path=model_type,
                device=device,
                compute_type=compute_type,
                download_root=config.MODEL_CACHE_DIR
            )
            
            # 尝试启用批量推理
            try:
                self.logger.info("正在启用批量推理...")
                self.model = BatchedInferencePipeline(model=self.model)
                self.logger.info("批量推理启用成功")
            except Exception as e:
                self.logger.warning(f"批量推理启用失败: {e}, 使用标准推理")
                
            self.logger.info(f"Whisper模型初始化成功: {model_type}")
        except Exception as e:
            self.logger.error(f"初始化Whisper模型失败: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
        # 初始化Silero VAD
        self.logger.info("正在初始化Silero VAD...")
        try:
            self.silero_vad_model, _ = torch.hub.load(
                "snakers4/silero-vad", 
                "silero_vad", 
                force_reload=False
            )
            self.silero_vad_model.to(device)
            self.use_silero_vad = True
            self.logger.info("Silero VAD初始化成功")
        except Exception as e:
            self.logger.warning(f"Silero VAD初始化失败: {e}, 使用WebRTC VAD")
            self.use_silero_vad = False
            
        # 初始化唤醒词检测
        if PORCUPINE_AVAILABLE:
            self.logger.info("正在初始化唤醒词检测...")
            try:
                self.porcupine = pvporcupine.create(
                    keywords=["hey computer", "ok assistant"],
                    sensitivities=[0.7, 0.7]
                )
                self.logger.info("唤醒词检测初始化成功")
            except Exception as e:
                self.logger.warning(f"唤醒词检测初始化失败: {e}")
                self.porcupine = None
        else:
            self.logger.warning("未找到pvporcupine库，唤醒词检测未启用")
            
        # 初始化多进程处理
        if self.use_multiprocessing:
            self.logger.info("初始化多进程转写...")
            self.transcript_queue = mp.Queue()
            self.transcript_process = mp.Process(
                target=self._transcription_worker, 
                args=(self.transcript_queue,)
            )
            self.transcript_process.start()
            self.logger.info("多进程转写初始化成功")
        
        # 添加VAD协同工作相关属性
        self.webrtc_speech_active = False
        self.silero_speech_active = False
        self.silero_working = False
        self.silero_deactivity_detection = True
        self.vad_executor = ThreadPoolExecutor(max_workers=2)
        
        # 添加实时转录相关属性
        self.enable_realtime_transcription = enable_realtime_transcription
        self.use_main_model_for_realtime = use_main_model_for_realtime
        self.realtime_model_type = realtime_model_type
        self.realtime_processing_pause = realtime_processing_pause
        self.realtime_model = None

        # 添加文本稳定化相关属性
        self.text_storage = []
        self.stabilized_text = ""
        self.stabilization_window = stabilization_window
        self.match_threshold = match_threshold
        
        # 添加断句相关配置
        self.vad_speech_pad_ms = vad_speech_pad_ms
        self.vad_min_speech_duration_ms = vad_min_speech_duration_ms
        self.vad_max_speech_duration_ms = vad_max_speech_duration_ms
        self.sentence_punctuation_boost = sentence_punctuation_boost
        self.initial_prompt = initial_prompt if initial_prompt else self._get_default_prompt(language)
    
    async def start(self) -> bool:
        """异步启动处理器"""
        self.logger.info("启动AudioProcessor")
        self.running = True
        self.last_transcription_time = time.time()
        return True
    
    async def stop(self) -> bool:
        """异步停止处理器"""
        self.logger.info("停止AudioProcessor")
        # 确保进行最终转写
        if len(self.cumulative_audio) > 0 and self.speech_frames > self.min_speech_frames:
            self.logger.info("执行最终转写...")
            await self.transcribe_audio()
        
        # 停止多进程
        if self.use_multiprocessing and self.transcript_process and self.transcript_process.is_alive():
            self.logger.info("停止转写进程...")
            self.transcript_queue.put(None)  # 发送终止信号
            self.transcript_process.join(timeout=3)
            if self.transcript_process.is_alive():
                self.logger.warning("转写进程未正常终止，强制终止")
                self.transcript_process.terminate()
        
        self.running = False
        self.reset_buffer()
        return True
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.stop()
        if self.vad_executor:
            self.vad_executor.shutdown(wait=True)
    
    def __enter__(self):
        """同步上下文管理器入口"""
        import asyncio
        asyncio.run(self.start())
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """同步上下文管理器退出"""
        import asyncio
        asyncio.run(self.stop())
        if self.vad_executor:
            self.vad_executor.shutdown(wait=True)
    
    async def process_audio(self, audio_data: bytes) -> bool:
        """处理音频数据"""
        if not self.running:
            self.logger.warning("AudioProcessor未运行，忽略音频数据")
            return False
        
        try:
            self.logger.debug(f"接收到音频数据: {len(audio_data)} 字节")
            
            # 处理音频数据块
            is_speech, audio_samples = await self.process_audio_chunk(audio_data)
            
            # 如果数据无效，返回
            if len(audio_samples) == 0:
                self.logger.warning("音频数据无效，忽略")
                return False
            
            # 累积音频数据
            previous_length = len(self.cumulative_audio)
            if previous_length == 0:
                self.cumulative_audio = audio_samples
                self.logger.debug(f"开始累积音频: {len(audio_samples)} 样本")
            else:
                self.cumulative_audio = np.concatenate([self.cumulative_audio, audio_samples])
                self.logger.debug(f"累积音频: 从 {previous_length} 增加到 {len(self.cumulative_audio)} 样本")
            
            # 更新语音状态
            if is_speech:
                if not self.is_speaking:
                    self.logger.info("检测到语音开始")
                    self.is_speaking = True
                self.speech_frames += 1
                self.silence_frames = 0
            else:
                if self.is_speaking:
                self.silence_frames += 1
                    if self.silence_frames > self.max_silence_frames:
                        self.logger.info(f"检测到语音结束，静音持续了 {self.silence_frames} 帧")
                        self.is_speaking = False
                        # 如果累积了足够的语音，触发转写
                        if self.speech_frames > self.min_speech_frames:
                            self.force_vad_trigger = True
                            self.logger.info(f"语音结束，累积了 {self.speech_frames} 帧语音，触发转写")
            
            # 计算当前缓冲区长度和自上次转写的时间
            current_time = time.time()
            current_buffer_seconds = len(self.cumulative_audio) / self.sample_rate
            time_since_last_transcription = current_time - self.last_transcription_time
            
            # 决定是否需要转写
            should_transcribe = False
            transcription_reason = ""
            
            if self.bypass_vad:
                # 绕过VAD模式下，一旦累积了足够的音频且自上次转写已经过去了足够时间就转写
                if current_buffer_seconds >= 2.0 and time_since_last_transcription >= 1.0:
                    should_transcribe = True
                    transcription_reason = f"绕过VAD模式，累积了 {current_buffer_seconds:.1f}s 的音频，距离上次转写 {time_since_last_transcription:.1f}s"
            else:
                # 使用VAD模式，在下列情况触发转写：
                # 1. 语音结束且累积了足够的语音
                if self.force_vad_trigger:
                    should_transcribe = True
                    transcription_reason = f"VAD检测到语音结束，累积了 {self.speech_frames} 帧语音"
                    self.force_vad_trigger = False
                # 2. 连续语音时间过长
                elif self.is_speaking and self.speech_frames > 150:  # 约3秒连续语音
                    should_transcribe = True
                    transcription_reason = f"连续语音时间过长 ({self.speech_frames} 帧)，中间转写"
                # 3. 距离上次转写时间过长且有足够的语音
                elif self.speech_frames > self.min_speech_frames and time_since_last_transcription > 5.0:
                    should_transcribe = True
                    transcription_reason = f"距离上次转写时间过长 ({time_since_last_transcription:.1f}s)，触发转写"
            
            # 如果缓冲区过长，也触发转写
            if current_buffer_seconds > 10.0:
                should_transcribe = True
                transcription_reason = f"缓冲区过长 ({current_buffer_seconds:.1f}s)，强制转写"
            
            # 如果应该转写
            if should_transcribe:
                self.logger.info(f"开始转写: {transcription_reason}")
                await self.transcribe_audio()
                # 重置语音帧计数（但保留语音状态）
                self.speech_frames = 0
            else:
                self.logger.debug(f"继续积累音频: 当前 {current_buffer_seconds:.1f}s，上次转写距现在 {time_since_last_transcription:.1f}s")
            
            # 在累积音频后添加实时转录处理
            if self.enable_realtime_transcription and len(self.cumulative_audio) > 0:
                # 只有当积累了足够的音频数据时才进行实时转录
                if len(self.cumulative_audio) / self.sample_rate > 0.5:  # 至少0.5秒
                    await self.process_for_realtime(self.cumulative_audio)
                    await asyncio.sleep(self.realtime_processing_pause)  # 避免处理过于频繁
            
            return True
        
        except Exception as e:
            self.logger.error(f"处理音频数据时出错: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    async def transcribe_audio(self) -> None:
        """转写累积的音频数据"""
        if len(self.cumulative_audio) == 0:
            self.logger.warning("没有累积的音频数据，跳过转写")
            return
        
        # 确保音频数据足够长
        audio_duration = len(self.cumulative_audio) / self.sample_rate
        if audio_duration < 0.5:  # 至少0.5秒
            self.logger.warning(f"音频太短 ({audio_duration:.2f}s)，跳过转写")
            return
        
        self.logger.info(f"==========================================")
        self.logger.info(f"准备转写 {audio_duration:.2f}s 的音频")
        self.logger.info(f"音频样本数: {len(self.cumulative_audio)}")
        self.logger.info(f"采样率: {self.sample_rate}")
        self.logger.info(f"语言: {self.language}")
        self.logger.info(f"模型: {self.model_type}")
        
        try:
            # 使用多进程处理
            if self.use_multiprocessing and self.transcript_queue:
                self.logger.info("使用多进程进行转写...")
                # 复制音频数据以避免竞态条件
                audio_copy = np.copy(self.cumulative_audio)
                # 放入转写队列
                self.transcript_queue.put((audio_copy, self.language))
                # 重置音频缓冲区
                self.logger.info(f"重置音频缓冲区，清理 {len(self.cumulative_audio)} 个样本")
                self.cumulative_audio = np.array([], dtype=np.int16)
                self.last_transcription_time = time.time()
                self.reset_buffer()
                return
                
            # 转换为float32用于Whisper模型
            audio_float32 = self.cumulative_audio.astype(np.float32) / 32768.0
            
            # 记录音频特征
            max_amplitude = np.max(np.abs(audio_float32))
            mean_amplitude = np.mean(np.abs(audio_float32))
            self.logger.info(f"音频特征: 最大振幅={max_amplitude:.4f}, 平均振幅={mean_amplitude:.4f}")
            
            # 记录详细的转写开始日志
            self.logger.info(f"开始转写 {len(audio_float32)/self.sample_rate:.2f}s 的音频 | 语言: {self.language} | 最大音量: {max_amplitude:.4f}")
            
            # 记录转写开始时间
            transcribe_start_time = time.time()
            
            # 记录模型信息，使用self.device而不是model.device
            self.logger.info(f"使用Whisper模型: {self.model_type}, 设备: {self.device}, 计算类型: {self.compute_type}")
            
            self.logger.info("调用Whisper模型转写...")
            # 使用Whisper模型转写，添加优化的参数
            segments, info = self.model.transcribe(
                audio_float32,
                language=self.language,
                beam_size=5,
                vad_filter=True,  # 启用内置VAD过滤，帮助过滤静音部分
                vad_parameters=dict(
                    min_silence_duration_ms=500,  # 最小静音持续时间，帮助断句
                    speech_pad_ms=self.vad_speech_pad_ms,  # 语音前后填充时间
                    threshold=0.5  # VAD阈值
                ),
                initial_prompt=self.initial_prompt,  # 使用引导断句的初始提示
                temperature=0.0,  # 降低温度参数以提高确定性
                best_of=3,  # 生成多个候选结果并选择最佳
                word_timestamps=True,  # 获取单词级时间戳，提高断句精度
            )
            
            # 获取转写结果
            segments_list = list(segments)
            
            # 计算转写耗时
            transcribe_time = time.time() - transcribe_start_time
            
            # 记录详细的转写结果日志
            self.logger.info(f"转写完成，耗时 {transcribe_time:.2f}s，检测到 {len(segments_list)} 个分段")
            
            # 记录语言检测结果
            if hasattr(info, 'language') and info.language:
                self.logger.info(f"检测到的语言: {info.language}, 置信度: {info.language_probability:.4f}")
                
            # 处理转写结果
            if segments_list:
                self.logger.info(f"检测到 {len(segments_list)} 个语音分段")
                
                for i, segment in enumerate(segments_list):
                    self.logger.info(f"分段 {i+1}: [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
                
                # 合并所有分段文本
            transcription = " ".join([segment.text for segment in segments_list]).strip()
                
                # 应用后处理优化断句
                processed_transcription = self._postprocess_transcript(transcription)
                
                # 如果有变化，记录原始和处理后的文本
                if processed_transcription != transcription:
                    self.logger.info(f"原始转写: '{transcription}'")
                    self.logger.info(f"断句后处理: '{processed_transcription}'")
                    transcription = processed_transcription
                
                if transcription:
                    self.logger.info(f"最终转写结果: '{transcription}'")
                    
                    # 发送转写结果
                    if self.callback:
                        try:
                            if asyncio.iscoroutinefunction(self.callback):
                                await self.callback(transcription)
                            else:
                                self.callback(transcription)
                        except Exception as e:
                            self.logger.error(f"转写回调错误: {e}")
                            self.logger.error(traceback.format_exc())
                else:
                    self.logger.warning("转写结果为空")
            else:
                self.logger.warning("没有检测到语音分段")
                
            # 重置音频缓冲区
            self.logger.info(f"重置音频缓冲区，清理 {len(self.cumulative_audio)} 个样本")
            self.cumulative_audio = np.array([], dtype=np.int16)
            self.last_transcription_time = time.time()
            self.reset_buffer()
            
            self.logger.info(f"==========================================")
        
        except Exception as e:
            self.logger.error(f"转写过程中出错: {e}")
            self.logger.error(traceback.format_exc())
    
    async def _run_silero_check(self, audio_data):
        """异步运行Silero VAD检查"""
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.vad_executor, self._check_silero_speech, audio_data)
        except Exception as e:
            self.logger.error(f"运行Silero检查时出错: {e}")
            return False

    def _check_silero_speech(self, audio_data):
        """使用Silero VAD检测语音"""
        try:
            self.silero_working = True
            
            tensor_data = torch.tensor(audio_data).float()
            tensor_data = tensor_data.to(self.device if self.device.startswith('cuda') else 'cpu')
            tensor_data = tensor_data / 32768.0
            
            speech_prob = self.silero_vad_model(tensor_data, self.sample_rate).item()
            is_speech = speech_prob > 0.5
            
            self.silero_speech_active = is_speech
            
            if is_speech:
                self.logger.debug(f"Silero VAD 检测到语音，概率: {speech_prob:.4f}")
            
            self.silero_working = False
            return is_speech
            
        except Exception as e:
            self.logger.error(f"Silero VAD处理出错: {e}")
            self.silero_working = False
            return False

    def _check_webrtc_speech(self, audio_data):
        """使用WebRTC VAD检测语音"""
        try:
            frame_size = int(self.sample_rate * self.frame_duration / 1000)
            num_frames = len(audio_data) // frame_size
            is_speech_count = 0
            
            for i in range(num_frames):
                frame = audio_data[i * frame_size: (i + 1) * frame_size]
                frame_bytes = frame.tobytes()
                
                is_speech = self.vad_model.is_speech(frame_bytes, self.sample_rate)
                if is_speech:
                    is_speech_count += 1
            
            speech_ratio = is_speech_count / num_frames if num_frames > 0 else 0
            is_speech = speech_ratio >= self.vad_threshold
            
            self.webrtc_speech_active = is_speech
            
            if is_speech:
                self.logger.debug(f"WebRTC VAD 检测到语音，比例: {speech_ratio:.2f}")
            
            return is_speech
            
        except Exception as e:
            self.logger.error(f"WebRTC VAD处理出错: {e}")
            return False

    async def process_audio_chunk(self, audio_chunk: bytes) -> tuple:
        """处理音频数据块，使用双VAD协同工作"""
        try:
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            
            if len(audio_data) == 0:
                self.logger.warning("接收到空的音频数据")
                return False, np.array([], dtype=np.int16)
            
            # 检查音量
            abs_max = np.max(np.abs(audio_data))
            if abs_max < 500:
                self.logger.debug("音频音量太低")
                return False, audio_data
            
            # 首先使用WebRTC VAD快速检测
            is_webrtc_speech = self._check_webrtc_speech(audio_data)
            
            # 如果WebRTC检测到语音，使用Silero VAD进行确认
            is_speech = False
            if is_webrtc_speech:
                if self.use_silero_vad and self.silero_vad_model and not self.silero_working:
                    is_silero_speech = await self._run_silero_check(audio_data)
                    is_speech = is_webrtc_speech and is_silero_speech
                else:
                    is_speech = is_webrtc_speech
            
            # 处理语音结束检测
            if self.is_speaking and not is_speech:
                if self.silero_deactivity_detection and self.use_silero_vad:
                    is_silence_confirmed = not await self._run_silero_check(audio_data)
                else:
                    is_silence_confirmed = not is_webrtc_speech
                
                if is_silence_confirmed:
                    self.silence_frames += 1
                    self.logger.debug(f"检测到可能的语音结束，静音帧数: {self.silence_frames}")
                else:
                    self.silence_frames = 0
                    is_speech = True
            
            return is_speech, audio_data
                
        except Exception as e:
            self.logger.error(f"处理音频块时出错: {e}")
            self.logger.error(traceback.format_exc())
            return False, np.array([], dtype=np.int16)
    
    def _transcription_worker(self, queue):
        """多进程转写工作函数"""
        try:
            self.logger.info("转写工作进程已启动")
            
            # 初始化本地Whisper模型
            local_model = WhisperModel(
                model_size_or_path=self.model_type,
                device=self.device,
                compute_type=self.compute_type,
                download_root=config.MODEL_CACHE_DIR
            )
            
            while True:
                try:
                    # 从队列获取数据
                    item = queue.get()
                    if item is None:  # 终止信号
                        self.logger.info("收到终止信号，转写工作进程结束")
                        break
                        
                    # 解包音频数据和语言
                    audio_data, language = item
                    
                    # 转换为float32
                    audio_float32 = audio_data.astype(np.float32) / 32768.0
                    
                    # 使用Whisper模型转写
                    segments, info = local_model.transcribe(
                        audio_float32,
                        language=language,
                        beam_size=5,
                        vad_filter=True
                    )
                    
                    # 处理转写结果
                    segments_list = list(segments)
                    if segments_list:
                        transcription = " ".join([segment.text for segment in segments_list]).strip()
                        if transcription and self.realtime_callback:
                            try:
                                self.realtime_callback(transcription)
                            except Exception as e:
                                self.logger.error(f"实时回调出错: {e}")
                except Exception as e:
                    self.logger.error(f"转写工作进程处理出错: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"转写工作进程初始化出错: {e}")
            self.logger.error(traceback.format_exc())
    
    def reset_buffer(self) -> None:
        """重置音频缓冲区和状态"""
        self.audio_buffer = []
        self.silence_frames = 0
        self.speech_frames = 0
        self.force_vad_trigger = False
        # 重置文本稳定化相关状态
        self.text_storage = []
        self.stabilized_text = ""

    def _find_match_position(self, stabilized_text: str, new_text: str, match_length: int = 10) -> int:
        """查找稳定文本在新文本中的匹配位置"""
        try:
            # 如果任一文本太短，无法匹配
            if len(stabilized_text) < match_length or len(new_text) < match_length:
                return -1
                
            # 获取稳定文本的尾部
            tail = stabilized_text[-match_length:]
            
            # 在新文本中查找匹配
            for i in range(len(new_text) - match_length + 1):
                current = new_text[i:i+match_length]
                if current == tail:
                    return i
                    
            return -1
        except Exception as e:
            self.logger.error(f"文本匹配出错: {e}")
            return -1

    async def process_for_realtime(self, audio_data: np.ndarray) -> None:
        """处理音频数据进行实时转录"""
        if not self.enable_realtime_transcription or len(audio_data) == 0:
            return
            
        try:
            # 转换为float32
            audio_float32 = audio_data.astype(np.float32) / 32768.0
            
            # 使用实时模型转写，优化参数
            segments, info = self.realtime_model.transcribe(
                audio_float32,
                language=self.language,
                beam_size=3,  # 使用更小的beam size加速实时转录
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=400,  # 实时转写时使用较短的静音期
                    speech_pad_ms=self.vad_speech_pad_ms,
                    threshold=0.5
                ),
                temperature=0.0,  # 降低温度增加确定性
                initial_prompt=self.initial_prompt,  # 使用初始提示引导断句
                word_timestamps=True,  # 启用单词级时间戳
            )
            
            # 合并结果
            realtime_text = " ".join([segment.text for segment in segments]).strip()
            
            if realtime_text:
                # 检查最后一句是否完整
                last_sentence_complete = False
                
                if self.language == "zh":
                    last_sentence_complete = any(c in realtime_text[-5:] for c in ['。', '？', '！'])
                else:
                    last_sentence_complete = any(c in realtime_text[-5:] for c in ['.', '?', '!'])
                
                # 存储转录文本及完整性状态
                self.text_storage.append({
                    'text': realtime_text,
                    'complete': last_sentence_complete
                })
                
                # 只保留最近的几个结果
                if len(self.text_storage) > self.stabilization_window:
                    self.text_storage = self.text_storage[-self.stabilization_window:]
                
                # 如果有多个转录结果，尝试稳定化
                if len(self.text_storage) >= 2:
                    # 优先使用完整句子作为稳定点
                    if len(self.text_storage) >= 2 and self.text_storage[-2].get('complete', False):
                        # 提取上一个完整句子
                        last_text = self.text_storage[-2]['text']
                        
                        # 在中文处理
                        if self.language == "zh":
                            # 查找最后一个完整句子的结束位置
                            sentence_end = max(
                                last_text.rfind('。'), last_text.rfind('？'), 
                                last_text.rfind('！')
                            )
                        else:
                            # 其他语言
                            sentence_end = max(
                                last_text.rfind('.'), last_text.rfind('?'), 
                                last_text.rfind('!')
                            )
                        
                        if sentence_end > 0:
                            # 使用完整句子作为稳定文本
                            potential_stable = last_text[:sentence_end+1]
                            if len(potential_stable) > len(self.stabilized_text):
                                self.stabilized_text = potential_stable
                                self.logger.debug(f"使用完整句子更新稳定文本: '{self.stabilized_text}'")
                    else:
                        # 找出共同前缀
                        prefix = os.path.commonprefix([
                            t.get('text', '') for t in self.text_storage[-2:]
                        ])
                        
                        # 如果共同前缀较长，更新稳定文本
                        if len(prefix) > len(self.stabilized_text):
                            self.stabilized_text = prefix
                            self.logger.debug(f"使用共同前缀更新稳定文本: '{self.stabilized_text}'")
                
                # 对实时转录结果应用后处理，获得更好的断句
                processed_realtime_text = self._postprocess_transcript(realtime_text)
                if processed_realtime_text != realtime_text:
                    self.logger.debug(f"实时转录断句后处理: '{processed_realtime_text}'")
                    realtime_text = processed_realtime_text
                
                # 在当前转录中查找稳定文本
                if self.stabilized_text:
                    match_pos = self._find_match_position(
                        self.stabilized_text, 
                        realtime_text, 
                        self.match_threshold
                    )
                    
                    if match_pos >= 0:
                        # 构建组合文本
                        combined_text = (
                            self.stabilized_text + 
                            realtime_text[match_pos + len(self.stabilized_text):]
                        )
                        
                        # 应用后处理
                        processed_combined = self._postprocess_transcript(combined_text)
                        if processed_combined != combined_text:
                            combined_text = processed_combined
                        
                        # 发送稳定文本
                        if self.realtime_callback:
                            try:
                                self.logger.debug(f"发送稳定转录: '{combined_text}'")
                                if asyncio.iscoroutinefunction(self.realtime_callback):
                                    await self.realtime_callback(combined_text, is_stable=True)
                                else:
                                    self.realtime_callback(combined_text, is_stable=True)
                            except Exception as e:
                                self.logger.error(f"稳定文本回调错误: {e}")
                
                # 发送原始转录文本（非稳定）
                if self.realtime_callback:
                    try:
                        self.logger.debug(f"发送实时转录: '{realtime_text}'")
                        if asyncio.iscoroutinefunction(self.realtime_callback):
                            await self.realtime_callback(realtime_text, is_stable=False)
                        else:
                            self.realtime_callback(realtime_text, is_stable=False)
                    except Exception as e:
                        self.logger.error(f"实时回调错误: {e}")
                        
        except Exception as e:
            self.logger.error(f"实时转录处理出错: {e}")
            self.logger.error(traceback.format_exc())

    # 初始化实时转录模型
    async def _init_realtime_model(self):
        if self.enable_realtime_transcription and not self.use_main_model_for_realtime:
            try:
                self.logger.info(f"正在初始化实时转录模型: {self.realtime_model_type}...")
                
                self.realtime_model = WhisperModel(
                    model_size_or_path=self.realtime_model_type,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=config.MODEL_CACHE_DIR
                )
                
                # 尝试启用批量推理
                try:
                    self.logger.info("正在为实时模型启用批量推理...")
                    self.realtime_model = BatchedInferencePipeline(model=self.realtime_model)
                    self.logger.info("实时模型批量推理启用成功")
                except Exception as e:
                    self.logger.warning(f"实时模型批量推理启用失败: {e}")
                
                self.logger.info(f"实时转录模型初始化成功: {self.realtime_model_type}")
                
            except Exception as e:
                self.logger.error(f"初始化实时转录模型失败: {e}")
                self.logger.error(traceback.format_exc())
                self.enable_realtime_transcription = False
        
        # 如果启用实时转录但使用主模型
        if self.enable_realtime_transcription and self.use_main_model_for_realtime:
            self.realtime_model = self.model
            self.logger.info("使用主模型进行实时转录")

    def _get_default_prompt(self, language: str) -> str:
        """根据语言获取默认初始提示，用于引导Whisper模型更好地断句"""
        prompts = {
            "zh": "以下是文字记录。请在适当位置断句，使用正确的标点符号:",
            "en": "The following is a transcript. Please add proper sentence breaks and punctuation:",
            "ja": "以下は記録です。適切な場所で文を区切り、正しい句読点を使用してください:",
            "ko": "다음은 기록입니다. 적절한 곳에서 문장을 끊고 올바른 문장 부호를 사용하세요:",
            "fr": "Voici une transcription. Veuillez ajouter des pauses de phrase appropriées et la ponctuation:",
            "de": "Das Folgende ist ein Transkript. Bitte fügen Sie geeignete Satzpausen und Zeichensetzung hinzu:",
        }
        return prompts.get(language, prompts["en"])

    def _postprocess_transcript(self, transcript: str) -> str:
        """对转写结果进行后处理，改进断句"""
        if not transcript or not self.sentence_punctuation_boost:
            return transcript
            
        processed = transcript
        
        # 根据语言应用不同的后处理规则
        if self.language == "zh":
            # 中文断句处理
            # 1. 确保句子结尾有适当的标点
            if processed and not processed[-1] in ['。', '？', '！', '.', '?', '!']:
                # 根据上下文推断可能的句点类型
                if '?' in processed or '？' in processed:
                    processed += '？'
                elif '!' in processed or '！' in processed:
                    processed += '！'
                else:
                    processed += '。'
                    
            # 2. 修复中英文混合标点
            processed = re.sub(r'([\.!?])([。！？])', r'\2', processed)
            processed = re.sub(r'([。！？])([\.!?])', r'\1', processed)
            
            # 3. 在没有标点的长句中添加逗号
            if len(processed) > 20 and '，' not in processed and ',' not in processed:
                # 查找可能的停顿位置（如助词"的"后面）
                for pause_word in ['的', '了', '着', '地']:
                    if pause_word in processed[10:-10]:  # 避免在句子开头或结尾处添加
                        idx = processed.find(pause_word, 10)
                        if idx > 0 and idx < len(processed) - 1:
                            processed = processed[:idx+1] + '，' + processed[idx+1:]
                            break
                            
        elif self.language in ["en", "fr", "de"]:
            # 西方语言断句处理
            # 1. 确保句子结尾有标点
            if processed and not processed[-1] in ['.', '?', '!']:
                if '?' in processed:
                    processed += '?'
                elif '!' in processed:
                    processed += '!'
                else:
                    processed += '.'
            
            # 2. 确保每个句子开头大写
            parts = re.split(r'([.!?])\s+', processed)
            for i in range(0, len(parts), 2):
                if i < len(parts) and parts[i] and parts[i][0].islower():
                    parts[i] = parts[i][0].upper() + parts[i][1:]
            processed = ''.join(parts)
            
            # 3. 修复常见的标点错误
            processed = re.sub(r'\s+([.,;:!?])', r'\1', processed)  # 移除标点前的空格
            processed = re.sub(r'([.,;:!?])([^\s])', r'\1 \2', processed)  # 确保标点后有空格
            
        return processed 