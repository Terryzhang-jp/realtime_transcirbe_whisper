from scipy.signal import butter, filtfilt, resample_poly
import pyaudio
import numpy as np
import logging
from typing import Optional, List, Dict, Any
from app import config

DESIRED_RATE = 16000
CHUNK_SIZE = 1024
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1

class AudioInput:
    def __init__(
            self,
            input_device_index: Optional[int] = None,
            debug_mode: bool = config.DEBUG,
            target_samplerate: int = config.DEFAULT_SAMPLE_RATE,
            chunk_size: int = config.DEFAULT_CHUNK_SIZE,
            audio_format: int = pyaudio.paInt16,
            channels: int = 1,
            resample_to_target: bool = True,
        ):

        self.input_device_index = input_device_index
        self.debug_mode = debug_mode
        self.audio_interface = None
        self.stream = None
        self.device_sample_rate = None
        self.target_samplerate = target_samplerate
        self.chunk_size = chunk_size
        self.audio_format = audio_format
        self.channels = channels
        self.resample_to_target = resample_to_target
        
        # 配置日志
        logging.basicConfig(
            level=logging.DEBUG if debug_mode else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("AudioInput")

    def get_supported_sample_rates(self, device_index: int) -> List[int]:
        """测试指定设备支持哪些标准采样率。"""
        standard_rates = [8000, 9600, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000]
        supported_rates = []

        device_info = self.audio_interface.get_device_info_by_index(device_index)
        max_channels = device_info.get('maxInputChannels')

        for rate in standard_rates:
            try:
                if self.audio_interface.is_format_supported(
                    rate,
                    input_device=device_index,
                    input_channels=max_channels,
                    input_format=self.audio_format,
                ):
                    supported_rates.append(rate)
            except:
                continue
        return supported_rates

    def _get_best_sample_rate(self, actual_device_index: int, desired_rate: int) -> int:
        """确定设备的最佳可用采样率。"""
        try:
            device_info = self.audio_interface.get_device_info_by_index(actual_device_index)
            supported_rates = self.get_supported_sample_rates(actual_device_index)

            if desired_rate in supported_rates:
                return desired_rate

            # 找不到完全匹配的，选择支持的最高采样率
            if supported_rates:
                return max(supported_rates)

            return int(device_info.get('defaultSampleRate', 44100))

        except Exception as e:
            self.logger.warning(f"确定采样率时出错: {e}")
            return 44100  # 安全的回退值

    def list_devices(self) -> List[Dict[str, Any]]:
        """列出所有可用的音频输入设备及其支持的采样率。"""
        devices = []
        try:
            self.audio_interface = pyaudio.PyAudio()
            device_count = self.audio_interface.get_device_count()

            for i in range(device_count):
                device_info = self.audio_interface.get_device_info_by_index(i)
                device_name = device_info.get('name')
                max_input_channels = device_info.get('maxInputChannels', 0)

                if max_input_channels > 0:  # 只考虑具有输入功能的设备
                    supported_rates = self.get_supported_sample_rates(i)
                    devices.append({
                        "index": i,
                        "name": device_name,
                        "max_input_channels": max_input_channels,
                        "supported_rates": supported_rates
                    })
                    self.logger.debug(f"设备 {i}: {device_name}")
                    self.logger.debug(f"  支持的采样率: {supported_rates}")

            return devices
        except Exception as e:
            self.logger.error(f"列出设备时出错: {e}")
            return []
        finally:
            if self.audio_interface:
                self.audio_interface.terminate()

    def setup(self) -> bool:
        """初始化音频接口并打开流"""
        try:
            self.audio_interface = pyaudio.PyAudio()

            if self.debug_mode:
                self.logger.debug(f"输入设备索引: {self.input_device_index}")
                
            # 获取实际设备索引
            actual_device_index = (self.input_device_index if self.input_device_index is not None 
                                else self.audio_interface.get_default_input_device_info()['index'])
            
            if self.debug_mode:
                self.logger.debug(f"实际选择的设备索引: {actual_device_index}")
                
            self.input_device_index = actual_device_index
            self.device_sample_rate = self._get_best_sample_rate(actual_device_index, self.target_samplerate)

            if self.debug_mode:
                self.logger.debug(f"在设备 {self.input_device_index} 上设置音频，采样率为 {self.device_sample_rate}")

            try:
                self.stream = self.audio_interface.open(
                    format=self.audio_format,
                    channels=self.channels,
                    rate=self.device_sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                    input_device_index=self.input_device_index,
                )
                if self.debug_mode:
                    self.logger.debug(f"音频录制成功初始化，采样率为 {self.device_sample_rate} Hz")
                return True
            except Exception as e:
                self.logger.error(f"以 {self.device_sample_rate} Hz 初始化音频流失败: {e}")
                return False

        except Exception as e:
            self.logger.error(f"初始化音频录制时出错: {e}")
            if self.audio_interface:
                self.audio_interface.terminate()
            return False

    def lowpass_filter(self, signal: np.ndarray, cutoff_freq: float, sample_rate: float) -> np.ndarray:
        """
        对信号应用低通巴特沃斯滤波器，防止混叠。

        参数:
            signal (np.ndarray): 要过滤的输入音频信号
            cutoff_freq (float): 截止频率（Hz）
            sample_rate (float): 输入信号的采样率（Hz）

        返回:
            np.ndarray: 过滤后的音频信号
        """
        # 计算奈奎斯特频率（采样率的一半）
        nyquist_rate = sample_rate / 2.0

        # 将截止频率标准化为奈奎斯特率（butter()需要）
        normal_cutoff = cutoff_freq / nyquist_rate

        # 设计巴特沃斯滤波器
        b, a = butter(5, normal_cutoff, btype='low', analog=False)

        # 应用零相位滤波（前向和后向）
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    def resample_audio(self, pcm_data: np.ndarray, target_sample_rate: int, original_sample_rate: int) -> np.ndarray:
        """
        过滤并将音频数据重采样到目标采样率。

        参数:
            pcm_data (np.ndarray): 输入音频数据
            target_sample_rate (int): 所需的输出采样率（Hz）
            original_sample_rate (int): 输入的原始采样率（Hz）

        返回:
            np.ndarray: 重采样的音频数据
        """
        if target_sample_rate < original_sample_rate:
            # 降采样时使用低通滤波器
            pcm_filtered = self.lowpass_filter(pcm_data, target_sample_rate / 2, original_sample_rate)
            resampled = resample_poly(pcm_filtered, target_sample_rate, original_sample_rate)
        else:
            # 升采样时不使用低通滤波器
            resampled = resample_poly(pcm_data, target_sample_rate, original_sample_rate)
        return resampled

    def read_chunk(self) -> bytes:
        """读取一块音频数据"""
        if not self.stream:
            raise RuntimeError("音频流未初始化")
        return self.stream.read(self.chunk_size, exception_on_overflow=False)

    def cleanup(self) -> None:
        """清理音频资源"""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            if self.audio_interface:
                self.audio_interface.terminate()
                self.audio_interface = None
        except Exception as e:
            self.logger.error(f"清理音频资源时出错: {e}") 