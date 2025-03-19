import numpy as np
from scipy.io.wavfile import write
import os
import librosa
import soundfile as sf

def generate_test_audio():
    """生成一个简单的测试音频文件"""
    # 创建测试目录
    test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tests")
    os.makedirs(test_dir, exist_ok=True)
    
    # 生成一个简单的音频信号 (正弦波)
    sample_rate = 16000
    duration_seconds = 3
    frequency = 440  # 440 Hz, A4音符
    
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
    audio_data = np.sin(2 * np.pi * frequency * t) * 32767 / 2  # 半音量的正弦波
    audio_data = audio_data.astype(np.int16)
    
    # 保存为WAV文件
    output_file = os.path.join(test_dir, "test_audio.wav")
    write(output_file, sample_rate, audio_data)
    
    print(f"生成了测试音频文件: {output_file}")
    return output_file

def generate_complex_test_audio():
    """生成一个更复杂的测试音频文件，模拟人声"""
    # 创建测试目录
    test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tests")
    os.makedirs(test_dir, exist_ok=True)
    
    # 生成一个复杂的音频信号 (多个频率的组合)
    sample_rate = 16000
    duration_seconds = 3
    
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
    
    # 基础频率 (模拟人声基频，约为85-255Hz)
    fundamental_freq = 150  # 基频
    
    # 创建多个谐波，模拟人声
    audio_data = np.sin(2 * np.pi * fundamental_freq * t) * 0.5  # 基频
    audio_data += np.sin(2 * np.pi * (fundamental_freq * 2) * t) * 0.25  # 第一谐波
    audio_data += np.sin(2 * np.pi * (fundamental_freq * 3) * t) * 0.125  # 第二谐波
    audio_data += np.sin(2 * np.pi * (fundamental_freq * 4) * t) * 0.0625  # 第三谐波
    
    # 添加一些变化，模拟音高变化
    modulation = 1.0 + 0.1 * np.sin(2 * np.pi * 2 * t)  # 缓慢的调制
    audio_data *= modulation
    
    # 添加一些噪声
    noise = np.random.normal(0, 0.05, len(audio_data))
    audio_data += noise
    
    # 应用淡入淡出效果
    fade_samples = int(sample_rate * 0.1)  # 100毫秒淡入淡出
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    audio_data[:fade_samples] *= fade_in
    audio_data[-fade_samples:] *= fade_out
    
    # 正规化并转换为16位整数
    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9  # 留出一些余量
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # 保存为WAV文件
    output_file = os.path.join(test_dir, "test_voice.wav")
    write(output_file, sample_rate, audio_data)
    
    print(f"生成了模拟人声测试文件: {output_file}")
    return output_file

def copy_sample_audio_if_exists():
    """如果有示例音频文件，复制到测试目录"""
    # 创建测试目录
    test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tests")
    os.makedirs(test_dir, exist_ok=True)
    
    # 检查是否存在sample_audio.wav文件
    sample_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sample_audio.wav")
    if os.path.exists(sample_file):
        # 加载音频文件
        audio_data, sr = librosa.load(sample_file, sr=16000, mono=True)
        
        # 确保音频长度适中（如果太长，截取前5秒）
        if len(audio_data) > 5 * 16000:
            audio_data = audio_data[:5 * 16000]
        
        # 将音频增益调整到合适水平
        audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9
        
        # 保存处理后的音频
        output_file = os.path.join(test_dir, "sample_audio.wav")
        sf.write(output_file, audio_data, 16000, subtype='PCM_16')
        
        print(f"处理并复制了示例音频: {output_file}")
        return output_file
    else:
        print("未找到示例音频文件")
        return None

if __name__ == "__main__":
    generate_test_audio()
    generate_complex_test_audio()
    copy_sample_audio_if_exists() 