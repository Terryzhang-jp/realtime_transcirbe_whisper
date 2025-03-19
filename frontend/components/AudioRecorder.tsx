import { useState, useEffect, useRef, useCallback } from 'react';
import audioTranscriptionService from '../lib/websocket';

interface AudioRecorderProps {
  onTranscriptionResult: (text: string) => void;
  language: string;
  modelType: string;
}

const AudioRecorder: React.FC<AudioRecorderProps> = ({
  onTranscriptionResult,
  language = 'zh',
  modelType = 'tiny',
}) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [audioDevices, setAudioDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string>('');
  const [audioLevel, setAudioLevel] = useState<number>(0);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const isMountedRef = useRef<boolean>(true);
  
  // 使用useRef存储回调函数，避免useEffect依赖变化导致重连
  const callbacksRef = useRef({
    onTranscriptionResult,
    language,
    modelType
  });
  
  // 当props变化时更新ref
  useEffect(() => {
    callbacksRef.current = {
      onTranscriptionResult,
      language,
      modelType
    };
  }, [onTranscriptionResult, language, modelType]);
  
  // 处理转写结果的回调函数
  const handleTranscriptionResult = useCallback((text: string) => {
    if (isMountedRef.current) {
      console.log(`%c====== 转写结果 ======`, 'background: #ff9800; color: white; padding: 4px 8px; border-radius: 4px;');
      console.log(`收到转写文本: "${text}"`);
      
      try {
        // 首先确认回调函数存在
        if (typeof callbacksRef.current.onTranscriptionResult !== 'function') {
          console.error('转写结果回调函数不是一个有效的函数');
          return;
        }
        
        console.log('调用父组件的onTranscriptionResult回调函数');
        callbacksRef.current.onTranscriptionResult(text);
        console.log('%c转写结果已成功传递给父组件', 'color: #4CAF50; font-weight: bold;');
      } catch (error) {
        console.error('调用父组件回调函数出错:', error);
      }
    } else {
      console.warn('组件已卸载，忽略转写结果');
    }
  }, []);
  
  // 连接WebSocket - 只在组件挂载时执行一次
  useEffect(() => {
    isMountedRef.current = true;
    const connectToWebSocket = async () => {
      try {
        if (!isMountedRef.current) return;
        
        setConnectionStatus('connecting');
        console.log('开始连接WebSocket...');
        
        await audioTranscriptionService.connect({
          onOpen: () => {
            if (!isMountedRef.current) return;
            setIsConnected(true);
            setConnectionStatus('connected');
            console.log('WebSocket连接已建立');
          },
          onClose: () => {
            if (!isMountedRef.current) return;
            setIsConnected(false);
            setConnectionStatus('disconnected');
            console.log('WebSocket连接已关闭');
            
            // 如果正在录音，停止录音
            if (isRecording) {
              console.log('WebSocket断开，停止录音');
              stopRecording();
            }
          },
          onError: (error) => {
            if (!isMountedRef.current) return;
            setIsConnected(false);
            setConnectionStatus('disconnected');
            console.error('WebSocket错误:', error);
          },
          onTranscription: (text) => {
            if (!isMountedRef.current) return;
            console.log('收到转写结果:', text);
            handleTranscriptionResult(text);
          },
          language: callbacksRef.current.language,
          model: callbacksRef.current.modelType,
        });
        
        console.log('WebSocket连接初始化完成');
      } catch (error) {
        if (!isMountedRef.current) return;
        console.error('连接WebSocket时出错:', error);
        setConnectionStatus('disconnected');
        
        // 5秒后尝试重连
        setTimeout(() => {
          if (isMountedRef.current) {
            console.log('尝试重新连接WebSocket...');
            connectToWebSocket();
          }
        }, 5000);
      }
    };

    // 尝试连接
    connectToWebSocket();

    // 组件卸载时清理
    return () => {
      console.log('AudioRecorder组件卸载，标记组件为未挂载');
      isMountedRef.current = false;
      // 不立即断开WebSocket连接，因为这可能导致其他组件的连接被中断
      // 而是让WebSocket服务自己管理连接生命周期
    };
  }, []); // 空依赖数组，确保只在组件挂载时运行一次
  
  // 使用单独的effect来更新配置
  useEffect(() => {
    if (isConnected) {
      console.log('更新WebSocket配置:', { language, modelType });
      audioTranscriptionService.updateConfig(language, modelType);
    }
  }, [language, modelType, isConnected]);
  
  // 列出可用的音频设备
  useEffect(() => {
    const getDevices = async () => {
      try {
        // 请求权限以列出音频设备
        await navigator.mediaDevices.getUserMedia({ audio: true });
        
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = devices.filter(device => device.kind === 'audioinput');
        setAudioDevices(audioInputs);
        
        if (audioInputs.length > 0 && !selectedDevice) {
          setSelectedDevice(audioInputs[0].deviceId);
        }
      } catch (error) {
        console.error('获取音频设备时出错:', error);
      }
    };
    
    getDevices();
    
    // 监听设备变化
    navigator.mediaDevices.addEventListener('devicechange', getDevices);
    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', getDevices);
    };
  }, [selectedDevice]);
  
  // 开始/停止录音
  const toggleRecording = async () => {
    if (!isConnected) {
      alert('WebSocket未连接，请等待连接建立');
      return;
    }
    
    if (isRecording) {
      stopRecording();
    } else {
      await startRecording();
    }
  };
  
  // 开始录音
  const startRecording = async () => {
    try {
      console.log('请求麦克风权限和初始化音频上下文...');
      // 创建音频上下文
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 16000,
      });
      audioContextRef.current = audioContext;
      
      // 请求麦克风访问权限
      console.log('请求麦克风访问权限，设备ID:', selectedDevice || '默认设备');
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId: selectedDevice ? { exact: selectedDevice } : undefined,
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      console.log('麦克风访问权限已获取，流已创建');
      mediaStreamRef.current = stream;
      
      // 先设置录音状态为true，确保handleAudioProcess能够处理数据
      setIsRecording(true);
      
      // 创建音频源
      const source = audioContext.createMediaStreamSource(stream);
      console.log('音频源已创建');
      
      // 创建分析器节点用于显示音量
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;
      
      // 创建处理器节点
      console.log('创建音频处理器节点');
      const processor = audioContext.createScriptProcessor(1024, 1, 1);
      processor.onaudioprocess = handleAudioProcess;
      source.connect(processor);
      processor.connect(audioContext.destination);
      processorRef.current = processor;
      
      // 开始音量可视化
      startVolumeMetering();
      
      console.log('录音已成功启动');
    } catch (error) {
      console.error('开始录音时出错:', error);
      alert(`无法访问麦克风: ${error}`);
      setIsRecording(false); // 如果出错，确保状态一致
    }
  };
  
  // 停止录音
  const stopRecording = () => {
    console.log('停止录音...');
    // 停止音量可视化
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    
    // 清理音频处理资源
    if (processorRef.current && audioContextRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    
    console.log('录音已停止，资源已清理');
    setIsRecording(false);
    setAudioLevel(0);
  };
  
  // 处理音频数据
  const handleAudioProcess = (e: AudioProcessingEvent) => {
    // 直接检查引用而不是状态，避免闭包捕获过时状态
    if (!mediaStreamRef.current || !audioContextRef.current || !processorRef.current) {
      console.log('音频资源未初始化，忽略音频处理');
      return;
    }
    
    if (!isConnected) {
      console.log('WebSocket未连接，忽略音频处理');
      return;
    }
    
    try {
      // 获取音频输入数据
      const inputBuffer = e.inputBuffer;
      
      // 将音频数据转换为16位整数
      const samples = new Float32Array(inputBuffer.getChannelData(0));
      const sampleCount = samples.length;
      
      // 验证采样率和声道配置
      if (inputBuffer.sampleRate !== 16000) {
        console.warn(`警告: 音频采样率 (${inputBuffer.sampleRate}Hz) 与期望的16000Hz不符`);
      }
      
      const int16Array = new Int16Array(sampleCount);
      
      // 添加音量增益处理
      const gainFactor = 8.0;  // 进一步增大音量，从4.0提高到8.0，可以根据需要调整
      
      // 记录音频统计信息，用于监控
      let maxAmplitude = 0;
      let sumAmplitude = 0;
      
      for (let i = 0; i < sampleCount; i++) {
        // 应用增益并限制范围在[-1,1]
        let sample = samples[i] * gainFactor;
        sample = Math.max(-1, Math.min(1, sample));
        
        // 将[-1,1]范围的浮点数转换为16位整数
        int16Array[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        
        // 统计信息
        const amplitude = Math.abs(int16Array[i]);
        maxAmplitude = Math.max(maxAmplitude, amplitude);
        sumAmplitude += amplitude;
      }
      
      // 计算平均振幅
      const avgAmplitude = sumAmplitude / sampleCount;
      
      // 记录音频统计信息，帮助调试
      console.log(`音频统计: 最大=${maxAmplitude}, 平均=${avgAmplitude.toFixed(2)}, 样本数=${sampleCount}`);
      
      // 如果音量太低，记录警告，但仍然发送数据
      if (maxAmplitude < 2000) {
        console.warn('警告: 音频音量可能太低，可能无法被语音检测捕获');
      }
      
      // 确保仍然连接正常再发送
      if (!isConnected) {
        console.warn('发送前检测到WebSocket未连接，跳过发送');
        return;
      }
      
      // 发送数据到WebSocket
      console.log(`发送音频数据: ${int16Array.buffer.byteLength} 字节`);
      try {
        audioTranscriptionService.sendAudioChunk(int16Array.buffer);
      } catch (error) {
        console.error('发送音频数据时出错:', error);
      }
    } catch (error) {
      console.error('处理音频数据时出错:', error);
    }
  };
  
  // 更新音频可视化
  const startVolumeMetering = () => {
    const updateVolume = () => {
      if (!analyserRef.current) return;
      
      const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
      analyserRef.current.getByteFrequencyData(dataArray);
      
      // 计算音量平均值
      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        sum += dataArray[i];
      }
      
      const average = sum / dataArray.length;
      setAudioLevel(average / 255); // 归一化为0-1范围
      
      animationFrameRef.current = requestAnimationFrame(updateVolume);
    };
    
    updateVolume();
  };
  
  // 处理设备选择变更
  const handleDeviceChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const deviceId = e.target.value;
    setSelectedDevice(deviceId);
    
    // 如果正在录音，停止并重新开始
    if (isRecording) {
      stopRecording();
      setTimeout(() => {
        startRecording();
      }, 500);
    }
  };
  
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 mb-8">
      <h2 className="text-2xl font-bold mb-4 text-gray-800 dark:text-white">音频录制</h2>
      
      {/* 连接状态 */}
      <div className="mb-4">
        <div className="flex items-center space-x-2">
          <div 
            className={`w-3 h-3 rounded-full ${
              connectionStatus === 'connected' ? 'bg-green-500' :
              connectionStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' :
              'bg-red-500'
            }`} 
          />
          <span className="text-sm text-gray-600 dark:text-gray-300">
            {connectionStatus === 'connected' ? '已连接' :
             connectionStatus === 'connecting' ? '正在连接...' :
             '未连接'}
          </span>
        </div>
      </div>
      
      {/* 设备选择 */}
      <div className="mb-4">
        <label htmlFor="device-select" className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-2">
          麦克风设备
        </label>
        <select
          id="device-select"
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500 dark:bg-gray-700 dark:text-white"
          value={selectedDevice}
          onChange={handleDeviceChange}
          disabled={isRecording}
        >
          {audioDevices.map(device => (
            <option key={device.deviceId} value={device.deviceId}>
              {device.label || `麦克风 ${device.deviceId.substring(0, 8)}...`}
            </option>
          ))}
          {audioDevices.length === 0 && (
            <option value="">无可用设备</option>
          )}
        </select>
      </div>
      
      {/* 音量显示 */}
      <div className="mb-4">
        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div 
            className="h-full bg-primary-500 transition-all duration-100"
            style={{ width: `${audioLevel * 100}%` }}
          />
        </div>
      </div>
      
      {/* 控制按钮 */}
      <button
        className={`w-full py-3 rounded-md font-bold transition-colors duration-200 ${
          isRecording
            ? 'bg-red-500 hover:bg-red-600 text-white'
            : 'bg-primary-500 hover:bg-primary-600 text-white'
        }`}
        onClick={toggleRecording}
        disabled={!isConnected}
      >
        {isRecording ? '停止录音' : '开始录音'}
      </button>
    </div>
  );
};

export default AudioRecorder; 