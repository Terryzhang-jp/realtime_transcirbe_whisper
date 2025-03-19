import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import dynamic from 'next/dynamic';

// 使用dynamic导入避免SSR渲染音频组件（它需要浏览器API）
const AudioRecorder = dynamic(
  () => import('../components/AudioRecorder'),
  { ssr: false }
);
import TranscriptionDisplay from '../components/TranscriptionDisplay';
import Header from '../components/Header';

const Home: React.FC = () => {
  const [transcriptions, setTranscriptions] = useState<string[]>([]);
  const [language, setLanguage] = useState<string>('zh');
  const [modelType, setModelType] = useState<string>('tiny');
  
  // 处理新的转写结果
  const handleTranscriptionResult = (text: string) => {
    setTranscriptions((prev) => [...prev, text]);
  };

  // 语言选项
  const languageOptions = [
    { value: 'zh', label: '中文' },
    { value: 'en', label: '英文' },
    { value: 'ja', label: '日文' },
    { value: 'ko', label: '韩文' },
    { value: 'fr', label: '法语' },
    { value: 'de', label: '德语' },
    { value: 'ru', label: '俄语' },
  ];

  // 模型选项
  const modelOptions = [
    { value: 'tiny', label: '超小型 (最快)' },
    { value: 'base', label: '基础型 (快速)' },
    { value: 'small', label: '小型 (平衡)' },
    { value: 'medium', label: '中型 (较精确)' },
    { value: 'large', label: '大型 (最精确, 较慢)' },
  ];
  
  return (
    <div className="min-h-screen flex flex-col">
      <Head>
        <title>实时语音转写系统</title>
        <meta name="description" content="基于FastAPI和Next.js的实时语音转写系统" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <Header />

      <main className="container mx-auto px-4 py-8 max-w-6xl flex-grow">
        <h1 className="text-4xl font-bold text-center mb-8 text-gray-800 dark:text-white">
          实时语音转写系统
        </h1>
        
        {/* 状态监控链接 */}
        <div className="mb-4 text-center">
          <a 
            href="http://localhost:8000/test/ws-status" 
            target="_blank" 
            rel="noopener noreferrer"
            className="inline-block px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
          >
            查看WebSocket状态
          </a>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
          {/* 配置面板 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-bold mb-4 text-gray-800 dark:text-white">配置选项</h2>
            
            <div className="space-y-4">
              {/* 语言选择 */}
              <div>
                <label htmlFor="language-select" className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-2">
                  识别语言
                </label>
                <select
                  id="language-select"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500 dark:bg-gray-700 dark:text-white"
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                >
                  {languageOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
              
              {/* 模型选择 */}
              <div>
                <label htmlFor="model-select" className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-2">
                  模型大小
                </label>
                <select
                  id="model-select"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500 dark:bg-gray-700 dark:text-white"
                  value={modelType}
                  onChange={(e) => setModelType(e.target.value)}
                >
                  {modelOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
              
              {/* 系统信息 */}
              <div className="mt-8 border-t pt-4 dark:border-gray-700">
                <h3 className="text-lg font-medium text-gray-800 dark:text-white mb-2">系统信息</h3>
                <ul className="text-sm text-gray-600 dark:text-gray-300 space-y-1">
                  <li>• 支持多种语言识别</li>
                  <li>• 支持多种Whisper模型</li>
                  <li>• 自动语音检测</li>
                  <li>• 低延迟实时转写</li>
                  <li>• 结果可导出保存</li>
                </ul>
              </div>
            </div>
          </div>
          
          {/* 音频录制组件 */}
          <AudioRecorder 
            onTranscriptionResult={handleTranscriptionResult}
            language={language}
            modelType={modelType}
          />
        </div>
        
        {/* 转写结果显示 */}
        <TranscriptionDisplay transcriptions={transcriptions} />
      </main>

      <footer className="mt-auto py-6 border-t border-gray-200 dark:border-gray-800">
        <div className="container mx-auto px-4 text-center text-gray-500 dark:text-gray-400">
          <p>© {new Date().getFullYear()} 实时语音转写系统 | 基于FastAPI和Next.js开发</p>
        </div>
      </footer>
    </div>
  );
};

export default Home; 