import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

interface TranscriptionDisplayProps {
  transcriptions: string[];
}

interface TranscriptionItem {
  text: string;
  timestamp: Date;
}

const TranscriptionDisplay: React.FC<TranscriptionDisplayProps> = ({ transcriptions }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [items, setItems] = useState<TranscriptionItem[]>([]);
  
  // 将传入的转写结果转换为带时间戳的项目
  useEffect(() => {
    const newItems = transcriptions.map((text, index) => {
      // 如果items中已有此索引的项，则保留其时间戳
      if (index < items.length) {
        return items[index];
      }
      // 否则创建新项目
      return {
        text,
        timestamp: new Date()
      };
    });
    setItems(newItems);
    
    // 记录接收到新转写结果的信息
    if (newItems.length > items.length) {
      console.log(`%c接收到新转写结果 (总数: ${newItems.length})`, 'background: #9C27B0; color: white; padding: 2px 6px; border-radius: 4px;');
    }
  }, [transcriptions]);
  
  // 当新的转写结果出现时，滚动到底部
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [items]);
  
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold mb-4 text-gray-800 dark:text-white flex justify-between items-center">
        <span>转写结果</span>
        {items.length > 0 && (
          <span className="text-sm font-normal bg-primary-100 dark:bg-primary-900 text-primary-800 dark:text-primary-200 px-2 py-1 rounded-full">
            共 {items.length} 条
          </span>
        )}
      </h2>
      
      <div 
        ref={containerRef}
        className="h-96 overflow-y-auto border border-gray-200 dark:border-gray-700 rounded-lg p-4 bg-gray-50 dark:bg-gray-900"
      >
        {items.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500 dark:text-gray-400">
            <p>开始录音后，转写结果将显示在这里</p>
          </div>
        ) : (
          <div className="space-y-4">
            {items.map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className="p-3 bg-primary-50 dark:bg-gray-800 rounded-lg border-l-4 border-primary-500"
              >
                <p className="text-gray-800 dark:text-gray-200">{item.text}</p>
                <div className="mt-1 text-xs text-gray-500 dark:text-gray-400 text-right">
                  {item.timestamp.toLocaleTimeString()}
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>
      
      <div className="mt-4 flex justify-between items-center">
        <div className="text-sm text-gray-500 dark:text-gray-400">
          {items.length > 0 ? `最后更新: ${items[items.length-1].timestamp.toLocaleTimeString()}` : "尚无转写结果"}
        </div>
        <button 
          className="px-4 py-2 text-sm font-medium text-primary-600 dark:text-primary-400 hover:text-primary-800 dark:hover:text-primary-300 disabled:opacity-50"
          onClick={() => {
            // 创建包含转写结果的文本
            const text = items.map(item => `[${item.timestamp.toLocaleTimeString()}] ${item.text}`).join('\n\n');
            
            // 创建Blob对象
            const blob = new Blob([text], { type: 'text/plain' });
            
            // 创建URL
            const url = URL.createObjectURL(blob);
            
            // 创建下载链接
            const a = document.createElement('a');
            a.href = url;
            a.download = `转写结果_${new Date().toISOString().split('T')[0]}.txt`;
            
            // 模拟点击
            document.body.appendChild(a);
            a.click();
            
            // 清理
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
          }}
          disabled={items.length === 0}
        >
          导出结果
        </button>
      </div>
    </div>
  );
};

export default TranscriptionDisplay; 