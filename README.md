# 实时语音转写系统

这是一个基于FastAPI和Next.js的实时语音转写系统，支持中英文等多种语言的语音识别。

## 功能特点

- 实时语音录制和转写
- WebSocket实时通信
- 支持多种语言识别（中文、英文、日语等）
- 多种转写模型大小选择（从tiny到large）
- 自动语音活动检测(VAD)
- 美观的用户界面
- 实时显示转写结果
- 结果导出保存

## 技术栈

### 后端

- **FastAPI**：高性能的Python Web框架
- **WebSocket**：实时双向通信
- **Faster Whisper**：高效的语音识别模型
- **WebRTC VAD**：语音活动检测
- **PyAudio**：音频处理库
- **Uvicorn**：异步ASGI服务器

### 前端

- **Next.js**：React框架
- **TypeScript**：类型安全的JavaScript
- **Tailwind CSS**：实用优先的CSS框架
- **Web Audio API**：浏览器音频处理
- **WebSocket**：实时通信

## 安装和运行

### 系统要求

- Python 3.8+
- Node.js 14+
- 推荐使用CUDA GPU以获得更好的性能（但不是必需的）

### 后端

1. 进入后端目录：
```bash
cd backend
```

2. 下载模型文件：
```bash
# 创建models目录
mkdir -p models

# 下载模型（根据需要选择不同大小的模型）
# tiny模型（最小，速度最快，但准确性较低）
python -c "from faster_whisper import WhisperModel; WhisperModel('tiny', download_root='models')"

# base模型（平衡大小和准确性）
python -c "from faster_whisper import WhisperModel; WhisperModel('base', download_root='models')"

# small模型
python -c "from faster_whisper import WhisperModel; WhisperModel('small', download_root='models')"

# medium模型（推荐，较好的准确性）
python -c "from faster_whisper import WhisperModel; WhisperModel('medium', download_root='models')"

# large-v3模型（最高准确性，但需要较多资源）
python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', download_root='models')"
```

3. 运行启动脚本：
```bash
chmod +x run.sh
./run.sh
```

这将创建虚拟环境，安装依赖，并启动FastAPI服务器。

### 前端

1. 进入前端目录：
```bash
cd frontend
```

2. 安装依赖：
```bash
npm install
# 或使用 yarn
yarn install
```

3. 运行开发服务器：
```bash
npm run dev
# 或使用 yarn
yarn dev
```

4. 在浏览器中访问：
```
http://localhost:3000
```

## 使用说明

1. 打开网页后，系统会自动连接到后端服务
2. 选择所需的识别语言和模型大小
3. 允许浏览器使用麦克风
4. 点击"开始录音"按钮
5. 开始说话，系统会实时转写您的语音
6. 转写结果会显示在下方的结果区域
7. 点击"停止录音"按钮结束录音
8. 可以点击"导出结果"按钮导出转写文本

## 自定义配置

- 后端端口：修改`backend/app/main.py`中的端口配置
- 前端API地址：修改`frontend/.env.local`文件中的`NEXT_PUBLIC_API_URL`
- 模型配置：修改`backend/app/audio/audio_processor.py`中的默认参数

## 部署

### 后端部署

可以使用Docker部署后端：

```bash
# 构建镜像
docker build -t realtime-transcription-backend ./backend

# 运行容器
docker run -p 8000:8000 realtime-transcription-backend
```

### 前端部署

Next.js应用可以部署到Vercel等平台：

```bash
# 构建生产版本
cd frontend
npm run build

# 启动生产服务器
npm start
```

## 注意事项

- 确保系统已安装Python 3.8+和Node.js 14+
- 确保麦克风工作正常
- 使用Chrome或Edge浏览器获得最佳体验
- 需要稳定的网络连接
- 较大的模型需要较多的计算资源
- 在生产环境中应配置CORS策略

## 许可证

MIT License 