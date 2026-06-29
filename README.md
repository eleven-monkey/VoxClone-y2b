# VoxClone-y2b

YouTube 视频翻译配音搬运（语音克隆版）

基于 GitHub Actions 的 YouTube 多人谈话节目翻译配音搬运工具。手动触发工作流后自动完成：
**下载音频 → WhisperX 转写 + 说话人分离 → 自动截取每位说话人参考音 → LLM 翻译字幕为中文 → moss-tts 语音克隆逐句配音 → 替换原音轨 → 上传 B 站**。

全程 CPU 运行，无需 GPU。创新点：**自动从原视频按说话人分离结果截取参考音，用 moss-tts 语音克隆实现多人个性化配音**。

## 工作流总览

```
Job1: ASR+说话人分离  →  artifacts(word_level.txt, speaker_ref/, speaker_ref_map.json)
Job2: 字幕翻译        →  artifacts(word_level_translated.txt)
Job3: 语音克隆配音    →  artifacts(dubbed_audio.mp3)
Job4: 视频合成+上传   →  B站
```

各 Job 独立超时与依赖安装，通过 artifacts 传递中间产物，单步失败可重跑，规避 Actions 单 job 6 小时上限。

## 快速开始

### 1. 配置 Secrets

在仓库 **Settings → Secrets and variables → Actions** 中添加以下 Secrets：

| Secret | 说明 | 获取方式 |
| --- | --- | --- |
| `AI_API_URL` | LLM 接口地址 | SiliconFlow: `https://api.siliconflow.cn/v1/chat/completions` |
| `AI_API_KEY` | LLM API Key | SiliconFlow 控制台获取 |
| `AI_MODEL` | 翻译用模型名 | 如 `THUDM/GLM-4-9B-0414`、`Qwen/Qwen3-8B` |
| `BILIBILI_SESSDATA` | B站 sessdata | 浏览器登录 B 站后从 Cookie 获取 |
| `BILIBILI_JCT` | B站 bili_jct | 同上（CSRF token） |
| `BILIBILI_BUVID3` | B站 buvid3 | 同上 |
| `YT_COOKIES` | YouTube cookies | yt-dlp `--cookies` 格式的 cookies 文件内容 |
| `HF_TOKEN` | HuggingFace token | 需在 HF 网页端接受 pyannote 协议 |

> **HuggingFace 协议**：WhisperX 的 Pyannote 说话人分离模型需要授权。登录 [HuggingFace](https://huggingface.co)，访问 [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) 和 [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) 页面接受使用协议，然后在 Settings → Access Tokens 创建 Read 权限 token。

### 2. 触发工作流

进入仓库 **Actions** 页面，选择「YouTube 视频翻译配音搬运」工作流，点击 **Run workflow**，填入：

- `video_url`：YouTube 视频 URL（必填）
- `max_speakers`：最大说话人数量（可选，留空自动检测）
- `translate_model`：翻译模型名（可选，留空用 `AI_MODEL`）

### 3. 等待完成

15-40 分钟视频在 CPU 模式下总耗时约 3-5 小时。各 Job 产物可通过 Actions 页面的 Artifacts 下载查看。

## 字幕格式约定

贯穿全流程的统一格式：

```
(HH:MM:SS.mmm) [Speaker 00] 文本内容
```

- ASR 输出：原文 + 时间戳 + 说话人标签
- 翻译输出：中文译文，时间戳和说话人标签原样保留
- TTS 输入：按 `[Speaker XX]` 匹配对应参考音克隆配音

## 项目结构

```
.
├── .github/workflows/dub_pipeline.yml   # 主工作流（4 个 job）
├── src/
│   ├── asr_diarize.py                   # WhisperX ASR + 说话人分离 + 参考音截取
│   ├── translate_subtitles.py           # LLM 并行字幕翻译
│   ├── tts_clone.py                     # moss-tts 语音克隆配音（核心）
│   ├── audio_utils.py                   # 音频工具（变速防重叠 + numpy 混音）
│   └── upload_bilibili.py               # 视频下载 + 音轨替换 + B站上传
├── config/api_config.json               # LLM API 配置模板
├── requirements/
│   ├── common.txt                       # 公共依赖
│   ├── asr.txt                          # ASR job 依赖
│   └── tts.txt                          # TTS job 依赖
├── scripts/install_moss_tts.sh          # moss-tts 安装脚本
└── README.md
```

## 技术栈

| 模块 | 技术 |
| --- | --- |
| ASR + 说话人分离 | WhisperX (small, int8, CPU) + Pyannote 3.0 |
| 语音克隆 TTS | MOSS-TTS-Nano（ONNX CPU 推理，单进程批量克隆） |
| 字幕翻译 | SiliconFlow OpenAI 兼容 API + ThreadPoolExecutor 并行 |
| 音频处理 | ffmpeg（变速、混音、音轨替换）+ pydub + numpy（共享内存混音） |
| 视频下载 | yt-dlp + Deno（YouTube JS 解密） |
| B 站上传 | bilibili-api-python（七牛线路，指数退避重试） |

## 核心创新：自动参考音截取

传统语音克隆需要手动准备参考音频。本项目自动完成：

1. WhisperX + Pyannote 分离出每位说话人的时间片段
2. 对每位说话人，从其所有片段中按时长降序排列
3. 取最长片段（≥5 秒则使用；不足则拼接多段直至满足）
4. 用 ffmpeg 从原音频裁剪为 wav，作为该说话人的克隆参考音
5. moss-tts 推理时按 `[Speaker XX]` 标签切换对应参考音

## 本地测试

各模块可独立运行测试：

```bash
# ASR + 说话人分离（需 HF_TOKEN 环境变量）
export HF_TOKEN=your_token
python src/asr_diarize.py --url "https://youtube.com/watch?v=xxx" --output_dir work

# 字幕翻译（需 AI_API_URL / AI_API_KEY 环境变量）
export AI_API_URL="https://api.siliconflow.cn/v1/chat/completions"
export AI_API_KEY=your_key
python src/translate_subtitles.py --input work/word_level.txt --output work/word_level_translated.txt

# 语音克隆配音（需先安装 moss-tts）
python src/tts_clone.py --input work/word_level_translated.txt --output work/dubbed_audio.mp3 --ref_map work/speaker_ref_map.json --moss_tts_dir MOSS-TTS-Nano

# 视频合成上传
python src/upload_bilibili.py --url "https://youtube.com/watch?v=xxx" --audio_path work/dubbed_audio.mp3
```

## 性能说明（CPU 模式，15-40 分钟视频）

| 步骤 | 预估耗时 |
| --- | --- |
| 音频下载 | 1-3 分钟 |
| WhisperX 转写 (small, int8) | 30-80 分钟 |
| Pyannote 说话人分离 | 20-60 分钟 |
| LLM 翻译（并行 5 线程） | 2-5 分钟 |
| moss-tts 逐句克隆配音 | 1-3 小时 |
| 视频下载 + 音轨替换 + 上传 | 5-10 分钟 |

二次运行时 HF 模型和 ONNX 模型有缓存，ASR 和 TTS 步骤会显著加速。

## 参考来源

- `actionsautopython`：参考项目（YouTube 字幕 + edge-tts 配音）
- `wshiperx说话人识别asr.py`：WhisperX + 说话人分离 + edge-tts Colab 流程
- `moss-tts代码参考.txt`：MOSS-TTS-Nano 安装与 ONNX 推理参考
