# -*- coding: utf-8 -*-
"""音频处理工具模块。

从参考项目 tts.py 提取并适配：
- parse_timestamp: 时间戳字符串转毫秒
- split_text_by_timestamp: 解析带 [Speaker XX] 标签的字幕，返回 (timestamp, speaker, text)
- adjust_audio_speed: ffmpeg atempo 变速防重叠
- fast_overlay: numpy 一次性混音
"""

import os
import re
import subprocess
import numpy as np
from pydub import AudioSegment

# ---------- 混音参数 ---------- #
SR = 24_000          # 统一采样率
N_CH = 1             # 单声道
WIDTH = 2            # 16-bit
MAX_INT = 2 ** (8 * WIDTH - 1) - 1
FADE_MS = 10         # 每段音频首尾淡入淡出时长
TARGET_RMS = 0.1     # 响度归一化目标 RMS（相对 MAX_INT）
INTER_SENTENCE_GAP_MS = 180  # 相邻配音段间保留的停顿，避免"赶场"感


def parse_timestamp(timestamp):
    """将时间戳字符串转换为毫秒。

    支持 (h:mm:ss), (hh:mm:ss), (mm:ss), (h:mm:ss.ms), (hh:mm:ss.ms), (mm:ss.ms)
    以及三位数分钟，例如 (123:34.56)
    """
    match = re.match(r'[\(（](?:(\d{1,2}):)?(\d{1,3}):(\d{1,2})(?:\.(\d{1,3}))?[\)）]', timestamp)
    if match:
        hours, minutes, seconds, milliseconds = match.groups()
        total_ms = 0
        if hours:
            total_ms += int(hours) * 3600 * 1000
        total_ms += int(minutes) * 60 * 1000
        total_ms += int(seconds) * 1000
        if milliseconds:
            total_ms += int(milliseconds.ljust(3, '0'))
        return total_ms
    return 0


def split_text_by_timestamp(text):
    """解析带时间戳和说话人标签的字幕。

    支持格式:
        (00:01:23.456) [Speaker 00] 大家好欢迎来到节目

    逐行解析，严格剥离时间戳和说话人标签，只返回纯正文。
    返回: [(timestamp_str, speaker, content), ...]
    其中 timestamp_str 为纯时间戳(无括号)，speaker 为 "Speaker 00" 或空字符串
    """
    # 行格式: (时间戳) [Speaker XX] 正文  或  (时间戳) 正文
    line_pattern = re.compile(
        r'[\(（](?P<ts>\d{1,2}:\d{2}:\d{2}\.\d{1,3}|\d{1,2}:\d{2}:\d{2}|\d{1,3}:\d{2}(?:\.\d{1,3})?)[\)）]'
        r'(?:\s*\[(?P<speaker>[^\]]+)\])?'
        r'\s*(?P<content>.+)'
    )

    segments = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = line_pattern.match(line)
        if not m:
            continue
        ts = m.group("ts")
        speaker = (m.group("speaker") or "").strip()
        content = (m.group("content") or "").strip()
        if content:
            segments.append((ts, speaker, content))

    return segments


def _to_int16_samples(audio: AudioSegment):
    """把 AudioSegment 转成 int16 numpy 数组（统一采样率/声道/位深）"""
    audio = (audio.set_frame_rate(SR)
             .set_channels(N_CH)
             .set_sample_width(WIDTH))
    return np.frombuffer(audio.raw_data, dtype=np.int16)


def _apply_fades(samples, fade_ms=FADE_MS):
    """在音频首尾施加线性淡入淡出，避免片段拼接 click/pop。"""
    fade_samples = int(fade_ms * SR / 1000)
    if fade_samples <= 0 or len(samples) <= 2 * fade_samples:
        return samples
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    faded = samples.copy()
    faded[:fade_samples] = (faded[:fade_samples] * fade_in).astype(faded.dtype)
    faded[-fade_samples:] = (faded[-fade_samples:] * fade_in[::-1]).astype(faded.dtype)
    return faded


def _normalize_volume(samples, target_rms=TARGET_RMS):
    """按 RMS 响度归一化到统一水平。"""
    samples_float = samples.astype(np.float64)
    rms = np.sqrt(np.mean(samples_float ** 2))
    if rms <= 0:
        return samples
    scale = (target_rms * MAX_INT) / rms
    # 限制最大增益，避免底噪被过度放大
    scale = min(scale, 10.0)
    scaled = np.clip(samples_float * scale, -MAX_INT - 1, MAX_INT)
    return scaled.astype(np.int16)


def adjust_audio_speed(input_path, output_path, speed_factor):
    """用 ffmpeg atempo 调整音频速度，防止重叠时产生刺耳尖啸。

    返回 True 成功 / False 失败。
    """
    factor = max(0.5, min(speed_factor, 2.0))
    try:
        # 高倍速时适度低通，低倍速时无需额外滤波
        if factor > 1.5:
            filter_chain = f'lowpass=f=12000,atempo={factor:.4f}'
        else:
            filter_chain = f'atempo={factor:.4f}'
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', input_path,
             '-filter:a', filter_chain,
             output_path],
            capture_output=True, text=True, timeout=120
        )
        return result.returncode == 0
    except Exception as e:
        print(f"变速失败 {input_path}: {e}")
        return False


def fast_overlay(audio_entries):
    """numpy 一次性混音。

    audio_entries: [(start_ms, AudioSegment), ...]  已按 start_ms 排序
    返回混音后的 AudioSegment
    """
    if not audio_entries:
        return AudioSegment.silent(duration=0, frame_rate=SR)

    # 计算总长度
    last_start, last_audio = audio_entries[-1]
    total_ms = last_start + len(last_audio) + 1000  # 留 1s 尾巴
    total_samples = int(total_ms * SR / 1000)

    # 普通 numpy 数组作为混音板（替代共享内存，简化单进程场景）
    buf = np.zeros(total_samples * N_CH, dtype=np.float32)

    for start_ms, audio in audio_entries:
        samples = _to_int16_samples(audio)
        # 先归一化音量，再淡入淡出，最后叠加
        samples = _normalize_volume(samples)
        samples = _apply_fades(samples)
        samples_float = samples.astype(np.float32)
        start_sample = int(start_ms * SR / 1000)
        end_sample = start_sample + len(samples_float)
        if end_sample > total_samples:
            end_sample = total_samples
            samples_float = samples_float[:end_sample - start_sample]
        buf[start_sample:end_sample] += samples_float

    # 软限幅替代硬削波，听感更自然
    buf = np.tanh(buf / MAX_INT) * MAX_INT
    out_bytes = buf.astype(np.int16).tobytes()

    return AudioSegment(data=out_bytes, sample_width=WIDTH, frame_rate=SR, channels=N_CH)


def mix_segments(segments):
    """将生成好的音频片段按各自起始时间混音，自动变速防重叠。

    segments: [(start_ms, audio_path), ...] 已按 start_ms 排序
    返回: 混音后的 AudioSegment
    """
    if not segments:
        return AudioSegment.silent(duration=0, frame_rate=SR)

    loaded = []
    temp_files = []
    for start_ms, path in segments:
        try:
            audio = AudioSegment.from_file(path)
        except Exception as e:
            print(f"警告: 无法解码 {path}，跳过: {e}")
            continue
        loaded.append((start_ms, path, audio))

    # 变速防重叠
    final_entries = []
    for i, (start_ms, path, audio) in enumerate(loaded):
        end_time = start_ms + len(audio)
        if i < len(loaded) - 1:
            next_start = loaded[i + 1][0]
            if end_time > next_start + 100:
                target = next_start - start_ms - INTER_SENTENCE_GAP_MS
                # 修正：只要当前音频超过可用空间就尝试变速
                if target > 0 and len(audio) > target:
                    factor = min(len(audio) / target, 2.0)
                    tmp_out = path + '.speed.wav'
                    tmp_in = path
                    # 若原文件非 wav，先转 wav 以便 ffmpeg 处理
                    if not path.endswith('.wav'):
                        tmp_in = path + '.in.wav'
                        audio.export(tmp_in, format='wav')
                        temp_files.append(tmp_in)
                    if adjust_audio_speed(tmp_in, tmp_out, factor):
                        try:
                            audio = AudioSegment.from_file(tmp_out)
                            temp_files.append(tmp_out)
                        except Exception:
                            print(f"变速后解码失败，保留原音频: {path}")
                    else:
                        print(f"变速失败，保留原音频: {path}")
        final_entries.append((start_ms, audio))

    # 一次性混音
    result = fast_overlay(final_entries)

    # 清理临时文件
    for tf in temp_files:
        try:
            os.remove(tf)
        except OSError:
            pass

    return result
