# -*- coding: utf-8 -*-
"""音频处理工具模块。

从参考项目 tts.py 提取并适配：
- parse_timestamp: 时间戳字符串转毫秒
- split_text_by_timestamp: 解析带 [Speaker XX] 标签的字幕，返回 (timestamp, speaker, text)
- adjust_audio_speed: ffmpeg atempo + lowpass 变速防重叠
- fast_overlay: numpy 共享内存一次性混音
"""

import os
import re
import subprocess
import numpy as np
from multiprocessing import shared_memory
from pydub import AudioSegment

# ---------- 混音参数 ---------- #
SR = 24_000          # 统一采样率
N_CH = 1             # 单声道
WIDTH = 2            # 16-bit
MAX_INT = 2 ** (8 * WIDTH - 1) - 1


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
    """按时间戳分割文本。

    支持带说话人标签的格式:
        (00:01:23.456) [Speaker 00] 大家好欢迎来到节目

    返回: [(timestamp_str, speaker, content), ...]
    其中 timestamp_str 为纯时间戳(无括号)，speaker 为 "Speaker 00" 或空字符串
    """
    # 时间戳模式
    ts_pattern = r'[\(（](\d{1,2})?:?(\d{1,3}):(\d{1,2})(?:\.(\d{1,3}))?[\)）]'
    # 说话人标签模式（可选）
    speaker_pattern = r'\s*\[([^\]]+)\]\s*'
    full_pattern = ts_pattern + r'(' + speaker_pattern + r')?(.+?)(?=' + ts_pattern + r'|$)'

    segments = []
    for match in re.finditer(full_pattern, text, re.DOTALL):
        # 组装时间戳字符串（去括号）
        h, m, s, ms = match.group(1), match.group(2), match.group(3), match.group(4)
        ts_parts = []
        if h:
            ts_parts.append(f"{int(h):02d}:{int(m):02d}:{int(s):02d}")
            if ms:
                ts_parts[-1] += f".{ms.ljust(3, '0')}"
        else:
            ts_parts.append(f"{int(m)}:{int(s):02d}")
            if ms:
                ts_parts[-1] += f".{ms.ljust(3, '0')}"
        timestamp_str = ts_parts[0]

        # group(5) 是 [Speaker XX] 捕获内容，group(6) 是正文
        speaker = match.group(5) or ""
        content = (match.group(6) or "").strip()
        if content:
            segments.append((timestamp_str, speaker, content))

    return segments


def _to_int16_samples(audio: AudioSegment):
    """把 AudioSegment 转成 int16 numpy 数组（统一采样率/声道/位深）"""
    audio = (audio.set_frame_rate(SR)
             .set_channels(N_CH)
             .set_sample_width(WIDTH))
    return np.frombuffer(audio.raw_data, dtype=np.int16)


def adjust_audio_speed(input_path, output_path, target_duration_ms, speed_factor):
    """用 ffmpeg atempo + lowpass 调整音频速度，防止重叠时产生刺耳尖啸。

    返回 True 成功 / False 失败。
    """
    factor = max(0.5, min(speed_factor, 2.0))
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', input_path,
             '-filter:a', f'lowpass=f=8000,atempo={factor:.4f}',
             output_path],
            capture_output=True, text=True, timeout=120
        )
        return result.returncode == 0
    except Exception as e:
        print(f"变速失败 {input_path}: {e}")
        return False


def fast_overlay(audio_entries):
    """numpy 共享内存一次性混音。

    audio_entries: [(start_ms, AudioSegment), ...]  已按 start_ms 排序
    返回混音后的 AudioSegment
    """
    if not audio_entries:
        return AudioSegment.silent(duration=0, frame_rate=SR)

    # 计算总长度
    last_start, last_audio = audio_entries[-1]
    total_ms = last_start + len(last_audio) + 1000  # 留 1s 尾巴
    total_samples = int(total_ms * SR / 1000)

    # 共享内存混音板
    shm = shared_memory.SharedMemory(create=True, size=total_samples * N_CH * 4)
    try:
        buf = np.ndarray((total_samples * N_CH,), dtype=np.float32, buffer=shm.buf)
        buf[:] = 0.0

        for start_ms, audio in audio_entries:
            samples = _to_int16_samples(audio).astype(np.float32)
            start_sample = int(start_ms * SR / 1000)
            end_sample = start_sample + len(samples)
            if end_sample > total_samples:
                end_sample = total_samples
                samples = samples[:end_sample - start_sample]
            buf[start_sample:end_sample] += samples

        np.clip(buf, -MAX_INT, MAX_INT, out=buf)
        out_bytes = buf.astype(np.int16).tobytes()
    finally:
        shm.close()
        shm.unlink()

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
                target = next_start - start_ms - 50
                if target > 100:
                    factor = min(len(audio) / target, 2.0)
                    tmp_out = path + '.speed.wav'
                    tmp_in = path
                    # 若原文件非 wav，先转 wav 以便 ffmpeg 处理
                    if not path.endswith('.wav'):
                        tmp_in = path + '.in.wav'
                        audio.export(tmp_in, format='wav')
                        temp_files.append(tmp_in)
                    if adjust_audio_speed(tmp_in, tmp_out, target, factor):
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
