#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""moss-tts 语音克隆配音（核心创新模块）。

流程:
  解析翻译字幕 (timestamp + speaker + text)
  → 按说话人匹配参考音
  → 单进程加载 moss-tts ONNX 模型，循环推理所有句子（切换参考音）
  → audio_utils 变速防重叠 + numpy 共享内存混音
  → 输出 dubbed_audio.mp3

输入字幕格式: (HH:MM:SS.mmm) [Speaker 00] 中文译文
"""

import os
import sys
import re
import json
import time
import argparse
import traceback

# 同目录导入 audio_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_utils import split_text_by_timestamp, parse_timestamp, mix_segments


def load_speaker_refs(ref_map_path):
    """加载说话人→参考音映射。"""
    with open(ref_map_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_subtitle(file_path):
    """解析翻译字幕，返回 [(timestamp_str, speaker, text), ...]。"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return split_text_by_timestamp(content)


def _patch_torchaudio_for_cpu():
    """monkey-patch torchaudio.load 用 soundfile 后端，绕开 torchcodec 的 CUDA 依赖。

    torchaudio 2.5+ 默认走 torchcodec 后端，在纯 CPU 环境（无 libnvrtc.so）
    会报 OSError。soundfile + libsndfile 是纯 CPU 的稳定方案。
    """
    import torchaudio
    import soundfile as sf
    import torch
    import warnings

    if getattr(torchaudio, "_voxclone_patched", False):
        return

    def _load_with_soundfile(filepath, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data, sr = sf.read(str(filepath), always_2d=True, dtype="float32")
        # data shape: (samples, channels) → tensor (channels, samples)
        waveform = torch.from_numpy(data).T
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return waveform, sr

    torchaudio.load = _load_with_soundfile
    torchaudio._voxclone_patched = True
    print("[patch] torchaudio.load 已替换为 soundfile 后端（绕开 torchcodec）")


def build_moss_runtime(moss_tts_dir):
    """加载 moss-tts ONNX runtime（只加载一次）。"""
    # 在 import moss-tts 之前 patch，确保 moss-tts 调用 torchaudio.load 时走 soundfile
    _patch_torchaudio_for_cpu()

    if moss_tts_dir and os.path.isdir(moss_tts_dir):
        sys.path.insert(0, os.path.abspath(moss_tts_dir))
    try:
        from onnx_tts_runtime import OnnxTtsRuntime
    except ImportError:
        # editable 安装后模块名可能不同，尝试备用导入
        try:
            from moss_tts.onnx_tts_runtime import OnnxTtsRuntime
        except ImportError as e:
            raise ImportError(
                f"无法导入 OnnxTtsRuntime，请确认 moss-tts 已安装。"
                f"moss_tts_dir={moss_tts_dir}。错误: {e}"
            )

    print("加载 moss-tts ONNX runtime (CPU)...")
    runtime = OnnxTtsRuntime(
        model_dir=None,          # None → 使用默认目录，缺失时自动下载
        thread_count=4,
        max_new_frames=375,
        do_sample=True,
        sample_mode="fixed",
        execution_provider="cpu",
    )
    return runtime


def load_progress(progress_path):
    """加载进度文件，返回 dict {total, completed(set), skipped(set)}。

    文件不存在或已损坏时返回空进度（重建）。
    """
    if not os.path.exists(progress_path):
        return {"total": 0, "completed": set(), "skipped": set()}
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "total": data.get("total", 0),
            "completed": set(data.get("completed", [])),
            "skipped": set(data.get("skipped", [])),
        }
    except (json.JSONDecodeError, OSError):
        print(f"[warn] 进度文件损坏，重建: {progress_path}")
        return {"total": 0, "completed": set(), "skipped": set()}


def save_progress(progress, progress_path):
    """原子写入进度文件（先写 .tmp 再 rename，避免写到一半被杀导致损坏）。"""
    tmp = progress_path + ".tmp"
    payload = {
        "total": progress["total"],
        "completed": sorted(progress["completed"]),
        "skipped": sorted(progress["skipped"]),
    }
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, progress_path)


def scan_existing_segments(segments_dir, total):
    """扫描 segments 目录，根据 seg_XXXXX.wav 文件重建 completed 集合。

    用于 resume 时校验文件实际存在（防止进度文件与磁盘不一致，
    例如上一轮进度文件丢失但 wav 还在）。
    """
    completed = set()
    if not os.path.isdir(segments_dir):
        return completed
    for fn in os.listdir(segments_dir):
        m = re.match(r"^seg_(\d+)\.wav$", fn)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < total:
                completed.add(idx)
    return completed


def clone_dub_segments(segments, speaker_ref_map, runtime, output_dir,
                       time_budget_seconds=0, resume=False,
                       progress_path=None, progress=None):
    """单进程循环推理所有句子，按说话人切换参考音。

    支持两类中断续作：
      - time_budget_seconds > 0：每段开始前检查已用时长，超预算则保存进度
        并返回 (results, timed_out=True)，由调用方决定是否续作。
      - resume=True：跳过进度中已 completed/skipped 的段，并把已存在于
        磁盘的 wav 重新计入 results，实现断点续作。

    segments: [(timestamp_str, speaker, text), ...]
    speaker_ref_map: {"Speaker 00": "ref.wav", ...}
    返回: (results, timed_out)
      results: [(start_ms, output_wav_path), ...]
      timed_out: bool，True 表示因达到时间预算而提前返回
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    fallback_ref = next(iter(speaker_ref_map.values()), None)

    total = len(segments)
    if progress is None:
        progress = {"total": total, "completed": set(), "skipped": set()}
    else:
        progress["total"] = total

    # resume 模式：扫描磁盘上已存在的 wav，校准 completed（防止进度丢失）
    if resume:
        existing = scan_existing_segments(output_dir, total)
        recovered = existing - progress["completed"]
        if recovered:
            print(f"[resume] 从磁盘恢复 {len(recovered)} 个已完成段")
        progress["completed"] |= existing

    need_save = progress_path is not None
    loop_start = time.time()

    for i, (ts_str, speaker, text) in enumerate(segments):
        # 续作：跳过已完成/已跳过的段
        if resume and (i in progress["completed"] or i in progress["skipped"]):
            if i in progress["completed"]:
                out_wav = os.path.join(output_dir, f"seg_{i:05d}.wav")
                results.append((parse_timestamp(f"({ts_str})"), out_wav))
            continue

        # 时间预算检查：每段开始前检查，留足收尾与上传时间
        if time_budget_seconds and time_budget_seconds > 0:
            elapsed = time.time() - loop_start
            if elapsed >= time_budget_seconds:
                print(f"\n=== 已达时间预算 ({elapsed:.0f}s >= {time_budget_seconds}s)，"
                      f"暂停保存进度 ===")
                if need_save:
                    save_progress(progress, progress_path)
                return results, True  # timed_out=True

        start_ms = parse_timestamp(f"({ts_str})")
        out_wav = os.path.join(output_dir, f"seg_{i:05d}.wav")

        # 匹配参考音
        ref_path = speaker_ref_map.get(speaker) or fallback_ref
        if not ref_path or not os.path.exists(ref_path):
            print(f"[{i+1}/{total}] 无参考音 for {speaker}，跳过: {text[:30]}")
            progress["skipped"].add(i)
            if need_save:
                save_progress(progress, progress_path)
            continue

        print(f"[{i+1}/{total}] {ts_str} [{speaker}] {text[:40]}...")
        try:
            result = runtime.synthesize(
                text=text,
                voice="Junhao",
                prompt_audio_path=ref_path,
                output_audio_path=out_wav,
                sample_mode="fixed",
                do_sample=True,
                streaming=True,
                max_new_frames=375,
                voice_clone_max_text_tokens=75,
                enable_wetext=True,
                enable_normalize_tts_text=True,
            )
            saved = result.get("audio_path", out_wav)
            if os.path.exists(saved):
                results.append((start_ms, saved))
                progress["completed"].add(i)
            else:
                print(f"  警告: 输出文件不存在 {saved}")
                progress["skipped"].add(i)
        except Exception as e:
            print(f"  克隆失败: {e}")
            traceback.print_exc()
            progress["skipped"].add(i)

        # 每段完成即保存进度（开销极小，确保任意时刻被杀都能续作）
        if need_save:
            save_progress(progress, progress_path)

    if need_save:
        save_progress(progress, progress_path)
    return results, False


def main():
    parser = argparse.ArgumentParser(description="moss-tts 语音克隆配音")
    parser.add_argument("--input", required=True, help="翻译字幕文件")
    parser.add_argument("--output", required=True, help="输出配音 mp3 路径")
    parser.add_argument("--ref_map", required=True, help="说话人参考音映射 JSON")
    parser.add_argument("--moss_tts_dir", default="", help="MOSS-TTS-Nano 目录路径")
    parser.add_argument("--work_dir", default="work_tts", help="临时工作目录（含 segments 与进度）")
    parser.add_argument("--time_budget_seconds", type=int, default=0,
                        help="TTS 循环时间预算（秒）。>0 时达到预算即暂停保存进度并退出（退出码 0），"
                             "由 workflow 决定是否开启续作 job。0=不限制（与原行为一致）")
    parser.add_argument("--resume", action="store_true",
                        help="续作模式：加载进度并跳过已完成/已跳过的段，仅处理剩余部分")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    seg_dir = os.path.join(args.work_dir, "segments")
    progress_path = os.path.join(args.work_dir, "_progress.json")
    need_resume_marker = os.path.join(args.work_dir, "NEED_RESUME")

    # 1. 解析字幕
    segments = parse_subtitle(args.input)
    if not segments:
        print("错误: 未解析到任何字幕段落")
        return 1
    print(f"解析到 {len(segments)} 条字幕")

    # 2. 加载参考音映射
    speaker_ref_map = load_speaker_refs(args.ref_map)
    print(f"说话人参考音: {speaker_ref_map}")

    # 3. 加载 moss-tts 模型
    runtime = build_moss_runtime(args.moss_tts_dir)

    # 4. 逐句克隆推理（带时间预算与续作）
    progress = load_progress(progress_path) if args.resume else None
    audio_entries, timed_out = clone_dub_segments(
        segments, speaker_ref_map, runtime, seg_dir,
        time_budget_seconds=args.time_budget_seconds,
        resume=args.resume,
        progress_path=progress_path,
        progress=progress,
    )
    print(f"成功生成 {len(audio_entries)} 条配音")

    if timed_out:
        # 超时是预期行为（非失败）：进度已保存，写标记文件，正常退出（码 0）
        # 由 workflow 的 NEED_RESUME 检测决定是否启动续作 job
        with open(need_resume_marker, "w", encoding="utf-8") as f:
            f.write("1")
        done = len(progress["completed"]) if progress else 0
        print(f"\n=== TTS 超时，已完成 {done}/{len(segments)} 段，需续作 ===")
        print(f"进度已保存: {progress_path}")
        return 0

    if not audio_entries:
        print("错误: 没有成功生成任何配音")
        return 1

    # 按起始时间排序
    audio_entries.sort(key=lambda x: x[0])

    # 5. 变速防重叠 + 混音
    print("开始混音（含变速防重叠）...")
    final_audio = mix_segments(audio_entries)

    # 6. 导出 mp3
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    final_audio.export(args.output, format="mp3")
    print(f"\n=== 配音完成 ===")
    print(f"输出: {args.output}")
    print(f"时长: {len(final_audio) / 1000:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
