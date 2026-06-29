#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视频下载 + 音轨替换 + B站上传。

从 actionsautopython/youtube_to_bilibili.py 适配:
  download_youtube_video (多格式回退)
  replace_audio_track (ffmpeg -c:v copy -c:a aac)
  download_thumbnail / convert_and_compress_to_jpeg
  generate_upload_config (LLM 翻译标题 + 生成标签)
  upload_to_bilibili (bilibili-api-python, 七牛线路, 6次重试)
"""

import os
import sys
import glob
import json
import time
import pickle
import argparse
import subprocess
from typing import Dict, Any

import requests
from PIL import Image
from bilibili_api import sync, video_uploader, Credential


def load_api_config(config_file):
    """加载 API 配置，回退环境变量。"""
    config = {}
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f) or {}
        except Exception as e:
            print(f"读取配置文件出错: {e}")
    return {
        "url": config.get("url") or os.environ.get("AI_API_URL", ""),
        "api_key": config.get("api_key") or os.environ.get("AI_API_KEY", ""),
        "model_name": config.get("model_name") or os.environ.get("AI_MODEL", "THUDM/GLM-4-9B-0414"),
    }


def download_youtube_video(url, output_path, cookies_file=None):
    """下载 YouTube 视频（仅视频流），多格式选择器回退。"""
    format_selectors = [
        "bestvideo[height<=1080]",
        "bestvideo[height<=720]",
        "bestvideo",
        "best[height<=1080]",
        "best[height<=720]",
        "best",
    ]
    for fmt in format_selectors:
        print(f"尝试格式: {fmt}")
        cmd = [
            "yt-dlp",
            "--extractor-args", "youtube:player_client=default,-web_safari",
            "--remote-components", "ejs:github",
            "--no-playlist",
            "-f", fmt,
            "-o", output_path,
            url,
        ]
        if cookies_file and os.path.exists(cookies_file):
            cmd.insert(1, "--cookies")
            cmd.insert(2, cookies_file)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
            if result.returncode == 0:
                print(f"视频下载成功: {output_path}")
                return True
            print(f"格式 {fmt} 失败: {result.stderr[:200]}")
            if "Requested format is not available" in result.stderr:
                continue
            return False
        except Exception as e:
            print(f"下载出错: {e}")
            return False
    print("所有格式选择器均失败")
    return False


def replace_audio_track(video_path, audio_path, output_path):
    """ffmpeg 替换音轨: -c:v copy -c:a aac。"""
    if not os.path.exists(audio_path):
        print(f"错误: 找不到音频 {audio_path}")
        return False
    print(f"替换音轨: {video_path} + {audio_path} → {output_path}")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
            output_path
        ], check=True)
        print(f"音轨替换完成: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg 错误: {e}")
        return False
    except FileNotFoundError:
        print("错误: 未找到 ffmpeg")
        return False


def download_thumbnail(url, output_tmpl, cookies_file=None):
    """下载视频缩略图。"""
    cmd = [
        "yt-dlp",
        "--extractor-args", "youtube:player_client=default,-web_safari",
        "--remote-components", "ejs:github",
        "--no-playlist", "--skip-download", "--write-thumbnail",
        "-o", output_tmpl,
        url,
    ]
    if cookies_file and os.path.exists(cookies_file):
        cmd.insert(1, "--cookies")
        cmd.insert(2, cookies_file)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        return result.returncode == 0
    except Exception as e:
        print(f"缩略图下载出错: {e}")
        return False


def convert_and_compress_to_jpeg(input_path, output_path, target_size_kb=50):
    """转 JPEG 并压缩到目标大小。"""
    if not os.path.exists(input_path):
        print(f"错误: 找不到 {input_path}")
        return False
    try:
        with Image.open(input_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            quality = 90
            img.save(output_path, "jpeg", quality=quality)
            size_kb = os.path.getsize(output_path) / 1024
            while size_kb > target_size_kb and quality > 4:
                quality -= 5
                img.save(output_path, "jpeg", quality=quality)
                size_kb = os.path.getsize(output_path) / 1024
            print(f"封面压缩完成: {output_path} ({size_kb:.1f} KB)")
            return True
    except Exception as e:
        print(f"压缩出错: {e}")
        return False


def get_video_title(url, cookies_file=None):
    """用 yt-dlp 获取视频标题。"""
    cmd = [
        "yt-dlp",
        "--extractor-args", "youtube:player_client=default,-web_safari",
        "--remote-components", "ejs:github",
        "--no-playlist", "--dump-json", "--skip-download",
        url,
    ]
    if cookies_file and os.path.exists(cookies_file):
        cmd.insert(1, "--cookies")
        cmd.insert(2, cookies_file)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        if result.returncode == 0:
            return json.loads(result.stdout).get("title")
    except Exception as e:
        print(f"获取标题出错: {e}")
    return None


def llm_translate_title(title, api_config):
    """LLM 翻译标题为中文。"""
    if not api_config.get("url") or not api_config.get("api_key"):
        print("警告: 未配置 AI API，使用原始标题")
        return title
    system_prompt = "# role\n爆款视频up主\n\n## 任务\n将英文标题翻译成吸引眼球的爆款视频中文标题。\n\n## 输出格式\n直接给出翻译后的中文标题，不要其他文字"
    payload = {
        "model": api_config["model_name"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": title},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_config['api_key']}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(api_config["url"], json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        translated = resp.json()["choices"][0]["message"]["content"].replace("**", "").strip()
        return translated or title
    except Exception as e:
        print(f"翻译标题出错: {e}，使用原始标题")
        return title


def llm_generate_tags(title, api_config):
    """LLM 根据标题生成标签。"""
    if not api_config.get("url") or not api_config.get("api_key"):
        return ["科普"]
    system_prompt = "# role\n视频内容标签专家\n\n## 任务\n根据标题生成5个左右中文关键词标签。\n\n## 输出格式\n逗号分隔的关键词列表，例如：关键词1,关键词2,关键词3"
    payload = {
        "model": api_config["model_name"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": title},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_config['api_key']}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(api_config["url"], json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        tags_raw = resp.json()["choices"][0]["message"]["content"].replace("**", "").strip()
        tags_list = [t.strip() for t in tags_raw.replace("，", ",").split(",") if t.strip()]
        return tags_list or ["科普"]
    except Exception as e:
        print(f"生成标签出错: {e}")
        return ["科普"]


def generate_upload_config(url, api_config, cookies_file=None):
    """生成上传配置（翻译标题 + 标签）。"""
    title = get_video_title(url, cookies_file)
    if not title:
        print("错误: 无法获取视频标题")
        return {}
    print(f"原始标题: {title}")
    translated = llm_translate_title(title, api_config)
    tags = llm_generate_tags(translated, api_config)
    print(f"翻译标题: {translated}")
    print(f"标签: {tags}")
    return {
        "title_desc": "(中配)" + translated,
        "tags": tags,
    }


async def upload_to_bilibili(video_path, cover_path, config, credential, max_retries=6):
    """上传到 B站（七牛线路，指数退避重试）。"""
    title_desc = config.get("title_desc", "默认标题")
    tags = config.get("tags", ["默认标签"])

    vu_meta = video_uploader.VideoMeta(
        tid=130, title=title_desc, tags=tags,
        desc=title_desc, cover=cover_path, no_reprint=True,
    )
    page = video_uploader.VideoUploaderPage(
        path=video_path, title=title_desc, description=title_desc,
    )
    uploader = video_uploader.VideoUploader(
        [page], vu_meta, credential, line=video_uploader.Lines.QN
    )

    @uploader.on("__ALL__")
    async def ev(data):
        print(data)

    for attempt in range(max_retries):
        try:
            await uploader.start()
            print("视频上传成功")
            return True
        except Exception as e:
            print(f"上传错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                delay = 10 * (2 ** attempt)
                print(f"等待 {delay}s 后重试...")
                time.sleep(delay)
    print("达到最大重试次数，上传失败")
    return False


def main():
    parser = argparse.ArgumentParser(description="视频下载 + 音轨替换 + B站上传")
    parser.add_argument("--url", required=True, help="YouTube 视频 URL")
    parser.add_argument("--audio_path", required=True, help="配音音频 mp3 路径")
    parser.add_argument("--api_config", default="config/api_config.json", help="API 配置文件")
    parser.add_argument("--cookies", help="YouTube cookies 文件路径")
    parser.add_argument("--work_dir", default="work_upload", help="工作目录")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    os.chdir(args.work_dir)

    if not os.path.exists(args.audio_path):
        print(f"错误: 找不到配音音频 {args.audio_path}")
        return 1

    api_config = load_api_config(args.api_config)
    cookies_file = args.cookies if (args.cookies and os.path.exists(args.cookies)) else None

    # 1. 生成上传配置（翻译标题 + 标签）
    upload_config = generate_upload_config(args.url, api_config, cookies_file)
    if not upload_config:
        print("生成上传配置失败")
        return 1

    # 2. 下载视频
    downloaded = "downloaded_video.mp4"
    if not download_youtube_video(args.url, downloaded, cookies_file):
        print("视频下载失败")
        return 1
    # 查找实际下载的文件
    videos = glob.glob("downloaded_video.*")
    if not videos:
        print("未找到下载的视频文件")
        return 1
    video_path = videos[0]

    # 3. 替换音轨
    final_video = "final_video.mp4"
    if not replace_audio_track(video_path, args.audio_path, final_video):
        print("音轨替换失败")
        return 1

    # 4. 下载并压缩封面
    download_thumbnail(args.url, "cover.%(ext)s", cookies_file)
    covers = glob.glob("cover.*")
    cover_jpeg = "cover.jpeg"
    if covers:
        if not convert_and_compress_to_jpeg(covers[0], cover_jpeg):
            cover_jpeg = None
    else:
        cover_jpeg = None

    # 5. 上传 B站
    sessdata = os.environ.get("BILIBILI_SESSDATA", "")
    bili_jct = os.environ.get("BILIBILI_JCT", "")
    buvid3 = os.environ.get("BILIBILI_BUVID3", "")
    if not sessdata or not bili_jct:
        print("错误: 未设置 BILIBILI_SESSDATA 或 BILIBILI_JCT")
        return 1

    credential = Credential(sessdata=sessdata, bili_jct=bili_jct, buvid3=buvid3)
    ok = sync(upload_to_bilibili(
        final_video, cover_jpeg, upload_config, credential
    ))
    if ok:
        print("\n=== 整个流程成功完成 ===")
        return 0
    else:
        print("\nB站上传失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
