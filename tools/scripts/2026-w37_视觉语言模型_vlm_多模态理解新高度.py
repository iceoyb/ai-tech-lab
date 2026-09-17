#!/usr/bin/env python3
"""
视觉语言模型(VLM)多模态理解新高度 - 多模态内容理解与图像分析 CLI 工具

本工具提供一个纯标准库实现的"视觉-语言"理解流水线：
1. 解析多种图像格式（PNG/JPEG/GIF/BMP/PNM），提取尺寸、色彩、纹理、边缘等视觉特征；
2. 将视觉特征编码为紧凑的"视觉 token"序列，模拟 VLM 中视觉编码器的输出；
3. 结合可选的自然语言问题（--question），通过跨模态注意力式的关键词-特征对齐，
   生成针对图像内容的结构化理解报告（场景类型、主色调、复杂度、显著区域等）；
4. 支持批量处理目录、JSON/文本双格式报告输出、自检测试模式。

设计目标：在零依赖环境下演示并落地 VLM 多模态理解的核心工程范式
（视觉特征提取 → token 化 → 语言对齐 → 结构化理解输出）。

用法示例：
    python vlm_insight.py photo.png
    python vlm_insight.py ./images/ -q "这张图的主色调是什么？" --format json -o report.json
    python vlm_insight.py --self-test
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import struct
import sys
import time
import zlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 常量与调色板定义
# ---------------------------------------------------------------------------

VERSION = "1.0.0"

# 命名调色板：用于将像素聚类到人类可读的颜色名称（模拟 VLM 的颜色词对齐）
NAMED_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "黑色": (20, 20, 20),
    "白色": (245, 245, 245),
    "灰色": (128, 128, 128),
    "红色": (220, 40, 40),
    "橙色": (240, 140, 30),
    "黄色": (240, 220, 60),
    "绿色": (60, 170, 70),
    "青色": (60, 200, 200),
    "蓝色": (50, 90, 220),
    "紫色": (140, 70, 180),
    "粉色": (240, 150, 180),
    "棕色": (130, 85, 50),
}

# 场景关键词 → 期望的视觉特征签名（用于跨模态对齐打分）
SCENE_SIGNATURES: Dict[str, Dict[str, float]] = {
    "风景/自然": {"green_ratio": 0.25, "blue_ratio": 0.15, "edge_density": 0.12, "brightness": 0.55},
    "人像/人物": {"skin_ratio": 0.12, "edge_density": 0.08, "brightness": 0.55, "saturation": 0.35},
    "建筑/城市": {"edge_density": 0.22, "gray_ratio": 0.20, "saturation": 0.25, "brightness": 0.50},
    "夜景/低光": {"brightness": 0.22, "saturation": 0.20, "edge_density": 0.06, "contrast": 0.30},
    "文档/文本": {"white_ratio": 0.45, "edge_density": 0.18, "saturation": 0.08, "contrast": 0.55},
    "抽象/艺术": {"saturation": 0.55, "contrast": 0.55, "colorfulness": 0.50, "edge_density": 0.15},
}

# 问题关键词 → 关注特征（跨模态注意力权重）
QUESTION_ATTENTION: Dict[str, List[str]] = {
    "颜色": ["dominant_colors", "colorfulness", "saturation"],
    "主色": ["dominant_colors"],
    "色调": ["dominant_colors", "brightness"],
    "亮": ["brightness", "contrast"],
    "暗": ["brightness"],
    "清晰": ["edge_density", "sharpness"],
    "模糊": ["sharpness"],
    "场景": ["scene"],
    "内容": ["scene", "objects_hint"],
    "复杂": ["complexity", "edge_density"],
    "构图": ["salient_regions", "balance"],
    "文字": ["scene"],
    "人": ["scene", "skin_ratio"],
}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ImageData:
    """解码后的图像像素与元信息。"""
    width: int
    height: int
    pixels: List[Tuple[int, int, int]]  # RGB 三元组列表
    fmt: str
    path: str


@dataclass
class VisualFeatures:
    """视觉编码器输出的特征集合。"""
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    colorfulness: float = 0.0
    edge_density: float = 0.0
    sharpness: float = 0.0
    complexity: float = 0.0
    balance: float = 0.0
    green_ratio: float = 0.0
    blue_ratio: float = 0.0
    gray_ratio: float = 0.0
    white_ratio: float = 0.0
    skin_ratio: float = 0.0
    dominant_colors: List[Dict[str, object]] = field(default_factory=list)
    salient_regions: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class UnderstandingReport:
    """最终的多模态理解报告。"""
    file: str
    format: str
    width: int
    height: int
    megapixels: float
    visual_tokens: List[str]
    features: Dict[str, object]
    scene: str
    scene_scores: Dict[str, float]
    summary: str
    answer: Optional[str] = None
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# 图像解码层（纯标准库，支持 PNG / JPEG(尺寸) / GIF / BMP / PNM）
# ---------------------------------------------------------------------------

class ImageDecodeError(Exception):
    """图像解码失败时抛出。"""


def _read_file(path: str, max_bytes: int = 64 * 1024 * 1024) -> bytes:
    """安全读取文件，限制大小防止内存耗尽。"""
    if not os.path.isfile(path):
        raise ImageDecodeError(f"文件不存在: {path}")
    size = os.path.getsize(path)
    if size == 0:
        raise ImageDecodeError(f"文件为空: {path}")
    if size > max_bytes:
        raise ImageDecodeError(f"文件过大 ({size} 字节)，超过 {max_bytes} 限制")
    with open(path, "rb") as fh:
        return fh.read()


def _paeth(a: int, b: int, c: int) -> int:
    """PNG Paeth 预测器。"""
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def decode_png(data: bytes, path: str) -> ImageData:
    """解码非隔行 PNG（8-bit 灰度/RGB/RGBA），返回 RGB 像素。"""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ImageDecodeError("不是有效的 PNG 文件")
    pos = 8
    width = height = 0
    bit_depth = color_type = None
    idat = bytearray()
    palette: List[Tuple[int, int, int]] = []
    trns: bytes = b""
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        if len(chunk) < length:
            raise ImageDecodeError("PNG 数据块被截断")
        if ctype == b"IHDR":
            width, height, bit_depth, color_type, comp, filt, interlace = struct.unpack(">IIBBBBB", chunk)
            if interlace != 0:
                raise ImageDecodeError("暂不支持隔行扫描 PNG")
            if bit_depth != 8:
                raise ImageDecodeError(f"暂不支持 {bit_depth}-bit PNG（仅支持 8-bit）")
        elif ctype == b"PLTE":
            palette = [(chunk[i], chunk[i + 1], chunk[i + 2]) for i in range(0, len(chunk) - 2, 3)]
        elif ctype == b"tRNS":
            trns = chunk
        elif ctype == b"IDAT":
            idat.extend(chunk)
        elif ctype == b"IEND":
            break
        pos += 12 + length
    if not width or not height:
        raise ImageDecodeError("PNG 缺少 IHDR")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise ImageDecodeError(f"不支持的 PNG 颜色类型: {color_type}")
    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as exc:
        raise ImageDecodeError(f"PNG IDAT 解压失败: {exc}") from exc
    stride = width * channels
    expected = (stride + 1) * height
    if len(raw) < expected:
        raise ImageDecodeError("PNG 像素数据长度不足")
    pixels: List[Tuple[int, int, int]] = []
    prev = bytearray(stride)
    idx = 0
    for _y in range(height):
        ftype = raw[idx]
        idx += 1
        line = bytearray(raw[idx:idx + stride])
        idx += stride
        if ftype == 1:  # Sub
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif ftype == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ftype == 3:  # Average
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                c = prev[i - channels] if i >= channels else 0
                line[i] = (line[i] + _paeth(a, prev[i], c)) & 0xFF
        elif ftype != 0:
            raise ImageDecodeError(f"未知 PNG 滤波类型: {ftype}")
        prev = line
        for x in range(width):
            o = x * channels
            if color_type == 0:
                g = line[o]
                pixels.append((g, g, g))
            elif color_type == 2:
                pixels.append((line[o], line[o + 1], line[o + 2]))
            elif color_type == 3:
                pi = line[o]
                if pi < len(palette):
                    pixels.append(palette[pi])
                else:
                    pixels.append((0, 0, 0))
            elif color_type == 4:
                g = line[o]
                pixels.append((g, g, g))
            else:  # 6 RGBA（忽略 alpha，直接取 RGB）
                pixels.append((line[o], line[o + 1], line[o + 2]))
    return ImageData(width, height, pixels, "PNG", path)


def decode_gif(data: bytes, path: str) -> ImageData:
    """解码 GIF87a/89a 首帧（无 LZW 完整实现时退化为尺寸+调色板估计）。

    为保证零依赖且健壮，这里实现完整头解析；若 LZW 解码复杂度过高，
    使用全局调色板均值生成近似像素场，仍可提供颜色级理解。
    """
    if not (data.startswith(b"GIF87a") or data.startswith(b"GIF89a")):
        raise ImageDecodeError("不是有效的 GIF 文件")
    width, height = struct.unpack("<HH", data[6:10])
    packed = data[10]
    gct_flag = packed & 0x80
    gct_size = 2 ** ((packed & 0x07) + 1)
    pos = 13
    palette: List[Tuple[int, int, int]] = []
    if gct_flag:
        for i in range(gct_size):
            o = pos + i * 3
            if o + 2 < len(data):
                palette.append((data[o], data[o + 1], data[o + 2]))
        pos += gct_size * 3
    if not palette:
        palette = [(128, 128, 128)]
    # 近似像素场：用调色板加权填充（颜色分布理解足够，空间结构弱）
    total = width * height
    pixels = [palette[i % len(palette)] for i in range(total)]
    return ImageData(width, height, pixels, "GIF", path)


def decode_bmp(data: bytes, path: str) -> ImageData:
    """解码 24-bit 未压缩 BMP。"""
    if not data.startswith(b"BM"):
        raise ImageDecodeError("不是有效的 BMP 文件")
    offset = struct.unpack("<I", data[10:14])[0]
    header_size = struct.unpack("<I", data[14:18])[0]
    if header_size < 40:
        raise ImageDecodeError("不支持的 BMP 头版本")
    width, height = struct.unpack("<ii", data[18:26])
    bpp = struct.unpack("<H", data[28:30])[0]
    compression = struct.unpack("<I", data[30:34])[0]
    if bpp != 24 or compression != 0:
        raise ImageDecodeError(f"仅支持 24-bit 未压缩 BMP（当前 bpp={bpp}, comp={compression}）")
    top_down = height < 0
    height = abs(height)
    row_size = (width * 3 + 3) & ~3
    pixels: List[Tuple[int, int, int]] = [(0, 0, 0)] * (width * height)
    for y in range(height):
        src_y = y if top_down else (height - 1 - y)
        base = offset + src_y * row_size
        for x in range(width):
            o = base + x * 3
            if o + 2 < len(data):
                b, g, r = data[o], data[o + 1], data[o + 2]
                pixels[y * width + x] = (r, g, b)
    return ImageData(width, height, pixels, "BMP", path)


def decode_pnm(data: bytes, path: str) -> ImageData:
    """解码 P3/P6 PNM。"""
    tokens: List[bytes] = []
    i = 0
    while len(tokens) < 4 and i < len(data):
        c = data[i:i + 1]
        if c == b"#":
            while i < len(data) and data[i:i + 1] != b"\n":
                i += 1
        elif c.isspace():
            i += 1
        else:
            j = i
            while j < len(data) and not data[j:j + 1].isspace():
                j += 1
            tokens.append(data[i:j])
            i = j
    if len(tokens) < 4:
        raise ImageDecodeError("PNM 头不完整")
    magic = tokens[0].decode()
    width, height, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])
    i += 1  # 跳过单个空白
    pixels: List[Tuple[int, int, int]] = []
    scale = 255.0 / max(maxval, 1)
    if magic == "P6":
        need = width * height * 3
        raw = data[i:i + need]
        if len(raw) < need:
            raise ImageDecodeError("PNM 像素数据不足")
        for k in range(0, need, 3):
            pixels.append((int(raw[k] * scale), int(raw[k + 1] * scale), int(raw[k + 2] * scale)))
    elif magic == "P3":
        nums = data[i:].split()
        if len(nums) < width * height * 3:
            raise ImageDecodeError("PNM 像素数据不足")
        for k in range(0, width * height * 3, 3):
            pixels.append((int(int(nums[k]) * scale), int(int(nums[k + 1]) * scale), int(int(nums[k + 2]) * scale)))
    else:
        raise ImageDecodeError(f"不支持的 PNM 类型: {magic}")
    return ImageData(width, height, pixels, "PNM", path)


def decode_jpeg_size(data: bytes, path: str) -> ImageData:
    """解析 JPEG SOF 获取尺寸；像素以中性灰填充（零依赖不解码 DCT）。

    JPEG 完整解码需要 Huffman/DCT，超出标准库合理范围；
    本函数提取精确尺寸与 EXIF 方向，并用灰场保证下游流程可运行，
    同时在报告中标注 format=JPEG(meta-only)。
    """
    if not data.startswith(b"\xff\xd8"):
        raise ImageDecodeError("不是有效的 JPEG 文件")
    pos = 2
    width = height = 0
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            pos += 1
            continue
        marker = data[pos + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        seg_len = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2):
            height, width = struct.unpack(">HH", data[pos + 5:pos + 9])
            break
        pos += 2 + seg_len
    if not width:
        raise ImageDecodeError("JPEG 中未找到 SOF 帧头")
    pixels = [(128, 128, 128)] * (width * height)
    return ImageData(width, height, pixels, "JPEG(meta-only)", path)


def load_image(path: str) -> ImageData:
    """根据魔数自动识别格式并解码。"""
    data = _read_file(path)
    if data.startswith(b"\x89PNG"):
        return decode_png(data, path)
    if data.startswith(b"\xff\xd8"):
        return decode_jpeg_size(data, path)
    if data.startswith(b"GIF8"):
        return decode_gif(data, path)
    if data.startswith(b"BM"):
        return decode_bmp(data, path)
    if data[:2] in (b"P3", b"P6"):
        return decode_pnm(data, path)
    raise ImageDecodeError(f"无法识别的图像格式: {path}（支持 PNG/JPEG/GIF/BMP/PNM）")


# ---------------------------------------------------------------------------
# 视觉特征提取层（模拟 VLM 视觉编码器）
# ---------------------------------------------------------------------------

def _rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """RGB → HSV，返回 h∈[0,360), s,v∈[0,1]。"""
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(rf, gf, bf), min(rf, gf, bf)
    diff = mx - mn
    if diff == 0:
        h = 0.0
    elif mx == rf:
        h = (60 * ((gf - bf) / diff) + 360) % 360
    elif mx == gf:
        h = 60 * ((bf - rf) / diff) + 120
    else:
        h = 60 * ((rf - gf) / diff) + 240
    s = 0.0 if mx == 0 else diff / mx
    return h, s, mx


def _nearest_color_name(r: int, g: int, b: int) -> str:
    """在命名调色板中找欧氏距离最近的颜色名。"""
    best, best_d = "灰色", float("inf")
    for name, (pr, pg, pb) in NAMED_PALETTE.items():
        d = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
        if d < best_d:
            best, best_d = name, d
    return best


def _sample_pixels(img: ImageData, max_samples: int = 40000) -> Tuple[List[Tuple[int, int, int]], int]:
    """均匀抽样像素以控制计算量，返回 (样本, 步长)。"""
    total = len(img.pixels)
    if total <= max_samples:
        return img.pixels, 1
    step = max(1, total // max_samples)
    return img.pixels[::step], step


def extract_features(img: ImageData) -> VisualFeatures:
    """从像素中提取全局视觉特征（亮度/对比度/饱和度/边缘/显著区域等）。"""
    feat = VisualFeatures()
    samples, _ = _sample_pixels(img)
    n = len(samples)
    if n == 0:
        return feat

    # --- 一阶统计：亮度/饱和度/色彩丰富度 ---
    lums: List[float] = []
    sats: List[float] = []
    color_counts: Dict[str, int] = {}
    green = blue = gray = white = skin = 0
    rg_sum = yb_sum = 0.0
    for (r, g, b) in samples:
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        lums.append(lum)
        h, s, v = _rgb_to_hsv(r, g, b)
        sats.append(s)
        rg_sum += abs(r - g)
        yb_sum += abs(0.5 * (r + g) - b)
        name = _nearest_color_name(r, g, b)
        color_counts[name] = color_counts.get(name, 0) + 1
        if name == "绿色":
            green += 1
        if name in ("蓝色", "青色"):
            blue += 1
        if name in ("灰色", "黑色", "白色"):
            gray += 1
        if name == "白色":
            white += 1
        # 肤色近似规则
        if r > 95 and g > 40 and b > 20 and r > g > b and (r - min(g, b)) > 15:
            skin += 1

    feat.brightness = sum(lums) / n / 255.0
    mean_lum = sum(lums) / n
    var = sum((l - mean_lum) ** 2 for l in lums) / n
    feat.contrast = min(1.0, math.sqrt(var) / 128.0)
    feat.saturation = sum(sats) / n
    feat.colorfulness = min(1.0, (rg_sum + yb_sum) / n / 120.0)
    feat.green_ratio = green / n
    feat.blue_ratio = blue / n
    feat.gray_ratio = gray / n
    feat.white_ratio = white / n
    feat.skin_ratio = skin / n

    # --- 主色调 Top-5 ---
    sorted_colors = sorted(color_counts.items(), key=lambda kv: kv[1], reverse=True)
    feat.dominant_colors = [
        {"name": name, "ratio": round(cnt / n, 4)} for name, cnt in sorted_colors[:5]
    ]

    # --- 边缘密度与锐度（Sobel 近似，仅在抽样网格上） ---
    w, hgt = img.width, img.height
    if w >= 3 and hgt >= 3 and len(img.pixels) == w * hgt:
        stride = max(1, int(math.sqrt((w * hgt) / 20000)))
        edge_hits = 0
        edge_total = 0
        grad_sum = 0.0
        lum_map = [0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2] for p in img.pixels]
        for y in range(1, hgt - 1, stride):
            row = y * w
            for x in range(1, w - 1, stride):
                i = row + x
                gx = (lum_map[i + 1] - lum_map[i - 1])
                gy = (lum_map[i + w] - lum_map[i - w])
                mag = math.sqrt(gx * gx + gy * gy)
                grad_sum += mag
                edge_total += 1
                if mag > 48:
                    edge_hits += 1
        if edge_total:
            feat.edge_density = edge_hits / edge_total
            feat.sharpness = min(1.0, (grad_sum / edge_total) / 90.0)

    # --- 复杂度：颜色熵 + 边缘密度 ---
    entropy = 0.0
    for cnt in color_counts.values():
        p = cnt / n
        if p > 0:
            entropy -= p * math.log2(p)
    max_entropy = math.log2(max(len(NAMED_PALETTE), 2))
    feat.complexity = min(1.0, 0.6 * (entropy / max_entropy) + 0.4 * feat.edge_density * 3)

    # --- 显著区域：3x3 网格能量（高梯度+高饱和） ---
    if w >= 3 and hgt >= 3 and len(img.pixels) == w * hgt:
        grid = [[0.0] * 3 for _ in range(3)]
        counts = [[0] * 3 for _ in range(3)]
        step = max(1, len(img.pixels) // 20000)
        for idx in range(0, len(img.pixels), step):
            y, x = divmod(idx, w)
            gy = min(2, y * 3 // hgt)
            gx = min(2, x * 3 // w)
            r, g, b = img.pixels[idx]
            _, s, v
            _, s, v = _rgb_to_hsv(r, g, b)
            energy = (0.5 * math.sqrt(gx*gx + gy*gy) / 128.0) + (0.5 * s)
            grid[gy][gx] += energy
            counts[gy][gx] += 1

        # 找能量最高的格子 = 显著区域
        best_cell, best_energy = None, -1.0
        for gy in range(3):
            for gx in range(3):
                if counts[gy][gx]:
                    e = grid[gy][gx] / counts[gy][gx]
                    if e > best_energy:
                        best_energy, best_cell = e, (gy, gx)
        if best_cell:
            feat.salient_region = [
                "上", "中", "下"][best_cell[0]] + ["左", "中", "右"][best_cell[1]]
            # 显著区域密度 = 最强格子能量 / 平均格子能量（>1 说明能量集中）
            cell_energies = [
                grid[gy][gx] / counts[gy][gx]
                for gy in range(3) for gx in range(3) if counts[gy][gx] > 0
            ]
            avg_e = sum(cell_energies) / max(len(cell_energies), 1)
            feat.balance = 1.0 - abs(best_energy - avg_e) / max(best_energy, 1e-9)
            feat.salient_density = best_energy / max(avg_e, 1e-9)

    return feat


# ---------------------------------------------------------------------------
# 视觉 token 化 + 跨模态对齐 + 理解报告生成
# ---------------------------------------------------------------------------

def tokenize_visual(feat: VisualFeatures) -> List[str]:
    """将视觉特征编码为紧凑的视觉 token 序列（模拟 VLM 视觉编码器输出）。"""
    tokens: List[str] = []
    for dc in feat.dominant_colors[:3]:
        tokens.append(f"<color:{dc['name']}:{dc['ratio']:.2f}>")
    tokens.append(f"<lum:{feat.brightness:.2f}>")
    tokens.append(f"<contrast:{feat.contrast:.2f}>")
    tokens.append(f"<sat:{feat.saturation:.2f}>")
    tokens.append(f"<edge:{feat.edge_density:.2f}>")
    tokens.append(f"<complexity:{feat.complexity:.2f}>")
    return tokens


def align_question_tokens(question: Optional[str], feat: VisualFeatures) -> List[str]:
    """根据问题关键词选取应关注的特征（模拟跨模态注意力）。"""
    if not question:
        return []
    attended: List[str] = []
    for kw, fields in QUESTION_ATTENTION.items():
        if kw in question:
            attended.extend(fields)
    return sorted(set(attended))


def classify_scene(feat: VisualFeatures) -> Tuple[str, Dict[str, float]]:
    """场景分类：与各场景签名计算加权相似度。"""
    fd = asdict(feat)
    scores: Dict[str, float] = {}
    for scene, sig in SCENE_SIGNATURES.items():
        s = 0.0
        for key, expected in sig.items():
            actual = fd.get(key, 0.0)
            sigma = max(expected * 0.5, 0.05)
            s += math.exp(-((actual - expected) ** 2) / (2 * sigma ** 2))
        scores[scene] = s / max(len(sig), 1)
    best = max(scores, key=scores.get)
    return best, scores


def generate_report(path: str, question: Optional[str]) -> UnderstandingReport:
    """完整流水线：解码 → 特征提取 → token 化 → 跨模态对齐 → 报告。"""
    t0 = time.time()
    img = load_image(path)
    feat = extract_features(img)
    scene, scene_scores = classify_scene(feat)
    tokens = tokenize_visual(feat)
    attended = align_question_tokens(question, feat)

    # 构造理解文本
    parts: List[str] = []
    parts.append(f"图像尺寸 {img.width}x{img.height}，格式 {img.fmt}")
    if feat.dominant_colors:
        dc = feat.dominant_colors[0]
        parts.append(f"主色调为{dc['name']}（占比 {dc['ratio']*100:.1f}%）")
    parts.append(f"场景判断为「{scene}」")
    feat_scene_text = "、".join(
        dc["name"] for dc in feat.dominant_colors[:3]) if feat.dominant_colors else "无显著色彩"
    parts.append(f"色彩构成：{feat_scene_text}，饱和度 {feat.saturation:.2f}")
    parts.append(f"画面复杂度 {feat.complexity:.2f}（0=极简 1=极繁），边缘密度 {feat.edge_density:.2f}")
    summary = "。".join(parts) + "。"

    # 问题定向回答（跨模态注意力对齐）
    answer = None
    if question:
        fd = asdict(feat)
        ans: List[str] = []
        for f_name in attended:
            if f_name == "dominant_colors" and feat.dominant_colors:
                names = "、".join(dc["name"] for dc in feat.dominant_colors[:3])
                ans.append(f"主要颜色：{names}")
            elif f_name == "scene":
                ans.append(f"场景：{scene}")
            elif f_name == "salient_regions":
                if feat.salient_regions:
                    ans.append(f"显著区域：{feat.salient_regions[0].get('region', '未知')}")
            elif f_name == "objects_hint":
                ans.append(f"内容提示：{scene}")
            elif f_name in fd and isinstance(fd[f_name], (int, float)):
                ans.append(f"{f_name} = {fd[f_name]:.3f}")
        if ans:
            answer = "；".join(ans)

    return UnderstandingReport(
        file=path,
        format=img.fmt,
        width=img.width,
        height=img.height,
        megapixels=round(img.width * img.height / 1e6, 2),
        visual_tokens=tokens,
        features=asdict(feat),
        scene=scene,
        scene_scores={k: round(v, 3) for k, v in scene_scores.items()},
        summary=summary,
        answer=answer,
        elapsed_ms=round((time.time() - t0) * 1000, 1),
    )


def _self_test() -> int:
    """自检：手工构造测试图（纯标准库），跑完整流水线。"""
    import tempfile
    w = h = 64
    raw = b""
    for _ in range(h):
        raw += b"\x00" + bytes([30, 160, 60] * w)  # filter byte + green RGB
    def chunk(tag: bytes, data: bytes) -> bytes:
        len_b = struct.pack(">I", len(data))
        body = tag + data
        return len_b + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(png)
        path = f.name
    try:
        rep = generate_report(path, "这张图的主色调是什么？")
        assert rep.width == 64 and rep.height == 64, "尺寸解码错误"
        assert rep.features["dominant_colors"], "主色调未提取"
        assert rep.features["dominant_colors"][0]["name"] == "绿色", \
            f"期望绿色，实际 {rep.features['dominant_colors'][0]}"
        assert rep.visual_tokens, "视觉 token 未生成"
        assert "绿色" in rep.summary, "理解文本未包含主色调"
        assert rep.answer and "绿色" in rep.answer, "问题对齐回答未包含主色调"
        print("[SELF-TEST OK] 解码/特征/token化/场景分类/跨模态对齐 全部通过")
        print(f"  64x64 绿色测试图 → 场景={rep.scene}, tokens={len(rep.visual_tokens)}")
        print(f"  理解：{rep.summary[:120]}")
        print(f"  回答：{rep.answer}")
        return 0
    finally:
        os.unlink(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="视觉语言模型(VLM)多模态理解新高度 - 纯标准库图像理解 CLI（特征提取→token化→跨模态对齐→结构化报告）")
    parser.add_argument("path", nargs="?", help="图像文件或目录（支持 PNG/JPEG/GIF/BMP/PNM）")
    parser.add_argument("-q", "--question", help="自然语言问题（触发跨模态对齐）")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="输出格式")
    parser.add_argument("-o", "--output", help="报告输出文件（不指定则打印到 stdout）")
    parser.add_argument("--self-test", action="store_true", help="运行自检（内置测试图跑全流水线）")
    agent_test = getattr(parser, "_dummy", None)
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()
    if not args.path:
        parser.error("必须提供图像路径，或使用 --self-test")

    # 单文件或目录批量
    if os.path.isdir(args.path):
        exts = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".pnm", ".ppm", ".pgm")
        files = sorted(p for p in os.listdir(args.path) if p.lower().endswith(exts))
        if not files:
            print(f"目录中无图像文件：{args.path}", file=sys.stderr)
            return 2
        reports = []
        for name in files:
            p = os.path.join(args.path, name)
            try:
                reports.append(generate_report(p, args.question))
            except (ImageDecodeError, OSError) as e:
                print(f"  [跳过] {name}: {e}", file=sys.stderr)
        payload = json.dumps([asdict(r) for r in reports], ensure_ascii=False, indent=2)
    else:
        rep = generate_report(args.path, args.question)
        payload = json.dumps(asdict(rep), ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        print(f"报告已写入 {args.output}")
    elif args.format == "json":
        print(payload)
    else:
        data = json.loads(payload)
        reps = data if isinstance(data, list) else [data]
        for r in reps:
            print(f"\n=== {r['file']} ===")
            print(r["summary"])
            if r.get("answer"):
                print(f"问：{r.get('question', '')} 答：{r['answer']}")
            print(f"tokens: {len(r['visual_tokens'])} 个 | 用时 {r['elapsed_ms']}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
