#!/usr/bin/env python3
'''
jingyaogong/minimind - MiniMind 极简语言模型训练/推理演示工具

MiniMind（github.com/jingyaogong/minimind）是一个从零训练小型语言模型的
开源项目。本脚本是它的配套演示：在纯标准库环境下，实现一个字符级
mini language model 的「训练 + 生成」完整闭环，用于直观理解
transformer 之前的经典 n-gram/马尔可夫语言模型范式。

功能：
  train   - 在训练文本上构建字符级 n-gram 语言模型（默认 3-gram）
  gen     - 用训练好的模型自回归生成文本
  ppl     - 计算一段文本的困惑度（perplexity）
  stats   - 查看模型统计信息

用法：
  python3 minimind_demo.py train corpus.txt -m model.json
  python3 minimind_demo.py gen -m model.json --prompt "今天" -n 50
  python3 minimind_demo.py ppl -m model.json test.txt
  python3 minimind_demo.py stats -m model.json
  python3 minimind_demo.py demo            # 内置语料自检
'''

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

VERSION = "1.0.0"


class MiniMindNGram:
    """字符级 n-gram（马尔可夫链）语言模型——minimind 的极简替身。

    模型结构（JSON）：
      {"n": 3, "vocab": [...], "counts": {"context": {"next_char": count}}}
    生成时按计数加权随机采样下一个字符，实现自回归展开。
    """

    def __init__(self, n: int = 3):
        self.n = n
        self.vocab: Counter = Counter()
        self.counts: dict = defaultdict(Counter)

    # ---------------- 训练 ----------------

    def train(self, text: str):
        """扫一遍语料构建 n-gram 计数表。"""
        self.vocab.update(text)
        n = self.n
        for i in range(len(text) - n):
            ctx = text[i:i + n]
            nxt = text[i + n]
            self.counts[ctx][nxt] += 1

    def save(self, path: str):
        data = {
            "n": self.n,
            "vocab": dict(self.vocab.most_common()),
            "counts": {ctx: dict(cnt) for ctx, cnt in self.counts.items()},
        }
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "MiniMindNGram":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        m = cls(n=data["n"])
        m.vocab = Counter(data["vocab"])
        m.counts = {ctx: Counter(cnt) for ctx, cnt in data["counts"].items()}
        return m

    # ---------------- 推理 ----------------

    def next_dist(self, context: str) -> dict:
        """给定上下文，返回下一字符的分布；回退到更短上下文直到命中。"""
        for k in range(self.n, 0, -1):
            ctx = context[-k:]
            if ctx in self.counts:
                return dict(self.counts[ctx])
        return {}

    def generate(self, prompt: str, length: int = 50,
                 temperature: float = 1.0) -> str:
        """自回归生成：逐字符采样，未见过的上下文回退到更短的 n-gram。"""
        out = list(prompt)
        for _ in range(length):
            dist = self.next_dist("".join(out))
            if not dist:
                # 上下文完全未见：从全局字符分布兜底
                dist = dict(self.vocab) or {"。": 1}
            chars = list(dist.keys())
            weights = [dist[c] ** (1.0 / max(temperature, 0.1))
                       for c in chars]
            out.append(random.choices(chars, weights=weights)[0])
        return "".join(out)

    def perplexity(self, text: str) -> float:
        """困惑度 = exp(平均负对数似然)。越低说明模型越能「理解」这段文本。"""
        total = 0.0
        cnt = 0
        for i in range(self.n, len(text)):
            dist = self.next_dist(text[:i])
            if not dist:
                dist = dict(self.vocab) or {"。": 1}
            total_cnt = sum(dist.values())
            p = dist.get(text[i], 0) / total_cnt
            if p <= 0:
                p = 1e-10  # 平滑：未见事件给极小概率
            total += -math.log(p)
            cnt += 1
        return math.exp(total / max(cnt, 1))

    def stats(self) -> dict:
        total_bigrams = sum(c for cnt in self.counts.values() for c in cnt.values())
        return {
            "order": self.n,
            "vocab_size": len(self.vocab),
            "contexts": len(self.counts),
            "total_transitions": total_bigrams,
            "top_chars": self.vocab.most_common(5),
        }


# ---------------------------------------------------------------------------
# 内置演示语料（自检用）
# ---------------------------------------------------------------------------

DEMO_CORPUS = (
    "人工智能正在改变世界。语言模型是人工智能的核心技术之一。"
    "语言模型通过预测下一个字符来学习语言的规律。"
    "最小化的语言模型只需要统计字符出现的频率。"
    "minimind 项目证明，一个人也可以从零训练一个语言模型。"
    "从数据到模型，从训练到推理，每一步都可以亲手实现。"
    "理解语言模型的最好方式，就是自己动手训练一个。"
) * 8


def cmd_demo(args) -> int:
    """内置语料全流程自检：train → save/load → gen → ppl → stats。"""
    print("== MiniMind Demo（字符级 n-gram 语言模型全流程）==\n")
    m = MiniMindNGram(n=3)
    m.train(DEMO_CORPUS)
    print(f"[1] 训练完成: {m.stats()['contexts']} 个上下文, "
          f"{m.stats()['total_transitions']} 次转移")

    # 序列化往返
    m.save("/tmp/minimind_demo_model.json")
    m2 = MiniMindNGram.load("/tmp/minimind_demo_model.json")
    assert m2.stats() == m.stats(), "序列化往返数据不一致"
    print(f"[2] 模型保存/加载一致 ✓")

    random.seed(42)
    gen = m.generate("语言模型", 30)
    print(f"[3] 生成: {gen}")

    ppl_train = m.perplexity(DEMO_CORPUS[:200])
    ppl_oov = m.perplexity("zzzz qq ww xx 1234 %%%%",
                           ) if False else m.perplexity("zzzqqqwww异常文本")
    print(f"[4] 困惑度: 训练文本={ppl_train:.3f} (低) vs 乱码={ppl_oov:.3f} (高)")
    assert ppl_oov > ppl_train, "乱码困惑度应显著高于训练文本"

    print(f"[5] 统计: {json.dumps(m.stats(), ensure_ascii=False)}")
    print("\n[DEMO OK] train/save/load/generate/perplexity 全部通过")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="jingyaogong/minimind 演示工具 - 字符级 n-gram 语言模型 train/gen/ppl/stats")
    sub = parser.add_subparsers(dest="cmd")

    p_demo = sub.add_parser("demo", help="内置语料全流程自检")
    p_demo.set_defaults(func=cmd_demo)

    p_train = sub.add_parser("train", help="在语料上训练模型")
    p_train.add_argument("corpus", help="训练语料文本文件")
    p_train.add_argument("-m", "--model", default="model.json", help="模型输出路径")
    p_train.add_argument("--order", type=int, default=3, help="n-gram 阶数（默认 3）")
    p_train.set_defaults(func=lambda a: _cmd_train(a))

    p_gen = sub.add_parser("gen", help="生成文本")
    p_gen.add_argument("-m", "--model", default="model.json", help="模型路径")
    p_gen.add_argument("--prompt", default="", help="起始提示文本")
    p_gen.add_argument("-n", "--length", type=int, default=50, help="生成长度")
    p_gen.add_argument("-t", "--temperature", type=float, default=1.0)
    p_gen.add_argument("--seed", type=int, default=None)
    p_gen.set_defaults(func=lambda a: _cmd_gen(a))

    p_ppl = sub.add_parser("ppl", help="计算文本困惑度")
    p_ppl.add_argument("-m", "--model", default="model.json", help="模型路径")
    p_ppl.add_argument("text_file", help="待评估文本文件")
    p_ppl.set_defaults(func=lambda a: _cmd_ppl(a))

    p_stats = sub.add_parser("stats", help="模型统计")
    p_stats.add_argument("-m", "--model", default="model.json", help="模型路径")
    p_stats.set_defaults(func=lambda a: _cmd_stats(a))

    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 1
    return args.func(args)


def _cmd_train(args) -> int:
    text = Path(args.corpus).read_text(encoding="utf-8")
    if len(text) < 100:
        print(f"⚠️  语料过短（{len(text)} 字符），建议 ≥1000", file=sys.stderr)
    m = MiniMindNGram(n=max(2, args.order))
    m.train(text)
    m.save(args.model)
    s = m.stats()
    print(f"训练完成: {s['contexts']} 上下文 / {s['total_transitions']} 转移"
          f" / 词表 {s['vocab_size']} → {args.model}")
    return 0


def _cmd_gen(args) -> int:
    m = MiniMindNGram.load(args.model)
    if args.seed is not None:
        random.seed(args.seed)
    print(m.generate(args.prompt, args.length, args.temperature))
    return 0


def _cmd_ppl(args) -> int:
    m = MiniMindNGram.load(args.model)
    text = Path(args.text_file).read_text(encoding="utf-8")
    print(f"{m.perplexity(text):.4f}")
    return 0


def _cmd_stats(args) -> int:
    m = MiniMindNGram.load(args.model)
    print(json.dumps(m.stats(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
