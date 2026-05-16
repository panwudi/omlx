<p align="center">
  <img alt="Flyto MLX" src="docs/images/icon-rounded-light.svg" width="140">
</p>

<h1 align="center">Flyto MLX</h1>
<p align="center">
  <b>Apple Silicon LLM 服务器 · Audio chat · DFlash 双引擎 · 中文模型预设</b><br>
  Based on <a href="https://github.com/jundot/omlx">oMLX</a> by <a href="https://github.com/jundot">@jundot</a>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-green" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
  <a href="https://gitee.com/panwudi/flyto-mlx"><img src="https://img.shields.io/badge/Gitee-mirror-c71d23" alt="Gitee mirror"></a>
</p>

---

**中文** | [English](#english)

## 简介

Flyto MLX 是面向**中国 Mac 用户**与**国产模型生态**优化的 Apple Silicon 本地 LLM 服务器，基于 [@jundot/oMLX](https://github.com/jundot/omlx) fork。在保留 oMLX 全部上游能力（OpenAI 兼容 API、多模型 LRU 调度、KV 分页缓存、Mac menubar GUI）的基础上，加入了上游尚未合并/未支持的功能：

| 能力 | 说明 |
|---|---|
| **Gemma 4 audio chat** | OpenAI `input_audio` content type 端到端支持，调用 `gemma4-e2b` / `gemma4-e4b` 直接听音频回答（不是 ASR 替代，是端到端 audio understanding） |
| **DFlash 双引擎 (Path A)** | Qwen / Gemma 4 双 backend，drafter co-loaded 优化 |
| **Tahoe 兼容** | macOS 26 NSStatusItem occlusion bit 修复 |
| **上游已修但未发版的 backport** | tokenizer lm_head、TokenBuffer cache hit seed、health-check Session 复用 等 5 处 |
| **中文模型预设** | Qwen 3.5 MoE/Dense / DeepSeek V4 / Gemma 4 / 等 alias 即装即用 |
| **Gitee 镜像 + ModelScope 模型源** | 国内 access 优化 |

## 安装

**推荐：Homebrew tap**（macOS 最自然的方式，跟上游 oMLX 同模式）

```bash
brew tap panwudi/flyto-mlx https://github.com/panwudi/flyto-mlx
brew install flyto-mlx
brew services start flyto-mlx     # launchd 自动起 server :8000

# CLI 试用
fmlx serve --port 8000             # 主名
omlx serve --port 8000             # 兼容上游 oMLX 脚本的 alias
```

**备用：pip 从 git 装**（Linux / 已有 Python 环境 / 想 editable 开发）

```bash
pip install git+https://github.com/panwudi/flyto-mlx@v0.4.1
# 国内访问慢的话用 Gitee 镜像（开通后）：
# pip install git+https://gitee.com/panwudi/flyto-mlx@v0.4.1
```

PyPI `pip install flyto-mlx` 暂不可用 — Flyto MLX 跟上游 oMLX 一样依赖几个未 release 到 PyPI 的 mlx-vlm commits（speculative utils refactor 等），PEP 508 禁止 PyPI package 含 git URL 依赖。等 mlx-vlm 0.6.x release 含相关 commits 后会启用 PyPI 通道。

## 快速试 audio chat

```bash
# 假设 server 已起在 :8000，API key 设为 mykey
python3 <<'PY'
import base64, requests, json
with open("recording.wav","rb") as f:
    b64 = base64.b64encode(f.read()).decode()
r = requests.post(
    "http://localhost:8000/v1/chat/completions",
    headers={"Authorization": "Bearer mykey"},
    json={
        "model": "gemma4-e2b",
        "max_tokens": 400,
        "temperature": 0.3,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "总结这段电话的关键信息"},
            {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}
        ]}]
    },
)
print(r.json()["choices"][0]["message"]["content"])
PY
```

## 跟上游 oMLX 的关系

Flyto MLX 是 oMLX 的下游 fork，遵循 Apache 2.0。我们**定期 cherry-pick 上游 bug fix 与新模型支持**，但不再向上游 PR 自家 feature（audio chat、DFlash 等）。如果你只想要纯上游体验，请用 [@jundot/oMLX](https://github.com/jundot/omlx)。

详细 attribution 与版权声明见 [NOTICE](NOTICE) 与 [LICENSE](LICENSE)。

## License

Apache License 2.0. Based on oMLX by [@jundot](https://github.com/jundot). 详见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

---

## English

Flyto MLX is a fork of [@jundot/oMLX](https://github.com/jundot/omlx) optimized for the Chinese Mac LLM community and sovereign-AI model ecosystem (Qwen, DeepSeek, Gemma 4). It preserves all upstream oMLX capabilities (OpenAI-compatible API, multi-model LRU scheduling, KV paged cache, menubar GUI) and adds:

- **Audio chat via OpenAI `input_audio`** — end-to-end Gemma 4 nano audio LLM through `/v1/chat/completions`
- **DFlash Path A double-engine** — Qwen and Gemma 4 backends with optimized drafter co-loading
- **macOS 26 Tahoe compatibility** — NSStatusItem occlusion bit fix
- **5 upstream-fixed-but-unreleased patches backported** — tokenizer lm_head, TokenBuffer cache hit seed, health-check session reuse, and more
- **Chinese model presets** — Qwen 3.5 MoE/Dense, DeepSeek V4, Gemma 4 aliases ready out of the box
- **Gitee mirror + ModelScope model registry** — for users in mainland China

Install: `pip install flyto-mlx`. CLI: `fmlx serve` (or `omlx serve` alias for upstream compatibility).

We periodically cherry-pick upstream fixes. We do **not** upstream our own features back. For pure upstream behaviour, please use [@jundot/oMLX](https://github.com/jundot/omlx) directly.

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
