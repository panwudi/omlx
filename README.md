<p align="center">
  <img alt="Flyto MLX" src="docs/images/icon-rounded-light.svg" width="140">
</p>

<h1 align="center">Flyto MLX</h1>
<p align="center">
  <b>Apple Silicon 本地大模型推理服务 · 音频对话 · DFlash 双引擎 · 国产模型预设</b><br>
  派生自 <a href="https://github.com/jundot/omlx">@jundot/oMLX</a>。
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-green" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
  <a href="https://gitee.com/panwudi/flyto-mlx"><img src="https://img.shields.io/badge/Gitee-mirror-c71d23" alt="Gitee 镜像"></a>
</p>

---

**中文** | [English](#english)

## 简介

Flyto MLX 是一款面向 Mac 用户、聚焦中文与国产模型生态（通义千问、DeepSeek、Gemma 4 等）的本地大模型推理服务。它派生自 [@jundot/oMLX](https://github.com/jundot/omlx)，完整保留上游所有能力——OpenAI 兼容 API、多模型按需调度、KV 分页缓存、菜单栏图形界面——并在此之上加入了上游暂未合入或不支持的新功能。

## 跟上游 oMLX 有什么不同

### 一、Gemma 4 音频对话

完整打通了 OpenAI `input_audio` 内容类型在 `/v1/chat/completions` 上的端到端链路。可以直接调 `gemma4-e2b` / `gemma4-e4b` 让模型"听到"音频后回答问题——不是简单的语音转写，而是带着声音特征（情绪、语速、犹豫、口音）参与推理。

实测一段 158 秒的中文销售电话录音，模型既能输出贴近原文的转写，也能回答"客户的态度是什么"、"关键诉求是什么"这类需要理解才能答的问题。

> 背景：上游 oMLX 把音频在六个不同位置悄悄切断了，从内容解析器、Pydantic schema 到 chat 模板和引擎调度。我们逐处修好。

### 二、DFlash 双引擎（Path A）

通义千问和 Gemma 4 两套推理后端共存，草稿模型与目标模型同载共用 Metal 内存。Mac mini / Studio 跑 30B 以上模型时，吞吐量提升明显。

### 三、macOS 26（Tahoe）兼容

新版 macOS 把菜单栏遮挡检测的标志位从 `0x2` 改成了 `0x2000`。不修这一处，菜单栏状态会判错。

### 四、回填上游已修但还没发版的五个修复

包括 tokenizer 词表大小取 lm_head 权重、缓存命中时 TokenBuffer 种子重建、健康检查复用 HTTP Session 防端口耗尽等。

### 五、中文模型预设

通义千问 3.5 系列（Dense 与 MoE）、DeepSeek V4、Gemma 4 全家的别名开箱即用。MoE 别名按上游模型卡的命名习惯显式标活跃参数量，例如 `qwen-moe-35b-a3b`、`qwen-moe-122b-a10b`、`gemma4-moe-26b-a4b`。

### 六、国内访问加速

Gitee 镜像（每小时自动同步 GitHub），后续接 ModelScope 作为模型下载备选源。

## 安装

**推荐方式：Homebrew tap**——Mac 上最自然的方式，跟上游 oMLX 同一路线。

```bash
brew tap panwudi/flyto-mlx https://github.com/panwudi/flyto-mlx
brew install flyto-mlx
brew services start flyto-mlx     # 通过 launchd 启动服务，默认监听 :8000
```

命令行入口：`fmlx serve --port 8000`（主入口）或 `omlx serve --port 8000`（保留作为兼容上游脚本的别名）。

**备选方式：从 git 直装**——适合 Linux 用户，或者已经有 Python 环境想做开发的人。

```bash
pip install git+https://github.com/panwudi/flyto-mlx@v0.4.1
# 国内访问慢的话可以用 Gitee 镜像（开通后）：
# pip install git+https://gitee.com/panwudi/flyto-mlx@v0.4.1
```

**为什么暂时不能直接 `pip install flyto-mlx`**：Flyto MLX 跟上游 oMLX 一样，依赖 mlx-vlm 几个尚未发布到 PyPI 的提交（speculative utils refactor 等）。而 PyPI 严格禁止包的依赖里出现 git URL（PEP 508 §6 的约束）。等 mlx-vlm 0.6.x 把这些提交正式发版后，我们会启用这条通道。

## 快速试一下音频对话

```python
import base64, requests

with open("recording.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    headers={"Authorization": "Bearer 你的密钥"},
    json={
        "model": "gemma4-e2b",
        "max_tokens": 400,
        "temperature": 0.3,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "总结这段电话的关键信息"},
                {"type": "input_audio",
                 "input_audio": {"data": audio_b64, "format": "wav"}}
            ]
        }]
    }
)
print(resp.json()["choices"][0]["message"]["content"])
```

## 跟上游 oMLX 的关系

Flyto MLX 是 oMLX 的下游派生，遵循 Apache 2.0 协议。

- **我们做的**：定期从上游回挑 bug 修复和新模型支持，保持不掉队
- **我们不做的**：不再把自己的功能（音频对话、DFlash 等）反向 PR 给上游

如果你只需要纯净的上游体验，请直接用 [@jundot/oMLX](https://github.com/jundot/omlx)。

完整的版权与署名信息见 [NOTICE](NOTICE) 与 [LICENSE](LICENSE)。

## 协议

Apache License 2.0。派生自 [@jundot](https://github.com/jundot) 的 oMLX。详见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

---

## English

Flyto MLX is a downstream fork of [@jundot/oMLX](https://github.com/jundot/omlx) optimized for the **Chinese Mac LLM community** and the **sovereign-AI model ecosystem** (Qwen, DeepSeek, Gemma 4). It preserves every upstream oMLX capability (OpenAI-compatible API, multi-model LRU scheduling, KV paged cache, menubar GUI) and adds:

- **End-to-end Gemma 4 audio chat via OpenAI `input_audio`** — call `gemma4-e2b` / `gemma4-e4b` and let the model *listen* to audio, not just transcribe it. Tested against a 158-second Chinese sales call: produces both a faithful transcription and meaningful answers to questions like "what's the customer's attitude?" or "what are the key asks?" — answers an ASR-then-LLM pipeline cannot give because it has lost the prosody and pauses.
- **DFlash Path A double-engine** — Qwen and Gemma 4 backends with drafter co-loaded for Metal-memory efficiency. Significant throughput gains for 30B+ models on Mac mini / Studio.
- **macOS 26 (Tahoe) compatibility** — NSStatusItem occlusion bit fix (`0x2000` instead of legacy `0x2`).
- **Five upstream-fixed-but-not-yet-released backports** — tokenizer `lm_head` vocab size, TokenBuffer cache hit seeding, health-check session reuse, and more.
- **Chinese model presets** — Qwen 3.5 (Dense and MoE), DeepSeek V4, Gemma 4 aliases ready out of the box. MoE aliases follow the upstream naming convention of explicit active params (`qwen-moe-35b-a3b`, `qwen-moe-122b-a10b`, `gemma4-moe-26b-a4b`).
- **Gitee mirror + ModelScope model registry** — for users in mainland China.

## Install

**Recommended: Homebrew tap** (same pattern as upstream oMLX)

```bash
brew tap panwudi/flyto-mlx https://github.com/panwudi/flyto-mlx
brew install flyto-mlx
brew services start flyto-mlx     # launchd service on :8000
```

CLI: `fmlx serve --port 8000` (primary), or `omlx serve --port 8000` (alias kept for compatibility with upstream scripts).

**Alternative: pip from git**

```bash
pip install git+https://github.com/panwudi/flyto-mlx@v0.4.1
```

`pip install flyto-mlx` (plain PyPI) is **not currently available**. Flyto MLX, like upstream oMLX, depends on unreleased mlx-vlm commits (speculative utils refactor and others) that PEP 508 §6 prevents from being expressed in PyPI packages. Once `mlx-vlm 0.6.x` releases with those commits, we'll enable the PyPI channel.

## Relationship to upstream oMLX

Flyto MLX is a downstream fork of oMLX, governed by Apache 2.0.

- **We do**: regularly cherry-pick upstream bug fixes and new model support
- **We don't**: upstream our own features (audio chat, DFlash, etc.) back to oMLX

For the pure upstream experience, please use [@jundot/oMLX](https://github.com/jundot/omlx) directly.

Full attribution in [NOTICE](NOTICE) and [LICENSE](LICENSE).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
