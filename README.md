# DSV4.1 TensorFold RR · 双 GB10 缓存修复与分块 PP 轮转

DeepSeek-V4.1-Flash EXL3 on two NVIDIA GB10 nodes: prefix-cache conflict repair and cooperative round-robin prefill with multi-stream decode.

**这是 ywytrew 基于原作者工作继续修改的实验分支。** 引擎原作者是 [ashhart / TensorFold contributors](https://github.com/ashhart/TensorFold)；DeepSeek-V4.1 的原生 Zig 模型实现、TP2 和部署配方来自 **[Bertholomus AI / Albert Lee](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig)**。本仓库保留上游 Git 历史、许可证和署名，新增工作仅限下述改动。完整来源见 [CREDITS.md](CREDITS.md)，原分支 README 保留在 [README.upstream.md](README.upstream.md)。

本分支以 `bertholomus/TensorFold` 的 `deepseek-v41-zig` 分支、提交 `dfc8fe50af303fcbdda0e839cae3a18eb53be2e1` 为公共基线。上层原始部署配方：[deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10)。

## 改了什么

- 修复长会话恢复后继续写入时，与其他已保留前缀快照的区间发生冲突的问题；必要时搬移，并保留原快照。
- 文本 prefill 按最多 2048 token 分块推进，多条未完成 PP 轮转；已有输出流可在 PP 块之间推进最多 8 轮 decode。
- 保留此前的图片内容缓存键、缓存命中 token 指标和 `/v1/models` 上下文长度字段。
- 保留实验性的“仅一条未完成 PP”模式，默认启动示例选择 **PP 轮转 + 多路 decode**。

**这里的并发是共享 CUDA stream 上的交错调度，不表示两个 PP kernel 同时执行。** 它主要改善“一条请求补充资料进入 PP，另一条正在输出却长时间停顿”的体验。首条冷请求的首 token 可能更晚，不保证单流 PP 不下降，也不保证所有负载总吞吐都增加。图片请求仍走同步 prefill 路径。池压力仍可能导致恢复失败；未合并另行试验的 pool 阈值 / GC 补丁。

## 实测摘录（2026-10-17）

双 GB10、TP2、双 RoCE，context/pool 1,048,576，prompt cache 2 GiB，parallel 4。两位用户各执行：约 190K 输入 → 长输出 → 模拟检索 1 秒 → 追加资料 PP → 长输出。每个冷/热循环输出总量均为 14,336 token。

| 同一双用户工作负载 | 冷循环完成时间 | 热循环完成时间 | 热轮追加 PP 时另一流最长相关停顿 |
|---|---:|---:|---:|
| 本地基线：同步 PP + 多路 decode | 352.862 s | 150.420 s | 7.925 s |
| **本分支：PP 轮转 + 多路 decode** | **342.570 s** | **149.120 s** | **2.594 s** |
| 实验选项：单路 PP + 多路 decode | 374.412 s | 155.385 s | 2.586 s |

每种模式仅一个冷循环、一个热循环，固定测试顺序；尤其不足 1% 的耗时差不能作为稳定加速保证。“基线”已经包含缓存修复，是调度对照，不是未经修改的公共上游。完整方法、校验和限制见 [BENCHMARKS.md](docs/dsv41-rr/BENCHMARKS.md)。

## 构建和启动

需要 Linux aarch64、两台 GB10、Zig 0.17.0、CUDA `nvcc`、兼容的容器环境，以及原作者配方要求的 checkpoint、kernel kit、Engram 文件和每个 rank 的缓存。**本仓库不分发权重、第三方 kernel kit 或预编译引擎。** 沿用原作者配方准备资源及遵守其各自许可证。

```bash
git clone --branch dsv41-rr https://github.com/ywytrew/dsv41-tensorfold-rr.git
cd dsv41-tensorfold-rr
ZIG_DIR=/path/to/zig-aarch64-linux-0.17.0 \
  IMAGE=dsv41-tp2-engine:local bash tools/dsv41-rr/build.sh
```

把同一次构建的整个输出目录复制到两台机器。先在 worker 执行 rank 1，再在 head 执行 rank 0。每台机器分别配置自己的目录、HCA、网卡和正确的 GID 索引：

```bash
# 在两台机器各自设置；占位符必须替换为实际配置。
export HEAD='<head 的互联地址>' RDMA_GID_INDEX='<实测 GID 索引>'
export DSV41_HOME="$HOME/dsv41" OUT_DIR='/absolute/path/to/build-out-rr'
export IMAGE='dsv41-tp2-engine:local'

# worker，双线示例：
bash tools/dsv41-rr/start-dual.sh 1
# head，双线示例：
bash tools/dsv41-rr/start-dual.sh 0
# 单线改用 start-single.sh；两端必须选相同连接方案。
```

示例监听 `127.0.0.1:18891`，不修改网关或已有服务；默认容器名为 `dsv41-rr-rank0/1`。它们是可移植示例，和本地生产入口脚本不同。使用前请阅读 [DEPLOYMENT.md](docs/dsv41-rr/DEPLOYMENT.md)，并用 `DRY_RUN=1` 查看命令。

| 变量 | RR 示例默认值 | 含义 |
|---|---:|---|
| `TF_DS_PREFILL_INTERLEAVE` | `1` | 启用文本分块调度；引擎本身默认关闭 |
| `TF_DS_PREFILL_SERIAL` | `0` | `1` 时只接纳一条未完成 PP，已有 decode 仍推进 |
| `TF_DS_PREFILL_TRACE` | `0` | `1` 输出逐块调度日志，供隔离测试使用 |
| `CTX` / `PCACHE` / `PARALLEL` | `1048576` / `2` / `4` | 两端 pool/context 必须一致 |

## 许可证与署名

沿用 [Apache-2.0](LICENSE)，保留 [NOTICE](NOTICE)、[LICENSES/](LICENSES/)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [DSV41 原始来源说明](tools/dsv41-zig/ATTRIBUTION.md)。本分支修改的文件含修改声明；文件清单见 [CHANGES.md](docs/dsv41-rr/CHANGES.md)。各模型、依赖和工具的原许可证继续适用。本分支不代表原作者或 NVIDIA、DeepSeek、Mia-AiLab 的认可。
