# FlexEngine 2.0

## An SLO-Aware, GPU-Optimized Modular LLM Inference Runtime

**工程研究项目 Proposal + 实验设计与最终评估报告模板**  
**版本：** 2.0 · 2026 年 11 月 8 日  
**状态：** Proposed / 尚未实现；本文所有实验结果均为计划，不代表已完成  
**硬件边界：** 单卡 NVIDIA RTX 3080 Ti，按用户提供的 16 GiB 显存配置规划；实验前以 `nvidia-smi` 实际容量为准  
**岗位定位：** LLM Inference Engineer · Model Runtime Engineer · GPU Performance Engineer

---

## 执行摘要

FlexEngine 2.0 旨在从零搭建一个**可运行、可解释、可测量**的单 GPU LLM 推理运行时：使用 Hugging Face 模型权重，但不把 `transformers.generate()` 或 vLLM 作为隐藏的调度/缓存/执行引擎。项目自行拥有请求状态机、Token-Budget Scheduler、Continuous Batching、Paged KV 分配器、直接读取 Block Table 的 Attention 执行路径、Sampler、SSE 流式接口和系统可观测性。第二阶段增加基于服务级目标（SLO）的调度、Prefix Cache 和 CUDA Graph；第三阶段根据 Nsight 分析选择一个真正值得优化的 Triton Kernel。

**核心问题：** 单 GPU、有限显存、动态请求到达下，调度与 GPU 执行策略怎样影响 TTFT、ITL/TPOT、P95/P99、Goodput、显存占用和公平性？

**项目评判标准不是“战胜 vLLM”。** 最终成果是：自主实现的引擎 + correctness & failure test suite + 受控消融 + 确定性负载/统计设计 + Nsight 证据链 + 可复现报告。

## 1. 工业需求与项目范围

### 1.1 招聘岗位对照

对照 NVIDIA Inference Performance、NVIDIA AI Inference、Red Hat vLLM 岗位（参考资料 [R1]—[R3]），岗位反复强调四个技术层次：

| 工业技能 | FlexEngine 可交付、可审查的证据 | 优先级 |
|---|---|---|
| 推理 Runtime 内部实现 | 调度器、Batch Admission、请求生命周期、真实 Paged KV Attention | P0 必须 |
| 端到端性能工程 | TTFT / ITL / Goodput / Tail Latency 的有控制变量实验 | P0 必须 |
| GPU 调试与优化 | CUDA Events、Nsight Systems/Compute、一次完整瓶颈定位→优化→复测 | P1 强烈推荐 |
| 生产可靠性 | 取消、OOM、回收、背压、公平性和性能回归 CI | P0 必须 |
| 推理高级优化 | Prefix Cache、CUDA Graph、Triton Fusion、可选 Speculative Decoding | P1/P2 |
| 分布式推理 | TP/PP/EP、NCCL、PD Disaggregation | **明确不包含** |

### 1.2 Scope / Non-goals

**纳入范围：** decoder-only dense Causal LM；一个 NVIDIA CUDA GPU；FP16 inference；OpenAI 风格 `/v1/completions`；Eager 与可选 CUDA Graph；Python 运行时、PyTorch 执行、Triton 热点优化；Prometheus 指标；CPU-only 调度/分配器单测。

**明确排除：** 多 GPU/NCCL、MoE 专用专家并行、训练/微调、大模型权重分片、真正的多节点服务、把 vLLM 代码包装起来伪称自主 Runtime。FP8/NVFP4 不作为 RTX 3080 Ti 的核心性能路线；若做量化，以该设备能正确执行的 INT8/INT4 路线为可选单独研究。

**重点工程边界：** 只有 Block Table/Free List 而仍将整个 K/V 拼接成连续 Tensor 运行普通 Attention，只能称为 *Paged KV Allocator / Materialization Baseline*；必须进一步实现 **attention kernel 直接依据 block table 读取非连续 KV**，才能把“Paged Attention”作为正式完成的核心成果。

## 2. 目标架构与接口契约

```text
Client / AIPerf
      |
FastAPI: /v1/completions (stream / non-stream)
      |
Admission Control -> bounded waiting queue -> Request State Machine
                                              |
                                 Token-Budget Scheduler
                            (FCFS / Decode-first / SLO-aware)
                                      |        |
                                Block Planner  Request Metadata
                                      |
                              Paged KV Manager
                         (free list / block table / refcount)
                                      |
                            Qwen Decoder Adapter
                       (HF weights; custom execution path)
                                      |
                    Eager Executor / Captured Decode Graph
                                      |
                 Attention Backend (reference / paged kernel)
                                      |
                   Sampler -> Detokenizer -> SSE Stream
                                      |
               Metrics + JSONL traces + NVTX/CUDA Events
```

### 2.1 模块责任与 API

- `Request`: `request_id`, arrival time, tokens, deadline/SLO, generated tokens, `WAITING/RUNNING/PREEMPTED/FINISHED/CANCELLED/FAILED`；显式定义合法状态转移。
- `SchedulerPolicy.select(requests, token_budget, kv_budget) -> SchedulePlan`：返回每个请求本轮新增 token 数、执行顺序和 preemption 决策；预填充和解码共享 Token Budget。
- `KVAllocator.reserve/apply/free/refcount/block_table`：逻辑页→物理页映射；完成/取消时无泄漏；可选共享前缀及 copy-on-write/引用计数语义。
- `ModelAdapter.forward(input_ids, positions, block_tables, slot_mapping, attention_metadata)`：使用 HF 已加载权重，但执行内部 Attention 和 KV 写回为自研代码。
- `ExecutorBackend.run(plan)`：`eager` 基线，后续 `cudagraph`；对于不支持形状显式 fallback 并计数。
- `Sampler.sample(logits, policy)`：greedy、temperature、top-k；EOS、`max_new_tokens`；每请求独立 RNG 状态用于验证。
- `RuntimeMetrics`: latency histogram、队列长度、running 数、KV 已用/空闲 Block、preempt 次数、Graph replay 命中率。

### 2.2 建议实施顺序

**M0：** 带 `past_key_values`/reference attention 的真正逐 token decoder；逐步验证最后一层 logits。

**M1：** 自己的多请求批处理与排队调度；先使用连续 KV baseline，完成 cancellation/streaming。

**M2：** 固定大小 KV Blocks，真实 block-table 消费路径；先 correctness 优先，随后避免跨块复制。

**M3：** 混合 prefill/decode 的 token-budget scheduling、chunking、backpressure、preemption。

**M4：** SLO-aware 调度 + prefix caching + GPU profiling；依据结果决定 CUDA Graph 和 kernel 方向。

## 3. 硬件可行性和模型矩阵

### 3.1 模型选择

| 模型（官方 Hugging Face） | 层数 / KV Heads / Head Dim | 估算 FP16 权重 | 估算 KV / token | 用途 | 优先级 |
|---|---|---:|---:|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | 24 / 2 / 64 | ~1 GiB | 12 KiB | 原型、分配器 correctness、Kernel 单测 | M0—M4 核心 |
| `Qwen/Qwen2.5-1.5B-Instruct` | 28 / 2 / 128 | ~3 GiB | 28 KiB | 正式吞吐量、调度消融、Profiling | 主 benchmark |
| `Qwen/Qwen2.5-3B-Instruct` | 36 / 2 / 128 | ~6 GiB | 36 KiB | 内存压力/规模敏感性 | 可选验证 |

注：这里“GiB”权重为量级估计，参数量 × 2 Bytes 只是初步近似；实际要用模型加载后的 `torch.cuda.memory_allocated`、`memory_reserved` 和 `nvidia-smi` 记录。KV 估算按 FP16、无共享前缀、全层 GQA 和每个 token 的 K/V：`2 × layers × kv_heads × head_dim × 2 bytes`。模型结构来自官方配置 [R7]—[R9]。

**为什么不直接用 7B FP16？** 7B 权重约 14 GB（十进制 GB），剩余给 CUDA Context、Attention 中间张量、KV Blocks、Graph Pool 的空间过少，不适合重复且稳定的实验。以 0.5B/1.5B 建立系统能力，比硬塞 7B 更合适。

### 3.2 推荐的首批配置

```yaml
model: Qwen/Qwen2.5-0.5B-Instruct
hardware: cuda:0
precision: float16
max_model_len: 2048
max_num_seqs: 8
max_num_batched_tokens: 256
kv_block_size: 16
kv_memory_fraction_of_observed_free_vram: 0.60
attention_backend: reference_contiguous  # 初期
scheduler: fcfs
execution_backend: eager
prefix_cache: false
cuda_graph: false
streaming: true
```

**容量纪律：** 初始化时在加载权重后测量空闲显存，预留 CUDA Context、cuBLAS 工作区、temporary activations、Graph private pool 以及安全余量；动态决定 `num_gpu_blocks`，而不是将全部 16 GiB 用作 KV。可用显存以实际测量为准。对 Graph 的显存额外占用另做实验。

### 3.3 环境前置检查

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_properties(0))"
nsys --version
ncu --version
```

针对 Windows + WSL2，先实测 Nsight Compute 的性能计数器权限；NVIDIA 文档指出 WSL2 可以进行分析，但需要满足驱动和主机 Performance Counters 设置要求 [R10]。Linux/WSL 的 GPU 可用性与具体 PyTorch/Triton wheel 应通过 sanity test 验证。

## 4. 策略设计：从工程功能到可比较的优化假设

| 编号 | Strategy | 核心变化 | 主要考察指标 | 预期 trade-off |
|---|---|---|---|---|
| S0 | Single-request, sequential eager | 每次只执行一个完整请求 | 正确性、无并发 latency | 简单但 GPU 利用率可能低 |
| S1 | FCFS continuous batching | 逐步加入/移除请求 | output tok/s、TTFT、P99 | 高负载时排队 |
| S2 | Decode-first + chunked prefill | 优先保障 decode，剩余 budget 给 prefill | ITL P95、TTFT P95 | 长 prompt 首 token 可能等待更久 |
| S3 | SLO-aware scheduling + aging | 按 TTFT/ITL slack、等待时长决定优先级 | Goodput、SLO violation、fairness | 估计误差/饥饿风险 |
| S4 | Paged KV attention | 直接从非连续 Blocks 加载 KV | VRAM、分配失败率、tok/s | block indirection 开销 |
| S5 | Prefix cache (exact token prefix) | 同前缀复用已验证 KV Blocks | cache hit、saved prefill tokens、TTFT | 内存占用、失效/引用计数 |
| S6 | Decode CUDA Graph | 固定 batch buckets 的 graph replay | CPU step gap、decode ITL、VRAM | Graph capture 占用显存、fallback |
| S7 | Targeted Triton optimization | 基于 profiler 优化一个热 kernel | kernel us、E2E ITL、正确性 | 局部改进不一定传递到 E2E |

### 4.1 三类调度策略如何公平比较

- **FCFS（S1）**：先到先处理，运行中请求优先延续但不强制绝对 decode-first；每步更新运行集合。
- **Decode-first（S2）**：先给活跃 decode 请求分配 token，再使用余量为等待中的 prefill 分配 chunk；控制 `max_num_batched_tokens`，避免长 prompt 阻塞所有 decode。
- **SLO-aware（S3）**：定义请求 TTFT target、运行中 ITL target；用最近执行耗时的 EWMA 估计 slack，结合 waiting age / bounded priority escalation 防止 starvation。记录每次决策和预测偏差；不要假设 SLO 优化必然提升吞吐。
- **OOM/preemption**：首版使用 recompute-based preemption（释放请求 KV Blocks、需要时重算上下文），记录 recompute token 数和恢复率；不强行实现 GPU↔CPU KV swap。vLLM V1 也是统一 token-budget 和 recompute 优先的有用参考 [R4]—[R5]。

### 4.2 单独的性能假设（待证实，不写成结论）

- **H1**：在长 prompt 与短 decode 混合到达时，S2 比 S1 降低 decode ITL P95，但 TTFT P95 可能上升。
- **H2**：S3 在存在可满足 SLO 的过载前区域提升 *SLO Goodput*，代价可能是更高队列等待不均衡。
- **H3**：高重复前缀下 S5 降低 prefill GPU work / TTFT；无重复前缀的随机负载可能只增加管理开销。
- **H4**：S6 在小 batch、kernel-launch-bound 的 decode 中更有价值；大 batch 或 Graph 内存压力下不一定受益。
- **H5**：S4 减少频繁重分配和复制造成的峰值内存与碎片；Paged Attention 本身的 block gather 也可能带来额外指令开销。
- **H6**：Triton kernel 的微基准收益只有在该 kernel 是 E2E 关键路径且没有明显调度瓶颈时，才可能转化为最终 ITL 提升。

## 5. 测试提案：正确性、稳定性、故障和性能

### 5.1 Correctness Test Matrix（优先于性能）

| 测试 ID | 测试内容 | 检查条件/结果 | 必须 |
|---|---|---|---|
| C01 | Tokenizer / prompt encoding | 与指定 tokenizer 版本和 chat template 一致 | 是 |
| C02 | 单请求 logits parity | 固定输入，reference vs 自研 forward，明确 `atol/rtol` | 是 |
| C03 | Prefill vs incremental decode | 同一 token 序列对应 logits；RoPE/position 一致 | 是 |
| C04 | Batch isolation | 不同 requests 混批，与各自独立运行输出对齐 | 是 |
| C05 | Paged KV block boundary | 跨 15→16→17 token、非对齐长度、不同层/头 | 是 |
| C06 | Block reclaim / refcount | 完成、取消、抢占后 free-list 总量守恒 | 是 |
| C07 | Prefix cache correctness | 相同前缀命中、不同前缀不命中、提前终止不污染 | P1 |
| C08 | CUDA Graph parity | 相同输入/batch，Eager vs graph logits 一致 | P1 |
| C09 | Sampling semantics | greedy 固定输出；采样 RNG 隔离；EOS 截断 | 是 |
| C10 | API streaming | SSE 逐增量 token，finish_reason，断开后清理 | 是 |
| C11 | Scheduler invariants | budget 不超配；请求无丢失、重复调度或非法状态 | 是 |
| C12 | 多模型 sanity | 0.5B 和 1.5B 都能完整运行 | 是 |

**数值标准：** FP16 优先比较最大绝对误差、平均绝对误差、相对误差和逐位置 top-1 agreement；初始探索性阈值可用 `atol=2e-2, rtol=2e-2`，须按层、算子和 logits 标度校准。不可仅凭采样文本相同判定数学正确；也不能将浮点 logits 的逐 bit 一致作为普适要求。

### 5.2 Reliability / Stress Test Matrix

| ID | 故障情形 | 必须验证 |
|---|---|---|
| F01 | 在第 N 个 token 取消请求 | 无残留 KV blocks，其他请求不中断 |
| F02 | 客户端主动断开 SSE | 服务端识别 disconnect、释放请求资源 |
| F03 | KV 空间不足 / 过载 | 预定的 preempt/拒绝/延后规则；不得 OOM 崩溃 |
| F04 | 短时间 100 个请求进入容量 8 的队列 | admission backpressure，429/503、Retry-After 语义一致 |
| F05 | 小请求持续到达，同时有长请求 | 最大等待时间、公平性和 starvation 监控 |
| F06 | 反复申请/回收几万次 blocks | 分配器不泄漏、空闲块数守恒 |
| F07 | CUDA OOM 和 kernel 异常注入 | fail-fast、结构化错误、进程是否可以可靠恢复；必要时重启 worker |
| F08 | Prefix 命中与取消同时发生 | 引用计数正确，已共享块不可被提前释放 |

**不应承诺 CUDA OOM 后同一进程永远自动恢复。** 对不可恢复 GPU 状态明确触发 worker restart，并在测试里界定可靠性等级。

### 5.3 负载数据集：五种必跑 Workload

| 类别 | Prompt tokens | Output tokens | 到达模式 | 目的 |
|---|---:|---:|---|---|
| W1 Interactive chat | 128 | 128 | 固定并发 1/2/4/8 | 交互式 decode |
| W2 Long-context Q&A | 1024 | 32 | Poisson | Prefill 计算压力 |
| W3 Decode-heavy | 256 | 512 | Poisson | 长时间 Decode 与 KV 持续增长 |
| W4 Mixed | W1/W2/W3 = 50/30/20% | 混合 | Poisson + burst | 队列干扰与 Tail Latency |
| W5 Prefix-heavy | 512 shared + 128 unique | 128 | 多轮同前缀与随机对照 | Prefix cache 收益与误命中风险 |

另设**真实文本 trace**（例如许可证允许的 ShareGPT 样例，完成清理和长度统计），但基准长度以实际 tokenizer token 数为准；若采用合成 token，必须保证输入合法、不会意外触发 EOS、输出长度在可控策略下产生。**固定输入 token 数不代表模型一定生成指定输出长度**；吞吐比较应使用真实生成 token 数，输出长度分层统计，必要时提供 benchmark-only EOS suppression 并明确与 production 配置区别。

### 5.4 性能指标与计算规则

| 指标 | 定义 | 必报统计 |
|---|---|---|
| TTFT | client send → 首个有效 token 到达 client | p50/p95/p99 |
| ITL | 相邻实际新 token 的到达时间差 | token-level + request-level p50/p95 |
| TPOT | 每请求 `(end - first_token) / (generated_tokens - 1)`，仅适用 ≥2 个 token | p50/p95 |
| E2E latency | send → request 完成/终止 | p50/p95/p99 |
| Output throughput | 总生成 tokens / measurement wall time | tokens/s |
| Request throughput | 成功完成 requests / wall time | requests/s |
| Goodput | 同时满足所定义 TTFT + ITL/E2E SLO 的**成功请求数 / wall time** | requests/s + pass rate |
| KV efficiency | usable block tokens、reserved tokens、内部碎片及 cache hit | 每工作负载分布 |
| GPU memory | CUDA allocated/reserved + device peak used | GiB + 时间曲线 |
| Fairness | 排队时间分位数、最大等待时间、不同长度请求的服务差异 | p95/p99/分类 |

**注意：** ITL 与 TPOT 不等价；一次 SSE 事件可包含多 token；Token Count、first-token 的定义和服务端/客户端时间口径必须写入实验元数据。GPU Events 测纯 GPU step，client wall-clock 测端到端，不能混用。

## 6. 正式实验矩阵与消融计划

### 6.1 六组必做实验

| Experiment | 控制变量 | 对照组 | 结果图表 |
|---|---|---|---|
| E1 基线/容量 | 模型 0.5B/1.5B；并发 1/2/4/8（容量许可 16） | sequential vs continuous batch | throughput–concurrency；TTFT/ITL P95 |
| E2 调度策略 | FCFS / decode-first / SLO-aware；W2/W4 | 相同模型、token budget、cache、request trace | Goodput–arrival rate；P99；fairness |
| E3 Token Budget | 64/128/256/512 tokens/step | 相同调度/请求混合 | ITL–TTFT Pareto、batch 组成 |
| E4 Paged KV | block_size 8/16/32；contiguous vs paged | 相同总 KV 容量及 workload | VRAM、fragmentation、tok/s、OOM/recompute |
| E5 Prefix Cache | off/on；shared prefixes 0/25/50/75% | 同模型、相同 exact token prefix | hit rate、saved prefill、TTFT |
| E6 GPU Optimization | eager / graph batch buckets；Triton kernel off/on | 相同 inputs 和正确性 | CPU dispatch gap、kernel us、ITL、Graph Pool |

**可选扩展 E7：** 1.5B vs 3B 的扩展性；E8：burst/cancellation 的故障恢复与背压；E9：简单 speculative decoding 的接受率、额外显存与净延迟。优先完成 E1—E6，不要让高级特性阻塞主项目。

### 6.2 统计设计和公平性

1. 所有配置使用相同预生成 request trace、相同 tokenizer、模型权重 revision、输入长度分布、采样设置和超时；记录 seed。
2. 先做 20—50 个 warm-up 请求，正式测量建议每配置至少 200 个请求或 120 s（按响应/设备容量调整），**不少于 3 次独立重复**；核心结果建议 5 次。
3. 执行顺序随机化或交替 ABBA，减少热状态、温度、GPU clocks 随顺序漂移带来的偏差；每次配置测温度与显卡频率。
4. 报告均值/中位数、p95/p99、重复运行之间的变异范围；主要比较用 request/run 层级 bootstrap 95% CI，不能把相邻 token 当作独立采样。
5. 同时记录 offered load 与 achieved load；发生大量超时/429 时不能仅用“完成请求的平均延迟”宣称提升。
6. 与 `transformers.generate()` 的比较定义为**模型执行参考线**，不能冒充等价的多请求 server；vLLM 才是可匹配 API、流式、同模型和采样条件的系统级对照。
7. vLLM 与 FlexEngine 的 tokenizer/权重/precision/`max_model_len`/KV 容量/Graph/Prefix/并发限制/负载种子都要公开；配置有不等价之处必须标注，不做夸大宣称。

### 6.3 SLO 选择方式

SLO 是 workload-specific，不直接借用其他 GPU 的毫秒绝对值。先通过 W1—W4 的 FCFS warm-up 找到容量拐点；然后选用**两档明确且一致的实验性门限**（例如按该基线的 TTFT p90、ITL p90 作为宽松门限，按 p75 作为严格门限），冻结门限再比较各策略。门限由预实验确定，不得根据测试策略结果重新调参。

## 7. Profiling 方案（回答“为什么快/慢”）

### 7.1 三层测量体系

| 层次 | 工具 | 标记/采集内容 | 主要问题 |
|---|---|---|---|
| L1 Service | OpenTelemetry / Prometheus / Python monotonic clock / AIPerf | arrival→queue→schedule→prefill→decode→SSE；p99；Goodput | 是队列、网络还是执行在拖慢用户？ |
| L2 GPU Timeline | NVIDIA Nsight Systems (`nsys`) + NVTX + CUDA Events | CPU launch 间隔、kernel 序列、stream、CUDA Graph、同步气泡 | GPU 在等 CPU？prefill 阻塞 decode？ |
| L3 Kernel | NVIDIA Nsight Compute (`ncu`) + PyTorch Profiler | occupancy、memory throughput、DRAM/L2、warp stall、Roofline | Attention/KV gather/RMSNorm 是访存还是计算瓶颈？ |

NVIDIA GPU 性能岗位明确列出 Nsight Systems、Nsight Compute、PyTorch Profiler 与 roofline/speed-of-light 方法 [R1]。Nsight 文档支持 CUDA/NVTX 时间线和 CUDA Graph 不同追踪粒度 [R11]—[R12]。

### 7.2 NVTX 事件定义

建议每个 engine iteration 增加：`sched/select`、`kv/allocate`、`kv/map`、`executor/prefill`、`executor/decode`、`attention/gather`、`sampler/topk`、`stream/flush`。每条事件附加 `batch_size`、`num_prefill_tokens`、`num_decode_tokens`、`kv_blocks_used`、`graph_hit`。

示例（脚本名将在代码仓库中实现）：

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none \
  --cuda-graph-trace=graph -o results/nsys/mixed_decode \
  python -m flexengine.benchmarking.profiling.engine --model Qwen/Qwen2.5-0.5B-Instruct \
  --workload mixed --steps 200 --batch-size 4

ncu --set full --target-processes all \
  -o results/ncu/paged_attention \
  python -m flexengine.benchmarking.profiling.kernel --kernel paged_attention --batch-size 4
```

**采样规则：** `nsys` 只抓稳定的少量代表性 step；`ncu` 更重，单独运行 kernel case；正式 QPS/latency 由不带 profiler 的服务实例测量。不要使用 `ncu` 下的 end-to-end 时间当正式性能结果。

### 7.3 按症状制定优化假设

| 观察 | 可能原因 | 进一步确认 | 对应调整 |
|---|---|---|---|
| GPU Timeline 中 kernel 间空白大 | CPU/Python dispatch overhead | NVTX + CUDA API launch time | CUDA Graph、降低 Python per-step 工作 |
| Prefill 后 decode ITL 尖峰 | 长 prompt 阻塞 | step 中 prefill/decode tokens 及 ITL 对齐 | chunk budget、decode-first |
| 高并发吞吐降低且 VRAM 满 | KV 容量/预占有/内存压力 | block occupancy、preempt、GPU reserved | block size、admission、重算策略 |
| Attention Kernel DRAM traffic 高 | 非连续 KV 访问低效 | NCU memory workload + roofline | block mapping/layout、Triton attention |
| Triton kernel 快但 E2E 不变 | Amdahl 定律；不在 critical path | E2E vs kernel time对齐 | 优化 CPU/调度而非继续打磨小算子 |
| Graph 捕获后反而变慢 | padding、fallback、pool 增长 | graph_hit、padded tokens、VRAM | batch bucket selection + selective graph |

## 8. Benchmark 工具链与目录组织

### 8.1 负载工具选择

优先使用 **NVIDIA AIPerf**，而不是把 GenAI-Perf 作为唯一长期依赖；NVIDIA 在 2026 年将 AIPerf 明确定位为 GenAI-Perf 的后继工具 [R13]。AIPerf 支持并发、Poisson arrivals、输入分布、逐 token latency 与 SLO Goodput，适合 API 系统对照。自研 Python trace replayer 补齐取消、共享前缀、每一步调度事件等特殊测试。

```bash
# 以下命令为计划使用示例，运行前用当前版本 aiperf --help 检查参数
# 要求服务已实现 /v1/completions，且模型别名与 --model 一致
aiperf profile \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --url http://127.0.0.1:8000 \
  --endpoint-type completions \
  --streaming \
  --concurrency 4 \
  --request-count 200

# Poisson 到达率测试
aiperf profile \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --url http://127.0.0.1:8000 \
  --endpoint-type completions \
  --streaming \
  --request-rate 2 \
  --arrival-pattern poisson \
  --benchmark-duration 120
```

**兼容性注意：** AIPerf 的 endpoint-type 枚举与版本可能变化；记录安装版本并以 `aiperf profile --help` 验证。SSE response format、finish_reason 和 token usage 接口可由兼容性测试自动校验。工具应在 vLLM 和 FlexEngine 上使用一致的设置。

### 8.2 推荐仓库结构

```text
src/flexengine/
  src/flexengine/
    api/             # OpenAI-compatible endpoints / SSE
    scheduler/       # policies / admission / lifecycle
    kv_cache/        # blocks / prefix / preemption
    models/          # Qwen2 adapter / position handling
    attention/       # reference / block-table / Triton
    executor/        # eager / cudagraph
    sampling/        # greedy / top-k / RNG state
    observability/   # metrics / tracing / NVTX
  tests/
    unit/            # state machine / allocator / policy
    correctness/     # logits parity / paged attention
    integration/     # multi-request / API / cancellation
    stress/          # OOM / preemption / repeated reuse
  src/flexengine/benchmarking/
    traces/          # seeded prompt distributions
    clients/         # AIPerf configs / custom replay
    micro/           # eager vs graph / attention kernels
    profiling/engine.py
    profiling/kernel.py
  src/flexengine/experiments/
    config/           # immutable scenario configs
    results/          # JSONL / CSV / profiler reports
    analysis/         # statistical summaries and plots
  docs/
    architecture.md
    design_decisions.md
    limitations.md
    final_report.md
  Dockerfile
  pyproject.toml
  .github/workflows/
```

### 8.3 最小可重现 Artifact Contract

每次 run 输出 `run_metadata.json`（git commit、driver、CUDA、torch、transformers、triton、GPU 型号/显存、模型 ID+revision、precision、SLO、随机种子、启动 flags、温度/时钟）以及 `requests.jsonl`（请求 id、长度、arrival、queue、first_token、last_token、finish_reason、tokens、error、SLO）和 `steps.jsonl`（step、scheduled token、batch composition、KV occupancy、graph_hit、GPU event time）。

固定的 `analysis/plots.py` 从 CSV/JSONL 再生所有图，CI 至少运行 CPU scheduler 单测、GPU correctness sanity（有 GPU Runner 时）、文档与格式检查。不要把 3080 Ti 性能门限硬写进没有同型号 GPU 的 GitHub Actions 公共 Runner；使用可选本地 GPU regression job。

## 9. 里程碑、优先级与风险控制

| 周期（建议） | 目标 | Exit Criteria |
|---|---|---|
| Week 1—2 | Reference Decoder + Qwen Adapter | 单请求 greedy、logits parity、pytest 可复现 |
| Week 3—4 | Request lifecycle + Batching + SSE | 同时 2/4 请求、正确取消、流式 API |
| Week 5—6 | Paged KV + Block-table Attention | Block integrity 全过，真实非连续 KV 读取 |
| Week 7—8 | Chunked Prefill + Preemption + Backpressure | Mixed workload 无泄漏，过载恢复有证据 |
| Week 9—10 | SLO-aware + Prefix + E1—E5 | 固定 trace 的消融表 + 初步统计 |
| Week 11 | CUDA Graph / Triton + Nsight | 至少一个 profile-guided optimization 完成闭环 |
| Week 12 | 完整 Benchmark + 最终报告 + Demo | E1—E6、报告和可重复实验命令齐备 |

**进度风险控制：** 若 Week 6 的真实 Paged Attention 未完成，先保留明确命名的 `materialization_baseline`，不把它写成真正 Paged Attention；先保证整条服务路径可运行。CUDA Graph、prefix、spec decoding 都不能取代基础 correctness。若只能完成一项高级优化，选择 profiling-guided CUDA Graph 或 Paged Attention kernel，并提供端到端受控消融。

## 10. 预期图表清单及最终报告模板

至少交付以下图表：

1. **吞吐–并发**：requests/s、output tokens/s，按模型 0.5B/1.5B 分组。
2. **交互性能**：TTFT 与 ITL p50/p95/p99 对比 S1/S2/S3，附成功率。
3. **SLO Goodput–arrival rate**：Poisson 不同请求速率；同图标出 SLO violation rate。
4. **Token-budget Pareto**：不同 batch token limit 的 TTFT P95 vs ITL P95。
5. **KV Cache**：已用 Block、内部碎片、preemption 次数、peak VRAM 随并发/Block Size 的变化。
6. **Prefix Cache**：shared-prefix ratio vs saved prefill tokens / TTFT。
7. **CUDA Graph**：Eager vs Graph 的 step latency、Graph hit rate、Graph private pool overhead。
8. **Nsight Case Study**：优化前后相同范围的 CPU/GPU timeline 截图 + Kernel 指标表。
9. **Correctness/Failure Summary**：C01—C12、F01—F08 通过率/不通过说明。
10. **Limitations**：模型种类、单卡、FP16、对照框架版本、未解决瓶颈及外推限制。

**最终正式报告建议结构：** Abstract → Industry Motivation → Architecture & Scope → Models & GPU Setup → Strategies → Implementation Details → Correctness & Reliability → Benchmark Methodology → Results & Ablations → Profiling Case Study → Threats to Validity → Conclusion → References。没有实测数据之前，Results 部分必须标记 `Pending Measurement`，不得编造速度提升数字。

## 11. Definition of Done 与简历呈现

### 必须满足的 DoD

- [ ] 两个官方 Qwen2.5 Dense Causal LMs 通过单卡完整文本生成。
- [ ] Scheduler、KV Cache、Model Executor 是自己的代码，不是包装其他 runtime。
- [ ] 多请求 continuous batching 和 token-budget chunked prefill 真实可运行。
- [ ] Block Table 被真实 Attention execution 消费，完成非连续 KV 访问。
- [ ] SSE、取消、背压、OOM/preempt、block cleanup 全有自动化测试。
- [ ] 至少 FCFS vs decode-first 的同 trace 对照及一种 SLO-aware 策略实验。
- [ ] AIPerf / 自研 harness 产生 TTFT、ITL、TPOT、P99、Goodput、VRAM。
- [ ] 使用 Nsight Systems 生成一次可审计瓶颈分析；如做 kernel 优化，再用 Nsight Compute 佐证。
- [ ] 交付独立 run metadata、结果数据、图表生成脚本、复现命令、Docker、最终报告。

### Resume bullets（仅在实际完成后使用）

**P0 完成后：**

> Engineered a single-GPU LLM inference runtime in Python/PyTorch with custom continuous-batching scheduler, token-budget chunked prefill, block-table-based paged KV attention, request lifecycle management, and OpenAI-compatible streaming APIs.

**P1/Profiling 完成并有数据后：**

> Designed reproducible inference benchmarks across Qwen2.5 models and mixed arrival workloads; profiled TTFT, P95/P99 inter-token latency, SLO goodput, KV-cache utilization, and GPU execution with Nsight Systems/Compute to guide runtime optimizations.

**CUDA Graph 完成后再补充：**

> Implemented static-bucket CUDA Graph decode with eager fallback and measured CPU dispatch overhead, replay coverage, performance trade-offs, and GPU memory cost under controlled workloads.

## 12. 参考资料与验证来源

[R1] NVIDIA, *Senior Software Engineer – AI Inference Performance*. https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/Senior-Software-Engineer---AI-Inference-Performance_JR2024262

[R2] NVIDIA, *Senior Software Engineer – AI Inference*. https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Senior-Software-Engineer---AI-Inference_JR2016392

[R3] Red Hat, *Senior Machine Learning Engineer (vLLM)*. https://redhat.wd5.myworkdayjobs.com/en-US/jobs/job/Boston/Senior-Machine-Learning-Engineer_R-060246

[R4] vLLM, *Architecture Overview*. https://docs.vllm.ai/en/latest/design/arch_overview/

[R5] vLLM, *Optimization and Tuning*. https://docs.vllm.ai/en/latest/configuration/optimization/

[R6] SGLang, *Unified Radix Cache*. https://www.sglang.io/blog/unified-radix-cache

[R7] Qwen, *Qwen2.5-0.5B-Instruct config*. https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json

[R8] Qwen, *Qwen2.5-1.5B-Instruct config*. https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json

[R9] Qwen, *Qwen2.5-3B-Instruct config*. https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/config.json

[R10] NVIDIA, *Nsight Compute System Requirements / WSL2*. https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/system-requirements.html

[R11] NVIDIA, *Nsight Systems User Guide*. https://docs.nvidia.com/nsight-systems/UserGuide/

[R12] NVIDIA, *Nsight Compute Profiling Guide*. https://docs.nvidia.com/nsight-compute/ProfilingGuide/

[R13] NVIDIA, *AIPerf documentation*. https://docs.nvidia.com/aiperf/reference/command-line-options

[R14] PyTorch, *CUDA semantics / CUDA Graphs*. https://docs.pytorch.org/docs/main/notes/cuda.html

[R15] NVIDIA, *Benchmarking LLM Inference at Scale with AIPerf*, 2026-09-18. https://developer.nvidia.com/blog/benchmarking-llm-inference-at-scale-with-aiperf

---

**报告使用说明：** 本文是可实施 Proposal 与报告结构，不是实验完成报告。执行后将第 6—10 节中的方案、空结果位和风险说明更新为实际结果、置信区间、原始数据路径与 Nsight 证据。所有性能结论必须在目标 RTX 3080 Ti 设备上测得。
