<div align="center">

### 一张 GPU 卡说"不支持"，但它一直在用

</div>

在一台双卡机器上量 P2P：`nvidia-smi topo -p2p`、`p2pBandwidthLatencyTest`、`nvbandwidth`、`cudaDeviceCanAccessPeer` —— 四个工具，四套完全独立的实现，异口同声：不支持。

但实测 all-reduce busbw / H2D bandwidth 的比值是 **0.72**。同一台机器修 NCCL 配置之前，这个比值是 **0.22**。四个工具的答案前后完全没变，因为它们查的是硬件能力标志位，标志位确实没变——但两次的训练扩展效率天差地别，因为变的是**实际的行为**，不是**声明的能力**。

这个"能力 vs 行为"的裂缝，就是下面七个项目要一起解决的问题。

---

#### 七个项目，一条故事线

```text
  nodebench           benchdoctor          tracedoctor          regressiondoctor
  测量层              静态找坑             动态找坑             时间维度找坑
  ──────────          ───────────          ───────────          ────────────────
  跑 benchmark,   ──▶ 扫 benchmark     ──▶ 扫 nsys trace,   ──▶ 对比两次 run,
  测 FLOPS/带宽/      脚本, 抓那些         抓静态分析           按每个指标自己
  PCIe/NCCL,          不崩溃但算错         看不见的             测出来的噪声,
  自带空闲对照组      的测量陷阱           运行时反模式         判断是不是真退化
  ───────────────────────── 这个数字, 能信吗 ─────────────────────────

  fitdoctor           telemetrydoctor      servedoctor
  容量与上限          监控口径             压测方法
  ──────────          ───────────          ───────────
  跑之前先算:     ──▶ 这一列监控数据   ──▶ 这份压测报告
  显存装不装得下,     我能拿它做什么       测的是服务,
  瓶颈在带宽还是      运算: 求均值? 取      还是压测器?
  算力, 上限多少      峰值? 乘以时长?      闭环 vs 开环
  ↑ 本来该是多少      ↑ 能做什么运算       ↑ 是关于谁的
```

前四个项目答的是同一个问题在四个层面的版本：**"这个数字，能信吗？"** 后三个各换了一次方向：第五个问 **"这个数字，本来该是多少？"**，第六个问 **"这个数字，我能拿去做什么运算？"**，第七个把镜头转过来对着测量装置本身，问 **"这个数字，是关于谁的？"**

- **[nodebench](https://github.com/liu-perf/nodebench)** —— 先把测量做对：FLOPS / 显存带宽 / PCIe / NCCL 四项实测，每次跑都自带一组 12 秒的空闲对照组（用来分辨"卡本身的噪声"和"负载引起的变化"），关键指标有 PyTorch 和原生工具链两套独立实现互相打分歧。
- **[benchdoctor](https://github.com/liu-perf/benchdoctor)** —— 把 nodebench 项目里踩过的坑写成静态检查规则，反过来扫 nodebench 自己：第一次跑出 13 处假阳性，逐条修规则、收窄判定条件，最后 13→0。
- **[tracedoctor](https://github.com/liu-perf/tracedoctor)** —— benchdoctor 自己在文档里承认"这几类问题脚本静态扫不出来，只能翻 nsys trace"，tracedoctor 就是把那几条手写发现自动化成规则，动态层面接着找。
- **[regressiondoctor](https://github.com/liu-perf/regressiondoctor)** —— 前三个都在问"这一次测得对不对"，这个问的是"这一次比上一次差，是真的差了吗"。第一版拿固定 5% 当阈值，被一份纯噪声的样例判成了退化；改成拿每个指标自己测出来的噪声当尺子之后，同一份样例判 `ok`，而真正越过噪声地板的下降照样抓得住。误判的那一版没有删，留在 `tests/test_differ_naive.py` 里，那条测试断言的是"它仍然会错"。
- **[fitdoctor](https://github.com/liu-perf/fitdoctor)** —— 换方向：给一份模型 config 和一张卡的实测档案，在跑之前就说显存装不装得下、瓶颈在哪一侧。第一版 KV cache 用了 `num_attention_heads` 而不是 `num_key_value_heads`，在 GQA 模型上偏大整整一个 group size（16 GiB vs 真实 4 GiB），而在 MHA 模型上完全正确——所以只能靠在真卡上分配对账才按得死（误差 ±0.0000%）。而那个验证脚本自己的第一版也是假通过的：`memory_allocated()` 说 8 GiB 卡上成功分配了 8.000 GiB，驱动其实只给了 4.544 GiB，剩下的被 WDDM 静默搬到了主机内存。**"跑了没 OOM"不是一个容量结论。**
- **[telemetrydoctor](https://github.com/liu-perf/telemetrydoctor)** —— 再换一次方向：给定这一列监控数据，我能拿它做什么运算。一条 `nvidia-smi` 采样文件看起来就是张表格，于是人们对它做表格该做的事——求均值、取最大值、速率乘时长——三件事都能跑通、都不报错，而在 GPU 监控数据上都是错的。工具给每一列建「聚合契约」，自动切阶段，**拒绝非法运算并把错误倍数一起给出来**（PCIe 计数器只观测每 1.376 s 中的 20 ms，速率乘时长小 69 倍）。顺手在单卡上实测证明 `utilization.gpu` 是时间口径不是强度口径：两个实际算力差 **978 倍**的负载，报出来的利用率只差 1.7 个百分点。
- **[servedoctor](https://github.com/liu-perf/servedoctor)** —— 前六个看的都是被测系统，这个把镜头转过来对着**测量装置本身**：这份压测报告测的是服务，还是压测器。同一台服务器、同一个配置速率，闭环压测报出的 q99 比开环短 **16.5 倍**，而两者最大值只差 1.8 倍——闭环看见了停顿，只是从来没有足够多的请求暴露在停顿下让它够得着百分位。还有一条纯算术的拒绝：P99 在 90% 置信下要 **299** 个请求才有上界，低于此工具拒绝报数并告诉你这份样本撑得起哪一个。**这是七个里唯一一个不需要 GPU 就能自己复现主结论的**——45 秒，纯 Python 标准库。

七个项目连起来读的顺序：先看 nodebench 怎么测对一个数字，再看 benchdoctor 怎么用工具审查自己的假设，接着看 tracedoctor 怎么把"承认看不见"的边界继续往下啃一层，然后看 regressiondoctor 怎么把这套怀疑推到时间轴上，再看 fitdoctor 怎么在还没跑之前就把这个数字应该落在哪说清楚（以及它怎么用四张真卡把**自己那条**推导规则的边界量出来、然后拒绝对 fp4 使用它），接着看 telemetrydoctor 怎么给监控的每一列划定「能做什么运算」的边界，最后看 servedoctor 怎么把怀疑指向测量装置本身。**后面的项目两次拿真机测量给自己刚写下的规则划了边界**——fitdoctor 拒绝对 fp4 用自己的比值表，telemetrydoctor 把 nodebench 早已定性写下的「利用率不是强度」量成了 978 倍。（这两句原来写的是「回头缩小了前面项目的适用范围」，不成立：那条比值检查 nodebench 里根本没有，而利用率那条 nodebench 自己写过。撤回的原话留在这里。）

七个项目的关键数字汇总在一页上：**[Dashboard](https://liu-perf.github.io/nodebench/dashboard.html)**——P2P 0.22→0.72、benchdoctor 假阳性 13→0、tracedoctor 73x→1.5x、跨工具吻合 0.1%、fitdoctor KV cache 4x、telemetrydoctor 同一利用率读数下算力差 978x、servedoctor 闭环压测尾延迟短 16.5x。

---

#### Writing

知乎专栏：**[推理性能笔记](https://www.zhihu.com/column/c_2084232113705033798)**

- [四个工具都说不支持 P2P，但它一直在用](https://zhuanlan.zhihu.com/p/2084227879668397581) — nodebench
- [KV cache 用量报 90%，可能只装了一半](https://zhuanlan.zhihu.com/p/2084342687113671847) — 给 vLLM 的那个指标补丁
- [我写了个容量规划工具，然后让它拒绝回答最常被问的那个问题](https://zhuanlan.zhihu.com/p/2084644096598087461) — queuebound
- [我写了十二个项目去质问别人的数字，然后发现我自己的 linter 在编数字](https://zhuanlan.zhihu.com/p/2084769509529875010) — drainlag
- [一个 1.35 亿参数的模型，每秒只解码 48 个 token，而 GPU 有 81% 的时间是空的](https://zhuanlan.zhihu.com/p/2085036453206156056) — decodefloor

每条结论都配一条能复现的命令。比如 servedoctor 那条——同一台服务器、同一个配置速率，闭环压测报出的 q99 比开环短 16.5 倍——**你可以自己跑**：45 秒，纯 Python 标准库，不需要 GPU、模型或网络。

<!--
START_SECTION:stats
（如果想加 GitHub 统计卡片/热力图，在这里插入，例如 github-readme-stats）
END_SECTION:stats
-->
