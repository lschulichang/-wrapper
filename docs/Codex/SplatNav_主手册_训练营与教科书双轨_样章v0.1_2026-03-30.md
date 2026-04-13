# 已废弃说明

这份文档是历史草稿，保留仅用于回看写作演进过程，不再作为当前项目的正式主线阅读材料。

从现在开始，请优先阅读 `docs/Codex` 目录下的新主线文档：

1. `00_统一说明_阅读导航与项目总摘要.md`
2. `01_专业术语全解_项目全词典.md`
3. `02_项目全景与边界_从论文到代码到数据流.md`
4. `03_输入输出与一次运行验收_从脚本到结果.md`
5. `04_顺序化快速上手_从第一天到批跑.md`
6. `05_阶段性目标_一周一月三个月.md`

如果你是第一次接手这个项目，不建议再把本文件当成起点。

# Splat-Nav 主手册（训练营 + 教科书双轨）样章 v0.1

> 面向对象：零基础或跨专业转入的师弟（以“机械/车辆背景，代码基础薄弱”为默认画像）  
> 文档目标：在“能理解”的基础上，尽快出成功结果，并且让成功可复现、可解释、可汇报  
> 版本日期：2026-03-30  
> 交付阶段：样章（约 2 万字）+ 全书目录（目标 10 万字）  
> 存放路径：`/home/howard/Splat-Nav/docs/Codex`  

---

## 写在前面：先把话说明白

你不需要一开始就成为“懂全部数学、会改全部算法”的人。你需要先成为“能稳定做成事的人”。

在科研和工程现场，真正有价值的不是“我知道很多术语”，而是下面这四件事：

1. 我能把任务拆开，知道今天应该做什么。
2. 我能把一次运行做完整，留下可复现证据。
3. 我遇到报错不会慌，能按顺序排查。
4. 我能把结果讲清楚：做了什么、为什么这样做、结果怎样、下一步怎么做。

这份主手册的写法会同时满足两条线：

1. 训练营线：每天有任务、有命令、有验收，保证你能尽快跑出成功。
2. 教科书线：每个动作背后有简明原理，保证你不是“照抄命令的操作员”。

你会看到我有时语气严格，因为我们做的是可复现研究，不是“差不多就行”。你也会看到我反复鼓励你，因为你当前的困难是正常的，不是能力问题，而是训练顺序问题。

---

## 第一部分：全书目录（10 万字目标）

说明：本次先交样章。下面是全书的目标结构与字数预算，你确认风格后，我会按此目录扩写到完整版本。

## 卷首：总导言与使用协议（预计 6000 字）

1. 你为什么会卡住：新手在机器人算法项目里的三类常见断点。
2. 这本手册如何使用：7 天主线、14 天补线、21 天加固线。
3. 学习纪律：不越边界、不跳步骤、不省记录。
4. 成功定义：不是“跑过一次”，而是“可复跑 + 可解释 + 可交接”。

## 卷一：项目总览与边界（预计 12000 字）

1. Splat-Nav 在做什么，和你要交付什么。
2. 三层结构：wrapper、upstream、data/runs。
3. 本机边界：账户、磁盘、GPU、CUDA、他人环境。
4. 验收口径：D1/D2/D3 如何从口头目标变成可检查证据。

## 卷二：7 天核心训练营（预计 30000 字）

1. Day1：环境认知与最小成功（健康检查、路径核验、一次命令闭环）。
2. Day2：单次 smoke 跑通与日志解读。
3. Day3：batch 基本实验与图表解释。
4. Day4：overnight 流程与监控脚本理解。
5. Day5：结果汇总、可行率与耗时分析。
6. Day6：代码走读（wrapper 到 upstream 调用链）。
7. Day7：汇报演练与答辩问答。

每一天统一模板：

1. 今日目标。
2. 原理短讲（15-30 分钟）。
3. 实操任务（命令级）。
4. 验收标准（必须产出物）。
5. 失败分支与救援路径。
6. 今日复盘模板。

## 卷三：14/21 天延展训练（预计 12000 字）

1. 第 2 周：稳定复跑、参数敏感性、实验纪律。
2. 第 3 周：小改动实验设计、对照组、结果解释。
3. 何时可以开始改算法，何时只该改实验设计。

## 卷四：代码理解与架构能力（预计 18000 字）

1. wrapper 脚本体系设计思想。
2. Python 入口脚本设计规范。
3. upstream 关键类：`SplatPlan`、`GSplatLoader`、`SafeFlightCorridor`。
4. 如何在不破坏可复现性的前提下重构。
5. 大厂风格代码清单：命名、注释、边界、日志、错误处理、配置管理。

## 卷五：排错实战库（预计 14000 字）

1. 按“现象”索引：无输出、CUDA 不可用、可行率异常、任务中断、路径错乱。
2. 按“层级”排查：系统层、环境层、代码层、数据层、实验设置层。
3. 案例复盘：为何同样命令在不同时间得到不同结果。

## 卷六：汇报与职业化输出（预计 8000 字）

1. 周报模板（导师视角）。
2. 组会汇报模板（技术 + 结论）。
3. 交接模板（给下一个新人）。
4. 面试叙事模板（你如何讲这个项目，体现工程与研究能力）。

---

## 第二部分：样章导学（你这一周到底怎么学）

你只有一个核心学习策略：先把“可运行系统”抓在手里，再逐步把“原理解释”补上。

换句话说：

1. 先有结果，再做理解深化。
2. 先会排错，再谈优化。
3. 先能复现，再谈创新。

请你记住以下三条纪律：

1. 任何实验必须落在 `runs/YYYY-MM-DD_HHMM_topic/`。
2. 任何“我跑过”都必须对应至少一个文件证据（`cmd.txt`、`stdout.log`、JSON 或图）。
3. 任何失败都必须写出“失败类型 + 已排查动作 + 下一步计划”。

很多同学不是输在智力，是输在“没有形成可执行的学习系统”。你只要按系统做事，进步会非常快。

---

## 第三部分：第 0 天预热（正式 Day1 前）

### 0.1 今日目标

1. 搞清楚项目边界，不踩雷。
2. 理解目录结构，不迷路。
3. 建立“结果证据”意识。

### 0.2 原理短讲：为什么先讲边界

你现在使用的是共享机器。共享机器的第一原则不是“快”，而是“可控”。

如果你今天图快去升级系统 CUDA、改别人账户下的环境、或者把大数据写爆系统盘，你后面一周都会在修锅。

所以我们先把边界写死：

1. 不升级系统 CUDA / 驱动。
2. 不动他人账户与路径。
3. 大文件走 `/data/howard`。
4. 你自己的工程入口是 `/home/howard/Splat-Nav`。

### 0.3 目录认知（必须会背）

你眼前这个项目，至少要有下面这张图在脑子里：

1. wrapper：`/home/howard/Splat-Nav`
2. upstream（官方算法树）：`/home/howard/Splat-Nav/splatnav-official`（软链入口）
3. runs 实体：`/home/howard/Splat-Nav/runs -> /data/howard/splatnav/runs`
4. conda：`/data/howard/miniconda3`

你需要在终端里自己验证，而不是“我感觉是这样”：

```bash
cd /home/howard/Splat-Nav
pwd
ls -lah
readlink -f /home/howard/Splat-Nav/splatnav-official
readlink -f /home/howard/Splat-Nav/runs
```

### 0.4 第 0 天验收

你必须能口头回答：

1. `scripts/` 里是算法主体吗？为什么？
2. `SplatPlan` 实现在哪个路径？
3. 为什么 `runs` 要落在 `/data`？
4. 如果你说“我跑过了”，你需要拿出哪三个文件？

### 0.5 第 0 天提交物

在你的 `notes` 里写 10 行即可：

1. 你今天确认的四个路径。
2. 你最担心的三个问题。
3. 你明天 Day1 的计划（不超过 6 行）。


---

## 第四部分：Day1 训练营（环境与最小成功）

### 1.0 今天的硬目标

今天结束前，你必须完成下面三件事：

1. 确认环境可用：`torch.cuda.is_available()` 为 `True`。
2. 确认路径可用：`splatnav-official` 与 `runs` 软链正确。
3. 完成一次“可证明的命令执行”，并落档到 `runs`。

注意：Day1 不求你理解全部算法，求你把“操作系统”搭起来。没有 Day1，后面所有内容都会变成玄学。

### 1.1 原理短讲（20 分钟）：环境为什么是第一生产力

你可能会觉得“我先看代码不行吗”。答案是不行。原因很简单：

1. 代码是静态文本，运行是动态系统。
2. 你的问题 70% 出在环境和路径，不在算法逻辑。
3. 如果环境不稳定，你今天看到的结果，明天会消失。

在大厂项目里，先把“可运行、可复现、可定位问题”的底层能力做扎实，是工程能力的起点。研究也是同理。

### 1.2 今日实操任务 A：基础核验（必须手打）

```bash
cd /home/howard/Splat-Nav
pwd
ls -lah
readlink -f /home/howard/Splat-Nav/splatnav-official
readlink -f /home/howard/Splat-Nav/runs
```

你应看到：

1. 当前目录是 `/home/howard/Splat-Nav`。
2. `splatnav-official` 最终指向 `/data/howard/repos/splatnav-official`。
3. `runs` 最终指向 `/data/howard/splatnav/runs`。

如果这三条有一条不成立，先停，不要继续。

### 1.3 今日实操任务 B：Conda 与 GPU 健康检查

```bash
source /data/howard/miniconda3/etc/profile.d/conda.sh
conda activate splatnav
env -u LD_LIBRARY_PATH python - <<'PY'
import torch
print('torch=', torch.__version__)
print('cuda=', torch.version.cuda)
print('available=', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu=', torch.cuda.get_device_name(0))
PY
```

你应看到 `available=True`。若为 `False`，你按下面排查顺序，不得跳步：

1. 先看 `nvidia-smi` 是否正常。
2. 再确认是否在 `splatnav` 环境。
3. 再确认是否加了 `env -u LD_LIBRARY_PATH`。
4. 最后才考虑执行 `scripts/fix_torch_cuda.sh`。

### 1.4 今日实操任务 C：用 tmux 启动“可追溯任务”

你需要形成习惯：长任务一律进 tmux，并写进 `runs`。这是你以后不丢进度的护城河。

```bash
cd /home/howard/Splat-Nav
bash scripts/run_in_tmux.sh day1_probe day1_probe "bash -lc 'source /data/howard/miniconda3/etc/profile.d/conda.sh && conda activate splatnav && env -u LD_LIBRARY_PATH python -c \"import torch;print(torch.cuda.is_available())\"'"
```

执行后，你需要在 `runs` 中找到对应目录，检查：

1. `cmd.txt`：命令是否被完整记录。
2. `stdout.log`：输出里是否有 `True`。
3. `env.txt`：是否记录了 GPU 与上游 commit。

### 1.5 今日验收标准（必须全部满足）

1. 你能给出一个今天新产生的 `runs/<timestamp>_day1_probe` 路径。
2. 你能截图或口述 `stdout.log` 中 `True`。
3. 你能解释为什么要 `env -u LD_LIBRARY_PATH`。
4. 你能说清 `scripts` 与 `splatnav-official` 的职责边界。

### 1.6 失败分支与救援策略

#### 现象 1：`python: command not found`

原因通常是没有激活 conda 环境。处理：

1. `source /data/howard/miniconda3/etc/profile.d/conda.sh`
2. `conda activate splatnav`
3. `which python`

#### 现象 2：`ImportError` 指向 CUDA 相关 `.so`

原因通常是库路径污染或 wheel 版本不匹配。处理：

1. 先加 `env -u LD_LIBRARY_PATH` 复测。
2. 再跑 `scripts/env_health_check.sh splatnav`。
3. 仍不行再执行 `scripts/fix_torch_cuda.sh splatnav`。

#### 现象 3：tmux 命令执行但没找到 runs 目录

可能是命令写错或权限异常。处理：

1. `ls -lah /home/howard/Splat-Nav/runs | tail`
2. 查 `scripts/run_in_tmux.sh` 中 topic 参数是否为空。
3. 检查当前用户是否为 `howard`。

### 1.7 今日复盘模板（你要写，不要偷懒）

请按以下结构写 15-20 行：

1. 今天我真正跑通了什么（写路径，不写空话）。
2. 今天我最关键的一个理解是什么。
3. 今天我犯的两个低级错误是什么。
4. 如果明天我再来，我最先做哪三步。

---

## 第五部分：Day2 训练营（第一次 smoke 全链路）

### 2.0 今天的硬目标

1. 独立完成一次 smoke 运行。
2. 拿到 `splatplan_smoke_output.json`。
3. 能读懂 JSON 里的关键字段，并说出“成功/失败”的判定。

### 2.1 原理短讲（25 分钟）：smoke 的价值

smoke 不是“小儿科”，而是系统联调的最小闭环。

你把 smoke 做扎实，意味着：

1. 环境是通的。
2. 路径是通的。
3. 上游算法类可以被正确 import。
4. 输入输出链路是可证明的。

如果 smoke 都不稳，batch 和 overnight 只是扩大混乱。

### 2.2 Day2 必备知识：一次 smoke 的输入与输出

输入至少有三类：

1. `scene`：场景名（如 `old_union`）。
2. `config.yml`：场景模型配置路径。
3. `method`：规划方法（如 `sfc-1` 或 `splatplan`）。

输出至少有三类：

1. `splatplan_smoke_output.json`：核心结果。
2. `stdout.log`：运行过程与报错线索。
3. `env.txt`：当时环境快照。

### 2.3 Day2 实操任务 A：找到可用 config

```bash
UPSTREAM=$(readlink -f /home/howard/Splat-Nav/splatnav-official)
find "$UPSTREAM/outputs" -type f -name config.yml | head -n 20
```

从输出中选一个真实存在的 `config.yml`，不要凭记忆拼路径。

### 2.4 Day2 实操任务 B：执行 smoke

```bash
cd /home/howard/Splat-Nav
bash scripts/run_splatplan_smoke.sh old_union /绝对路径/到/config.yml sfc-1
```

运行结束后，脚本会打印本次 `run_dir`。你必须进入该目录核查：

```bash
cd /home/howard/Splat-Nav/runs/你的run目录
ls -lah
```

必须有：

1. `splatplan_smoke_output.json`
2. `stdout.log`
3. `cmd.txt`
4. `env.txt`

### 2.5 Day2 实操任务 C：读 JSON（不许只看文件存在）

你要至少回答以下问题：

1. 顶层 `meta` 里记录了什么。
2. `plan_time_sec` 大概是多少。
3. `output` 里是否有 `feasible`。
4. 若 `feasible=false`，这是“程序崩溃”还是“规划不可行返回”。

建议快速查看方法：

```bash
python - <<'PY'
import json
p='/home/howard/Splat-Nav/runs/你的run目录/splatplan_smoke_output.json'
obj=json.load(open(p,'r'))
print('keys=',list(obj.keys()))
print('plan_time_sec=',obj.get('plan_time_sec'))
out=obj.get('output',{})
if isinstance(out,dict):
    print('output_keys=',list(out.keys())[:20])
    print('feasible=',out.get('feasible'))
PY
```

### 2.6 Day2 验收标准

你必须提交以下四项证据：

1. 一个真实 run 路径。
2. JSON 中 `plan_time_sec` 与 `feasible` 的值。
3. 一段你自己写的结论（不少于 80 字）：这次运行意味着什么。
4. 一条你下一次想改的参数（如 method 或 scene），并说明理由。

### 2.7 教授式点评：你现在最容易犯的两个误区

误区一：只要脚本结束就算成功。  
纠正：必须看输出文件内容与关键字段，不是看命令行有没有报错。

误区二：把 `feasible=false` 当成自己操作失败。  
纠正：`feasible=false` 在很多情况下是有效实验结果，重点是你要证明程序正常走完且结果可解释。

### 2.8 Day2 复盘要求

写一段“结构化复盘”：

1. 我今天输入了什么。
2. 程序做了什么。
3. 输出给了我什么。
4. 我对结果的解释是什么。
5. 我明天准备做什么更严格的实验。


---

## 第六部分：Day3 训练营（batch 与图表）

### 3.0 今天的硬目标

1. 独立运行一次小规模 batch。
2. 读懂 `batch_results.json` 的主结构。
3. 生成并解释至少两张图（可行率与耗时）。

### 3.1 原理短讲（30 分钟）：为什么需要 batch，而不是只看一次 smoke

一次 smoke 的本质是“单点观测”。

单点观测有两个问题：

1. 容易受随机性影响。
2. 不能反映方法在不同起终点上的总体行为。

batch 的作用就是把单点变成分布。你关注的不再是“这次行不行”，而是：

1. 可行率如何（成功概率）。
2. 规划耗时如何（效率）。
3. 方法之间是否有稳定差异。

你记住一句话：研究结论来自统计，不来自一次好运气。

### 3.2 Day3 实操任务 A：跑一个可控规模 batch

建议先小规模，避免你第一天就跑超长任务。

```bash
cd /home/howard/Splat-Nav
source /data/howard/miniconda3/etc/profile.d/conda.sh
conda activate splatnav

CONFIG=/绝对路径/到/config.yml
OUT=/home/howard/Splat-Nav/runs/$(date +%Y-%m-%d_%H%M)_day3_batch_smoke/batch_results.json
mkdir -p "$(dirname "$OUT")"

env -u LD_LIBRARY_PATH python scripts/run_official_style_batch.py \
  --config "$CONFIG" \
  --scene old_union \
  --methods sfc-1,sfc-2,sfc-3,sfc-4,splatplan \
  --trials 30 \
  --seed 1 \
  --keep_traj_limit 5 \
  --output "$OUT"
```

注意：`trials=30` 仅用于教学样章。正式比较时按你们组内口径扩大。

### 3.3 Day3 实操任务 B：出图

```bash
FIG_DIR="$(dirname "$OUT")/figs"
env -u LD_LIBRARY_PATH python scripts/visualize_batch_results.py \
  --input "$OUT" \
  --output_dir "$FIG_DIR"
```

你应看到至少这些文件：

1. `feasible_rate.png`
2. `plan_time_boxplot.png`
3. `summary.md`
4. （可选）轨迹叠加图

### 3.4 Day3 实操任务 C：写结构化结论

你不要只说“跑完了”。你应写成下面这种结论模板：

1. 本次实验设置：场景、方法、trials、seed。
2. 核心指标：每种方法的 `feasible_rate` 与 `avg_plan_time_sec`。
3. 初步观察：哪种方法可行率高、哪种方法速度快。
4. 保留意见：样本量是否足够、是否需要换 seed 验证。

这才叫研究记录。

### 3.5 Day3 验收标准

1. 你能提供 `batch_results.json` 绝对路径。
2. 你能提供 `figs` 目录路径。
3. 你能口头解释可行率图与箱线图分别表达什么。
4. 你能写出一段不少于 150 字的“科学上谨慎”的结论。

### 3.6 失败分支与救援策略

#### 现象 1：运行很慢，像卡住一样

先看是否真的卡住：

```bash
nvidia-smi
ps -ef | rg -i 'run_official_style_batch.py|python'
```

如果 GPU/CPU 都在工作，就不是卡死，是计算中。

#### 现象 2：`batch_results.json` 未生成

先看命令输出尾部；再核查 `--output` 目录是否存在写权限；再看是否中途 import 报错。

#### 现象 3：图脚本报错

常见原因：输入 JSON 结构不完整或字段缺失。先用 `python -m json.tool` 检查 JSON 完整性。

### 3.7 教授式点评：你从今天开始要有“变量控制”意识

你要尽快养成实验习惯：

1. 每次只改一个核心变量。
2. 其余条件保持不变。
3. 所有变化写入 `notes`。

否则你会在第三周完全不知道“到底是哪一步导致结果变化”。

---

## 第七部分：Day4 训练营（overnight 与自动汇总）

### 4.0 今天的硬目标

1. 理解 overnight 脚本的工作循环。
2. 能在 tmux 中安全启动与监控任务。
3. 能在任务结束后定位汇总报告目录。

### 4.1 原理短讲（25 分钟）：为什么长任务一定要工程化

长任务的最大敌人不是算法，是过程不可控。

典型风险：

1. SSH 断开任务消失。
2. 跑了 8 小时却没有日志可追溯。
3. 任务中断后不知道跑到第几轮。
4. 第二天只剩“我感觉昨晚跑过”。

所以你必须把长任务交给三件套：

1. tmux 会话。
2. runs 目录记录。
3. 汇总脚本输出。

### 4.2 Day4 实操任务 A：理解 `run_overnight_12h.sh`

你必须读懂脚本结构，不要求你改动，但要求你会解释：

1. 时间上限如何控制。
2. `iter` 如何增长。
3. 每轮输出落在哪里。
4. 失败后为什么继续下一轮。

建议你边看边记“伪代码”：

1. 初始化参数。
2. while 未到时间上限。
3. 创建 run 目录。
4. 运行 batch。
5. 运行可视化。
6. iter++。
7. 结束时写 completed。

### 4.3 Day4 实操任务 B：安全启动一个短版 overnight（教学）

为了教学，先跑 0.5 小时或 1 小时，确认流程。

```bash
cd /home/howard/Splat-Nav
bash scripts/run_in_tmux.sh overnight_demo overnight_demo \
"bash -lc 'cd /data/howard/splatnav/datasets/splatnav_min && bash /home/howard/Splat-Nav/scripts/run_overnight_12h.sh old_union /data/howard/splatnav/datasets/splatnav_min/splatnav11202024_images2/config_images2.yml 1 60 sfc-1,sfc-2,sfc-3,sfc-4,splatplan 5'"
```

### 4.4 Day4 实操任务 C：监控任务

你要会三个层次的监控：

1. 会话层：tmux 窗口是否存活。
2. 进程层：python 是否在跑。
3. 资源层：GPU 是否有利用率。

常用命令：

```bash
/data/howard/miniconda3/bin/tmux ls
ps -ef | rg -i 'run_overnight_12h.sh|run_official_style_batch.py|python'
nvidia-smi
```

### 4.5 Day4 实操任务 D：结束后找汇总

任务结束后，你要能定位这几类目录：

1. `*_formal_batch_old_union_iterXX`：每一轮 batch 结果。
2. `*_overnight_12h_formal`：主任务日志。
3. `*_overnight_report_old_union`：汇总报告目录。

并能打开：

1. `overnight_summary.csv`
2. `overnight_summary.md`
3. `feasible_rate_over_iters.png`
4. `avg_plan_time_over_iters.png`

### 4.6 Day4 验收标准

1. 你能画出 overnight 的循环流程图（文字版也可以）。
2. 你能给出一次运行的主日志路径。
3. 你能给出一次汇总报告路径。
4. 你能解释“为什么会出现 watchdog 重启后多出一个空 iter 目录”。

### 4.7 教授式点评：你现在进入“研究工程分水岭”

很多同学到 Day4 会分成两类：

1. 只会点运行，不会看过程。
2. 会看过程，会解释异常。

你要成为第二类。工程能力最可贵的部分，就是你能控制过程，而不是只看结果。

---

## 第八部分：Day5 训练营（结果解释与实验纪律）

### 5.0 今天的硬目标

1. 对 overnight 汇总做一次规范解读。
2. 写出“可行率-耗时”双维结论。
3. 输出一页可给导师看的阶段报告草稿。

### 5.1 原理短讲（30 分钟）：指标解释中的三个陷阱

陷阱一：只看可行率，不看耗时。  
陷阱二：只看平均值，不看分布。  
陷阱三：只看一个 seed，就下强结论。

你要形成双维思维：

1. 可行率代表“成功概率”。
2. 耗时代表“效率成本”。

高可行率但特别慢，和低可行率但很快，都不一定是好方案，要结合目标场景判断。

### 5.2 Day5 实操任务 A：从 CSV 提取方法均值

你至少要能算出每种方法的平均可行率和平均耗时。可以用 Python 或表格工具。

建议模板：

1. 方法名。
2. 平均 `feasible_rate`。
3. 平均 `avg_plan_time_sec`。
4. 你对该方法定位的一句话判断。

### 5.3 Day5 实操任务 B：写“谨慎结论”

范式如下（请你替换数字）：

“在 `old_union` 场景、固定配置与给定试验规模下，`splatplan` 在可行率上表现最优，但耗时高于部分 SFC 变体。该结论目前基于单场景与有限 seed，建议在更多场景与多次重复下验证稳定性后再作普适判断。”

这叫科研表达。你不是在直播带货，不需要绝对化措辞。

### 5.4 Day5 验收标准

1. 一张方法对比表（至少 5 种方法）。
2. 一段 150-300 字结论，含“结论 + 限制 + 下一步”。
3. 一条“明天准备读哪段代码、为什么”的计划。


---

## 第九部分：Day6 训练营（代码走读与架构意识）

### 6.0 今天的硬目标

1. 说清 wrapper 与 upstream 的调用关系。
2. 能沿着一次 smoke 调用链找到关键类。
3. 输出一份“读码笔记”，而不是只贴路径。

### 6.1 原理短讲（30 分钟）：为什么你必须读代码

你已经会跑了，但“会跑”不等于“会做研究”。

研究和工程能力的分水岭在于：

1. 你是否知道结果从哪来。
2. 你是否能解释改动会影响哪一层。
3. 你是否能在必要时做最小重构。

如果你不读代码，你会永远停在“命令收藏家”阶段。你的上限会很低。

### 6.2 Day6 读码主线（建议 2.5 小时）

请按顺序阅读，不要跳：

1. `/home/howard/Splat-Nav/scripts/run_splatplan_smoke.sh`
2. `/home/howard/Splat-Nav/scripts/smoke_splatplan.py`
3. `/home/howard/Splat-Nav/splatnav-official/splatplan/splatplan.py`
4. `/home/howard/Splat-Nav/splatnav-official/splat/splat_utils.py`
5. `/home/howard/Splat-Nav/splatnav-official/SFC/corridor_utils.py`

### 6.3 Day6 关键问题（你必须能答）

1. smoke 命令在哪一步创建 run 目录？
2. `smoke_splatplan.py` 如何定位 upstream 根？
3. `method=splatplan` 与 `method=sfc-1` 在构造阶段有何分叉？
4. `generate_path` 的输入是什么，输出是什么？
5. 输出 JSON 为什么要做序列化转换（tensor/numpy -> python 基础类型）？

### 6.4 Day6 实操任务：画“调用链图”

请用文字或 mermaid 画出最小调用链：

```text
run_splatplan_smoke.sh
  -> smoke_splatplan.py main
    -> build_planner
      -> GSplatLoader
      -> SplatPlan 或 SafeFlightCorridor
    -> planner.generate_path
    -> JSON 写盘
```

你要在每一层旁边写一句“这层在干什么”。

### 6.5 Day6 验收标准

1. 一份不少于 300 字的读码笔记。
2. 一张调用链图。
3. 三个“我现在能改、但不会破坏主链路”的小改动建议。

### 6.6 教授式点评：你要学会“改动边界”

你从今天开始要区分三类改动：

1. 运行参数改动：低风险，优先尝试。
2. wrapper 逻辑改动：中风险，需要回归验证。
3. upstream 算法改动：高风险，需要实验设计与对照。

科研不是“改得越多越厉害”，而是“在可控边界里做有效变化”。

---

## 第十部分：Day7 训练营（汇报能力与一周结题）

### 7.0 今天的硬目标

1. 完成一份可给导师看的周总结。
2. 进行 10 分钟口头讲解演练。
3. 明确下周计划（保守方案与进阶方案）。

### 7.1 原理短讲（20 分钟）：为什么汇报能力是核心能力

你未来无论去科研院所还是互联网大厂，都不会只评价“你会不会写代码”。

别人会看你是否能：

1. 讲清问题。
2. 讲清方法。
3. 讲清证据。
4. 讲清限制和下一步。

这四条做到了，你在团队里就不是“执行者”，而是“可托付的人”。

### 7.2 Day7 实操任务 A：周总结模板（可直接填）

1. 本周目标（最多 3 条）。
2. 本周完成（附 runs 路径）。
3. 关键结果（附 JSON 或图路径）。
4. 遇到问题与处理动作。
5. 当前风险与规避策略。
6. 下周计划（保守版 + 进阶版）。

### 7.3 Day7 实操任务 B：10 分钟讲解提纲

建议结构如下：

1. 第 1 分钟：项目目标与本周范围。
2. 第 2-4 分钟：环境与流程打通情况。
3. 第 5-7 分钟：batch/overnight 核心结果。
4. 第 8-9 分钟：问题与限制。
5. 第 10 分钟：下周计划。

### 7.4 Day7 实操任务 C：模拟问答（你先自问自答）

你至少回答这 8 个问题：

1. 你为什么说自己“跑通了”，证据在哪？
2. `feasible=false` 在你的实验里如何解释？
3. 为什么 runs 放在 `/data`？
4. wrapper 与 upstream 的关系是什么？
5. 你如何保证别人可复现你的结果？
6. 你本周最大失败是什么，怎么修复的？
7. 你准备怎么做下一步实验变量控制？
8. 如果现在让你交接给另一个新人，你会给哪三个文件？

### 7.5 Day7 验收标准

1. 一份周总结（文字版）。
2. 一份 10 分钟讲稿提纲。
3. 一份“下周计划看板”（任务、负责人、完成定义、风险）。

### 7.6 一周结题判定（通过/不通过）

通过条件：

1. 至少一条 smoke 证据链完整。
2. 至少一条 batch 证据链完整。
3. 能解释核心指标，不把报错与不可行混为一谈。
4. 有结构化复盘，不是口头记忆。

不通过常见原因：

1. 命令跑了，但没有可追溯证据。
2. 输出文件有了，但本人解释不清。
3. 报错只会重试，不会排查。

结论：不通过不是失败，是提醒你“训练顺序要回到基础环节”。

---

## 第十一部分：14 天与 21 天延展路线（保守优先，可拉长战线）

### 11.0 为什么要延展

一周完成的是“入门级可执行能力”。

真正能独立带小项目，通常需要至少 2-3 周，把以下能力补齐：

1. 稳定复跑能力。
2. 基础实验设计能力。
3. 小规模代码改造能力。

### 11.1 第 2 周（Day8-Day14）目标

1. 做 3 组对照实验（至少控制 1 个变量）。
2. 每组实验都提供 JSON + 图 + 150 字解释。
3. 修订脚本注释与文档一致性（只在 wrapper 层）。
4. 输出一份“我的排错字典”（至少 10 个情景）。

推荐日程：

1. Day8-9：多 seed 对照。
2. Day10-11：不同 method 组合对照。
3. Day12：整理图表和结果解释。
4. Day13：补文档和脚本注释。
5. Day14：第二次汇报演练。

### 11.2 第 3 周（Day15-Day21）目标

1. 选择一个小改动点（参数策略、统计逻辑、可视化增强三选一）。
2. 做改动前后对照实验。
3. 写“改动影响分析”，说明收益与副作用。

你必须遵守一个纪律：

1. 改动范围小。
2. 实验对照清楚。
3. 回滚路径明确。

### 11.3 延展阶段验收清单

1. 能稳定重复得到同类结果。
2. 能解释结果变化来自哪个变量。
3. 能给出下一阶段最值得投入的方向。

---

## 第十二部分：样章中的教科书补充（核心概念精讲）

### 12.1 `feasible` 与“程序成功”的区别

这是新手最容易混淆的概念。

1. 程序成功：程序按设计走完，正常输出结果文件。
2. feasible 为真：在当前约束下，规划找到可行轨迹。
3. feasible 为假：程序成功运行，但规划判定该样本不可行。

因此，`feasible=false` 常常是有效实验数据，而不是“你把环境搞坏了”。

### 12.2 为什么 `LD_LIBRARY_PATH` 会影响结果

简化理解：

1. Python 里的深度学习库依赖底层动态库。
2. 系统会按搜索路径找这些库。
3. 如果路径里混入不匹配版本，就会出现诡异报错。

项目里常见策略是：

1. 用 conda 环境隔离。
2. 运行关键 python 时临时去掉 `LD_LIBRARY_PATH`。

这不是“玄学技巧”，是动态链接机制的工程化应用。

### 12.3 为什么强调 runs 命名规范

命名规范的价值不是“好看”，而是团队协作下的可追踪性。

当你一个月后回看实验，如果目录名没有时间与主题，你几乎不可能快速定位当时做了什么。

推荐结构：

`YYYY-MM-DD_HHMM_主题`

主题建议包含：

1. 任务类型（smoke/batch/overnight/report）。
2. 场景信息（如 old_union）。
3. 特殊设置（如 seed 或方法集）。

### 12.4 为什么先做 wrapper 层，再碰 upstream 算法

因为 wrapper 层是“实验控制台”。

你在控制台都不稳定时，直接改算法会造成双重不确定性：

1. 你不知道错误来自环境还是算法。
2. 你不知道结果变化来自改动还是噪声。

先把控制台稳定，再改算法，是成熟团队的共识。


---

## 第十三部分：排错十讲（样章版）

说明：这一节不是把所有错误背下来，而是训练你“先分类，再定位”的思维。

### 13.1 排错总框架（先背这个）

面对任何问题，你先问自己四个问题：

1. 这是环境问题、路径问题、数据问题还是算法结果问题？
2. 这个问题可复现吗？复现条件是什么？
3. 我已经验证了哪几步？证据文件是什么？
4. 我下一步动作是否最小且可回滚？

### 13.2 十个高频情景与第一动作

1. 没有输出文件。  
第一动作：看 `stdout.log` 尾部 80 行，不要先重跑。

2. 输出文件有，但 JSON 打不开。  
第一动作：`python -m json.tool 文件路径` 验证完整性。

3. `torch.cuda.is_available()` 为 `False`。  
第一动作：`env -u LD_LIBRARY_PATH` 复测。

4. `nvidia-smi` 正常但程序报 CUDA 库错误。  
第一动作：确认 conda 环境 + 动态库路径。

5. 批跑很慢。  
第一动作：核对 `trials`、方法数、GPU 是否被占用。

6. 可行率异常低。  
第一动作：先做小样本 smoke，看是否普遍不可行还是个别样本。

7. overnight 停止。  
第一动作：检查主日志是否 completed，再看 watchdog 与 post-report 日志。

8. 汇总报告没有生成。  
第一动作：检查 post 脚本匹配模式是否与实际命令一致。

9. 找不到 config。  
第一动作：在 upstream 的 outputs 下 `find`，不要手拼路径。

10. 结果与师兄不一致。  
第一动作：对齐 commit、config、method、trials、seed、设备信息。

### 13.3 一条铁律：先保留现场，再做动作

你在排错时最不该做的一件事是“立刻重复覆盖现场”。

正确做法：

1. 先复制出 `stdout.log` 与关键 JSON。
2. 先记录当前命令与环境。
3. 再做最小改动验证。

这和刑侦一样，先保全现场，再做推理。

### 13.4 你给师兄/导师的求助格式（标准版）

请按以下格式发，拒绝“哥我又不行了”：

1. 现象：一句话。
2. 命令：完整可复制。
3. 运行路径：`runs/...`。
4. 错误片段：`stdout.log` 尾部 80 行。
5. 你已做的两步排查。
6. 你判断最可能的两个原因。

当你这样求助时，别人会愿意帮你，因为你在共同承担问题，而不是把问题丢给别人。

---

## 第十四部分：大厂规范导向（样章版）

### 14.1 你现在就要养成的 8 条代码规范

1. 路径不写魔法字符串，优先统一入口变量。
2. 命令行参数要有默认值和帮助信息。
3. 错误信息要能指导下一步动作。
4. 日志要写关键上下文，不写废话。
5. 每个脚本只做一件核心事。
6. 改动前后都要可回归验证。
7. 文档与脚本保持同步。
8. 不在共享环境做破坏性操作。

### 14.2 注释规范（你应采用的风格）

好注释不是翻译代码，而是解释“为什么这样做”。

坏注释示例：

1. `# 设置变量 x`

好注释示例：

1. `# 使用数据盘路径，避免系统盘空间被 runs 持续占满`

### 14.3 你可以尝试的第一个工程化小改动

在不改 upstream 算法的前提下，你可以做一个“参数配置化”小改动，例如：

1. 把 overnight 关键参数抽到一个配置文件。
2. 主脚本读取配置并回显。
3. 运行目录里保存本次配置快照。

这类改动有三个优点：

1. 风险可控。
2. 收益清晰。
3. 能训练你的工程思维。

### 14.4 你暂时不要碰的改动

1. 不要在未建立基线前改求解器核心参数并宣称“算法提升”。
2. 不要在没有对照实验时改多个变量。
3. 不要在没有回滚方案时改公共脚本核心逻辑。

你需要学会克制。克制是成熟工程师和研究者的共同特征。

---

## 第十五部分：附录（样章版）

### 附录 A：一周最小命令清单

#### A.1 环境与路径

```bash
cd /home/howard/Splat-Nav
readlink -f /home/howard/Splat-Nav/splatnav-official
readlink -f /home/howard/Splat-Nav/runs
source /data/howard/miniconda3/etc/profile.d/conda.sh
conda activate splatnav
env -u LD_LIBRARY_PATH python -c "import torch;print(torch.cuda.is_available())"
```

#### A.2 smoke

```bash
bash scripts/run_splatplan_smoke.sh old_union /绝对路径/config.yml sfc-1
```

#### A.3 batch

```bash
env -u LD_LIBRARY_PATH python scripts/run_official_style_batch.py \
  --config /绝对路径/config.yml \
  --scene old_union \
  --methods sfc-1,sfc-2,sfc-3,sfc-4,splatplan \
  --trials 30 \
  --seed 1 \
  --keep_traj_limit 5 \
  --output /home/howard/Splat-Nav/runs/时间戳_topic/batch_results.json
```

#### A.4 可视化

```bash
env -u LD_LIBRARY_PATH python scripts/visualize_batch_results.py \
  --input /home/howard/Splat-Nav/runs/时间戳_topic/batch_results.json \
  --output_dir /home/howard/Splat-Nav/runs/时间戳_topic/figs
```

### 附录 B：一周训练评分量表（导师/师兄可直接用）

#### B.1 评分维度

1. 执行力（30 分）：能否按计划完成任务并产出证据。
2. 理解力（25 分）：能否解释关键概念与指标。
3. 排错力（25 分）：能否按步骤定位问题并记录。
4. 表达力（20 分）：能否结构化汇报结果与下一步。

#### B.2 等级定义

1. A（85-100）：可独立推进并可辅助新人。
2. B（70-84）：可独立完成主线任务，偶尔需要指导。
3. C（60-69）：能完成部分任务，但证据链与解释能力不足。
4. D（<60）：执行与理解均不稳定，需回炉基础训练。

### 附录 C：你这一周应形成的 10 个“肌肉记忆”

1. 任何任务先确认路径。
2. 任何运行先确认环境。
3. 任何长任务都进 tmux。
4. 任何结果都落 runs 并留痕。
5. 任何错误先看日志再行动。
6. 任何结论都附证据与限制。
7. 任何改动都控制变量。
8. 任何沟通都结构化表达。
9. 任何脚本改动都可回滚。
10. 任何阶段都先稳后快。

### 附录 D：给师弟的一段话（请你反复看）

你现在的焦虑不是因为你笨，而是因为你同时面对了“新环境 + 新工具 + 新任务 + 新评价标准”。

把这件事拆开，你会发现每天真正要做的事并不多。你只要做到：

1. 今天把今天的任务做完。
2. 今天把今天的证据留下。
3. 今天把今天的问题写清。

一周以后，你就会明显比现在稳很多。两三周以后，你会开始具备“能独立带一段任务”的能力。

这就是我们要的结果：你不是被项目推着跑，而是你在掌控项目。

---

## 样章收束与下一步

本样章已经给出：

1. 全书 10 万字目录（结构与字数预算）。
2. 7 天训练营主线（Day0-Day7）。
3. 14/21 天延展方案。
4. 教科书式概念补充。
5. 排错与规范附录模板。

你确认风格后，我下一步将做两件事：

1. 按同样风格扩写到完整 10 万字（单文件，不拆）。
2. 在完整版中加入“逐行读码样例”“导师问答库（50 题）”“实操作业答案样例”。
