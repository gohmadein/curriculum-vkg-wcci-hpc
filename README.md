# CurriculumAgent + VKG + TR：118-bus、两个训练种子、Slurm提交包

目标：补齐论文大表最后的118-bus CurriculumAgent行。**这是适配版训练，不是SMAAC换名，也不是原论文环境的逐项复现。**

## 实验定义

| 列 | 方法 | 模型 |
|---|---|---|
| A | CurriculumAgent原生部署搜索，40次simulate/步 | 每种子一份Senior |
| B | A的前30次未满意时，用最后10次搜索VKG候选 | 与A同权重 |
| D | B +断线端子重写 | 与A同权重 |
| C | A +断线端子重写 | 与A同权重 |

固定训练种子0、1，**只训练2个共享Senior，随后8组评估×100条test=800局**。B/D抽样种子固定为对应训练种子，不冒充额外的独立训练种子。不按test选种子、checkpoint或阈值。

### 训练链路和适配范围

1. **Teacher产物复用**：使用现有、非VKG、训练集产生的387个set_bus动作库，四格共同使用；不重新从头跑官方Teacher动作发现。库SHA随冻结配置校验。
2. **Tutor重新生成**：官方curriculumagent1.1.2的GeneralTutor，覆盖全部1507train，每条完整回放一次，记录硬标签及完整4295维观察。原生阈值0.925/0.999；离线教师扫描预算放开并独立计数，不应用于线上评估。
3. **Junior重新训练**：原生四层1000单元ReLU、两处0.25dropout、NLLLoss和Adam5e-4；纯PyTorch循环移植，固定100epochs、batch256，不按验证分数挑epoch。
4. **Senior重新训练**：原生Torch actor/value模型、原生Senior动作执行/安全代打/奖励折叠方法，通过原生RLlib2.5.1 PPO训练；50iterations，batch4000、minibatch128、30轮SGD、lr5e-5、gamma0.99。约200k高层采样步，精确数量以RLlib记录为准。
5. 导出最后一轮policy.npz，使用训练状态核对Torch与NumPy输出误差，再用公共eval_baselines.py完成四格评估。

Senior保留上游softmax输出及终止/截断惯例，不静默修改为其他算法。训练时的重连试算也复用部署端缓存/40次预算代理，耗尽后候选视为无效。部署topo=False与现有case14部署行一致；不使用额外观察归一化。原生PPO_Reward作为训练附加奖励，冻结Redisp/L2RPN与物理参数不改。

**118节点VKG只筛选基线同一份387动作库，不扩充动作集合。** 候选库加入空动作仅供schema表示，策略仍为387类。原生重连/复合动作分支保留，TR仅改写候选中断线端子；A中已有上游机制不人为移除。没有新增随机候选R对照，不能将所有接入差异一概解释为纯知识推理贡献。

## 为什么使用两个Python环境

仿真和评估继续用你已安装的SMAAC **Python3.13/Grid2Op1.12.5/NumPy2** 环境，不降级它。
原生RLlib2.5.1需要旧Python/NumPy，因此另建 **Python3.10/NumPy1.23.5/Torch2.8.0** 训练环境。两个环境通过本机进程的stdio JSON交接数值观察、动作和奖励；没有网络端口，不安装TensorFlow、NNI或Lightning。

原生Torch模型和Senior方法从官方1.1.2源码抽取，保留原文件与许可证在upstream/；改动说明见docs/ADAPTATIONS.md。Junior按原模型/损失移植，不用项目SB3-PPO代替Senior。

## 1. 下载项目

私有仓库： https://github.com/gohmadein/curriculum-vkg-wcci-hpc

可登录GitHub下载ZIP并上传超算，解压为`curriculum-vkg-wcci-hpc`。推荐放在原SMAAC目录的旁边：

```text
/seu_share/home/qiguilin/230269237/111/
  smaac-vkg-wcci-hpc/        # 已有Python3.13和已解压电网数据
  curriculum-vkg-wcci-hpc/   # 新代码；解压ZIP后按需重命名
```

**不需要重新上传1.7GB数据。**本包不带数据、不带旧IEEE14模型的导出权重。

## 2. 安装独立训练环境

在新项目目录执行（登录节点、有网络）：

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
bash hpc/setup_training.sh
```

它只新建当前项目`.venv-training`，不修改原SMAAC环境。Conda下载源与PIP_INDEX_URL不同；若Conda连接失败需按平台镜像设置处理。Torch及CUDA依赖可能需要数GB下载。也可设置TORCH_INDEX_URL指定兼容的官方CUDA wheel源。

不要直接`uv sync`或安装原pyproject.toml：那是原项目Python3.13/CPU锁，不能用来安装旧Ray训练器。

## 3. 指定现有环境和数据

以下路径按当前超算实际目录写好：

```bash
export HPC_SIM_PYTHON=/seu_share/home/qiguilin/230269237/111/smaac-vkg-wcci-hpc/.venv-conda/bin/python
export VKG_DATA_ROOT=/seu_share/home/qiguilin/230269237/111/smaac-vkg-wcci-hpc/l2rpn_wcci_2022-frozen.tar_2026101318291010
```

VKG_DATA_ROOT指向`l2rpn_wcci_2022`的父目录。HPC_TRAIN_PYTHON默认是本项目`.venv-training/bin/python`，HPC_OUTPUT默认是本项目`runs/curriculum_wcci_v1`。默认兄弟目录布局相同时环境脚本会自动找到上述路径，但显式设置更直观。

所有脚本**不含#SBATCH --mem**，因为平台按CPU核数分配内存。Tutor申请18核、16worker；每个训练任务18核、1张GPU，最多同时2任务。Slurm的CUDA_VISIBLE_DEVICES保持原样，不指定物理卡号。

## 4. 一次提交完整链路

```bash
bash hpc/submit.sh
```

会依次提交并打印3个作业号，保存在`runs/submission.json`：

1. Tutor数据准备（CPU）；
2. 依赖Tutor成功的种子0/1训练数组（GPU），各自自动评估A/B/C/D；
3. 依赖两个种子均成功的汇总与结果打包（CPU）。

阶段失败时后继任务不会继续训练，可能显示DependencyNeverSatisfied；先读取失败日志，不重复整批提交。脚本发现现有submission.json会阻止意外重复提交。

`--time=30天`是沿用当前可提交模板的上限，不是预计耗时。真实数据生成量和RLlib吞吐未在超算测过，不能预先承诺结束时间。第一轮日志出来后再估算。

## 5. 查看进度

```bash
cat runs/submission.json
squeue -u "$USER"
```

对应日志：

```text
slurm_output/prepare_<jobid>.out/.err
slurm_output/<arrayjob>_0.out/.err
slurm_output/<arrayjob>_1.out/.err
slurm_output/summary_<jobid>.out/.err
```

运行状态：

```text
runs/curriculum_wcci_v1/preparation/status.json
runs/curriculum_wcci_v1/status_s0.json
runs/curriculum_wcci_v1/status_s1.json
runs/curriculum_wcci_v1/models/senior_s0/run_meta.json
runs/curriculum_wcci_v1/models/senior_s0/training.log
runs/curriculum_wcci_v1/models/senior_s0/simulation.log
```

run_meta记录Junior epoch和Senior iteration；simulator日志用于环境接口报错。任务退出码传给Slurm，不能把.extern的COMPLETED当成实验成功。

## 6. 失败后的恢复

Tutor按chronic保存数据及SHA，可重用已经完成且指纹相同的shard。Junior每epoch保存模型/优化器/RNG；Senior每5iteration保存标准RLlib checkpoint。恢复同一种子任务会复用它们；完成训练的Senior不重训，已完成评估不重跑。

**RLlib checkpoint不包含完整Grid2Op进程状态，因此不是逐位相同轨迹的精确续训**；恢复会重新建立train-only模拟器。已保存计数/最终预算保持，不使用test挑恢复点。硬件故障可能丢失最后一个checkpoint后的工作。

只重提失败的准备/种子/汇总任务，可在原目录使用：

```bash
sbatch hpc/prepare.sbatch
sbatch --array=0 hpc/formal.sbatch
sbatch hpc/aggregate.sbatch
```

这些不是让你现在全部重复提交的命令。先确认旧任务是否仍运行；准备成功后才手工提交训练，两个种子都成功后才汇总。运行中不要改代码或依赖，否则指纹检查会拒绝混合结果。

## 7. 结果自动保存与交回

成功后生成：

```text
runs/curriculum_wcci_v1/summary/table.md
runs/curriculum_wcci_v1/summary/cells.csv
runs/curriculum_wcci_v1/summary/per_seed.csv
runs/curriculum_wcci_v1/summary/per_episode_verified.csv
runs/curriculum_wcci_v1/summary/contrasts.csv
curriculum_wcci_results_<summaryjobid>.tar.gz
curriculum_wcci_results_<summaryjobid>.tar.gz.manifest.json
```

**将最后两个文件发回原对话。**包内含800局、逐种子结果、审计、配置、运行日志和模型快照；原始大规模Tutor数据留在超算，不必传回。汇总脚本要求恰好2模型/8评估/800局，不齐不会生成完整大表。

若只想先导出失败诊断，也可以在计算节点执行：

```bash
source hpc/environment.sh
"$HPC_SIM_PYTHON" hpc/export_results.py --out "$HPC_OUTPUT" --archive curriculum_partial_results.tar.gz
```

包内数值按实际结果记录；没有性能好才继续的筛选门槛，非法/歧义动作保留。表格均值±SD按两个训练种子，不把800局当800个训练重复；配对统计先在同一chronic内平均两个种子。

## 当前验证范围

已完成源码/接口核对、冻结输入校验以及Python/Shell静态语法检查。**尚未在目标集群安装旧Ray环境、运行Tutor全量生成或实际训练；不宣称已通过运行验证。**如果安装或计算节点报错，把日志发回原对话处理。
