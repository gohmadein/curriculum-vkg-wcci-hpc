# 118节点 GNN soft-label IL 四格（2026-09-30）

目标：补大表118节点IL的一整行，不重复训练已有MLP-PPO。
本项目GNN编码器+soft-label监督训练适配版，不声称原论文逐项复现。

## 与case14的共有协议和明确差异

- `experiments/wcci2022_ppo.yaml`：沿用已完成118节点MLP-PPO的物理环境与动作库。
- 同一训练集教师库387个母线动作+空动作=388（四格共享，库不使用VKG）。
- `experiments/wcci2022_split.json`：train1507/val30/test100，固定目录ID，不重新划分。
- 标签仅用全部1507条train；风险阈值0.9，重连优先/平静空动作，扫描合法候选、最低rho教师。
- 每chronic最多300个决策状态，回放最长2017步；没有决策的chronic保存0行文件，覆盖审计不遗漏。
- A/B普通动作标签，C/D重写标签，四格共用同一状态集；线上0次simulate。
- 图编码器原来拒绝储能端子：本适配版新增STORAGE静态类型、母线/线路边（共532个端子节点）。
  不增加storage_power/charge等冻结观测列表之外的信息。case14仍为4种类型，参数形状和行为不变。
- 网络hidden64/layers3/features256，线性动作头，Adam3e-4、30epochs、batch32。
  batch相对case14缩小以控制大图内存，四格一致；无test调参、无早停。
- 温度{.003,.01,.03}仅按A3个种子×全部30val选，平手取大温度；随后固定。
- 正式四格各5种子，test100，2000回合。公共既有test，不称新确认集。

## 自动流程

`run_wcci_il_pipeline.py`：标签生成→全标签指纹/覆盖审计→四格val短测（8回合）→
温度val选择（270回合）→20个模型→2000test回合。所有正式评估使用公共harness。
只依技术门控继续，不因分数差而改协议；错误会阻断后续并记录。

CPU资源：标签12worker，训练最多4worker单线程，评估最多12worker总量；
与LJN的16worker并行，阶段峰值28核。4090任务不变，不使用任何GPU。
输出分步status.json/report.md、逐episodeCSV、模型与数据指纹、源代码提交、温度曲线。
离线模拟次数从各状态合法+可重写数求和，与零在线模拟分开报告。

已有2条训练chronic短测：14个状态，约21秒，388候选正确；
四格8样本forward/backward均有限，约0.10--0.12秒；该小样本不作为正式耗时保证。
正式标签数量影响训练时长，生成后按epoch实测给ETA，不沿用case14两小时的估计。

## Frozen inputs

- `experiments/wcci2022_ppo.yaml` SHA256 `c106473cd0dac63a78ffb7ad743d9c0ee29b70e763bd223fef01262b6c156c03`
- `experiments/wcci2022_split.json` SHA256 `2b72d4c139c7ea1ec7c7cf151dc7d0b30c3b09dbbf3da2e508df7900cb40515c`
- `experiments/wcci2022_library.npy` SHA256 `bac023b053f811cc3c93e3c18aecdbddad26a683b0fe56990aeb63b1bdaf13f1`
