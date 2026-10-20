# 原生方法与部署适配

上游curriculumagent1.1.2，wheel SHA256:
e679b1cc7c5114de962189d2dceb0603a116108ec89c5bbcadea2b70184de6ce。
原始逐Python文件摘要在upstream/original_manifest.json，许可证upstream/LICENSE。

- Teacher阶段复用训练集387动作库，不宣称从头重跑官方Teacher搜索。
- Tutor直接使用官方GeneralTutor和common.utilities。离线simulate上限取消并独立记录；物理环境、train划分不改。全1507train一次回放。
- Junior保留原生默认4x1000/ReLU、dropout0.25、NLLLoss、Adam；移植到纯PyTorch循环，固定100epochs，不进行NNI搜索或验证集checkpoint选择。
- Senior TorchModel从原类逐字抽取，仅去除TF/Lightning导入，不改变forward/value/softmax/权重映射。
- Senior run_next_action/step/check_terminated_truncated用AST抽取；初始化和reset改为通过冻结make_env创建且仅采样train。保留上游终止/截断标记，不声称修复过其算法。
- 保持topo=False与部署策略一致，无额外归一化。原PPO_Reward增加logger兼容参数，作为additional reward，物理环境主奖励不替换。代理step仅将原生训练奖励交给Senior。
- 终止状态若含NaN，用最后一个有限观测占位；非终止NaN直接失败。防止无法序列化和非有限模型输入；原生终止惯例仍保留，须在论文方法适配中披露。
- Python3.10/NumPy1/RLlib2.5和Python3.13/NumPy2/Grid2Op1.12.5进程隔离，JSON传输数值；不是SB3-PPO替换。老环境Torch2.8，仿真/评估原环境Torch2.14，模型最终导出NumPy矩阵。
- Torch/NumPy导出误差在train状态检查，阈值1e-4，失败不会产出正式完成标记。
- 部署基线保留原submission端口MyAgent的动作排序、重连、复合操作、idx0真值行为及40调用缓存预算。
- 四格同权重，VKG仅检索同一387库，TR不去掉基线已有原生机制。B/D各仅一条固定抽样种子/训练种子，不与旧case14的五抽样设定混淆。
- A/B/C/D均由同一公共eval_baselines.py评估；旧case14权重不用于118。

预算与模型共享方式属于本次预定适配协议，不以测试分数选择。详细冻结值在hpc/protocol.json。

- Senior训练环境也使用部署端现有StepSimulator：每个真实物理步缓存重复试算并限制40次，预算耗尽时返回无效候选，避免原生重连辅助重复调用使环境抛限额异常。Tutor离线扫描不使用该代理。
