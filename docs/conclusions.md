# Conclusions, limitations, and open questions

## Current conclusion (English)

Within the evaluated simulator, recipes, and budgets, we have not established a general quality–cost advantage for the proposed fixed-context structured model. Persistent mechanism context combined with behavioral supervision improves post-wait response-difference prediction relative to the implemented free-four-state comparator, and this direction recurs across independently generated datasets. However, persistently conditioned recurrent baselines recover part of that capability. Under the equal complete-fit time protocol, the wider GRU performs better on the recorded mechanism and short-horizon tasks, while the structured candidate retains an advantage in several long-horizon rollouts with fewer persistent state coordinates.

These results indicate a task- and resource-dependent trade-off rather than either a universally superior architecture or a general impossibility result. The original experiments retain their recorded outcomes. Open questions include whether long-horizon benefits survive stronger structure-matched comparators, whether smaller persistent state produces measurable deployment value, and whether these observations transfer beyond the current simulator. Those are untested possibilities, not implied benefits or commitments to extend the present protocols.

## 当前结论（中文，可直接用于仓库）

在已评估的模拟系统、训练配方与资源预算下，尚未建立所提出固定上下文结构相对合理对照的普遍质量—成本优势。持久机制上下文与行为监督结合，在静置后的响应差分预测上相对现有自由四维对照呈现了可重复的收益；但持续读取校准信息的普通循环模型能够复现部分能力。在相同完整训练时间下，宽GRU在已记录的机制和短期任务上表现更好，而结构化候选以较小的持久状态，在部分长期滚动预测中仍有优势。

这些结果更符合依赖任务与资源条件的设计取舍，而非普遍更优的新架构，也不构成对相关结构的一般否定。原实验的正、负和混合结论保持不变。较小状态是否能产生实际部署收益、长期预测信号能否在更强结构匹配对照及外部系统上保持，仍是需要独立检验的开放问题，而不是现有结果已经支持的结论。

## Evidence boundary

| Supported within the recorded setting | Not established |
|---|---|
| Some structure–supervision combinations help post-wait differences relative to the tested free-state comparator | A universal need to hard-split static and dynamic latent coordinates |
| C1 mechanism-direction improvement repeated within one simulator | Full replication of all protection criteria: only 1/3 datasets passed |
| Wider recurrent models benefit from continuous calibration access and longer training within the same time cap | A cross-stage causal estimate that every B1→T1 difference is caused only by update count |
| T1 wide GRU wins all recorded paired mechanism and short-horizon comparisons | Superiority of every GRU implementation, or a general impossibility result for structured dynamics |
| The small structured model retains some long-horizon quality advantage | Independently confirmed deployment speed, memory savings, universal stability, or accurate parameter identification |

## Keep four kinds of status separate

1. **Historical scientific decision:** preserve `PARTIAL_REPLICATION`, `MIXED_TRADEOFF_NO_DOMINANCE`, and `MIXED_TIME_BUDGET_TRADEOFF` exactly.
2. **Engineering evidence:** a timeout, missing replay, or numerical failure is not automatically a negative learning result.
3. **Research interpretation:** describe what positive and negative comparisons support, with their resource and task scopes.
4. **Future question:** mark it untested; do not turn it into an automatic GPU job or reinterpret a closed gate.

The original PI-4 elementwise replay failure stays visible alongside later task-level replay support. T1's CUDA-startup repair is recorded as a versioned execution change, not silently merged into the original source identity. The original T1 startup-failure archive was located and inspected during assembly; its INCONCLUSIVE status remains in the [incident record](incidents/t1_cuda_initialization.md).

## Why the repository has value

The reusable contribution is an inspectable experimental system: controlled compositional tasks, information-matched baselines, explicit resource policies, behavioral diagnostics, and source-linked outcomes that survive numerical checking. The collection can inform what to compare and what not to infer. This statement does not imply publication acceptance, proven novelty, or a performance benchmark against all popular architectures.

## Questions worth keeping open, without promising answers

- Does the long-horizon advantage remain against a conditioned recurrent comparator with comparable structural constraints and tuned integration?
- Does a strict persistent-state budget matter in an application once encoder storage, solver workspaces, precision, and actual hardware are counted?
- Can calibration-invariant mechanism inference improve with different data or representations, without simply amplifying sensitivity to nuisance initial conditions?
- Which findings survive measurement noise, different component laws, unknown interconnections, and independently collected trajectories?

No retrospective horizon selection, new statistical-significance claim, or extension of a frozen budget is authorized by this list. New evidence belongs to a new protocol with new provenance.

Sources: the byte-traceable [records](../results/evidence_manifest.json), [generated tables](../results/derived/TABLES.md), and [study map](study_map.md). Supplied historical review notes are editorial sources, not an additional independent training replicate.
