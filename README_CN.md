# Mechanism Memory Studies｜机制记忆研究记录

**校准信息、持久上下文、可组合动力学与训练预算取舍的受控实验。**

[English](README.md) · [阶段索引](docs/study_map.md) · [方法与指标](docs/methods.md) · [结果表](results/derived/TABLES.md) · [结论与开放问题](docs/conclusions.md) · [复现说明](docs/reproduction.md)

这个仓库研究：模型能否从部件接口观测中学习响应规律，在运动状态改变或衰减后保留有用信息，并在静置、再激励和重新连接后继续使用它们。它不是“已经胜过主流模型的新架构”发布，也不是证明所有相关路线均无价值的失败合集。

## 当前认识

候选每个部件保留两维固定上下文与两维动态状态，结合结构化动力学和行为监督。四个数是学习出的潜在坐标，不等于已经辨识出的四个物理量。

PI-4中观察到机制差分收益；C1在三套同模拟器独立抽样数据中重复了正方向，但只有一套通过完整保护门槛。B1加入能够持续读取校准的普通模型后，结果仍存在任务取舍。T1按相同600秒完整拟合上限比较时，宽条件GRU在全部机制与短期配对中误差更低；候选在7/9个对该GRU的长期配对、9/9个对图GRU的长期配对中误差更低，同时持久状态较小。这些是有限配对计数，不是独立任务数量或统计显著性。

**目前未建立普遍的质量—成本优势；仍保留了关于状态组织、监督与长期预测之间取舍的可检验线索。** 后续能否利用这些线索，需要新的能力假设和独立评价，不能事后将原门槛未通过改写成成功。[完整结论](docs/conclusions.md)

## 不训练即可查看和重建结果

```bash
python scripts/rebuild_results.py
python -m unittest discover -s tests -v
```

以上仅核对小文件哈希、重建表格与对照原决策，不读取权重、不运行GPU、不重新训练。主动刷新派生表格使用：

```bash
python scripts/rebuild_results.py --write
```

完整模型回放和原协议训练使用对应的冻结代码版本及结果资产，不能让最新版执行器冒充所有历史版本。查看[复现层级与入口](docs/reproduction.md)。

## 能拿走什么

读者可以使用受控模拟任务、结构化与条件基线、相同更新数和相同时间上限的比较协议、带配对与校准干预的指标，以及可追溯的正负结果。数学恒等式、单元测试通过、预测回放和独立重新训练属于不同证据层级。

主线为PI-1至PI-4/T1。早期Wick、LocalPolyNet和RG研究保留为独立历史脉络，不将其定理当成当前动力学结果的证明；也不把所有研究单元命名成新架构。[历史背景](docs/historical_context.md)

## 范围与开放性

当前证据限于同一模拟器、已知连接、无测量噪声与有限参数范围。没有充分公开方法复现、外部真实系统确认，亦没有测得部署延迟或显存优势。原协议按实际状态收束，但新假设和独立复现仍然开放。

发布应保留方法、指标、冻结协议和异常说明，去除个人路径、凭据及内部Agent指令。参与工作时使用过AI辅助，历史“独立复核”指特定文件/数值检查，不应暗示独立人类同行评审。唯一作者以 [chumingyzx](https://github.com/chumingyzx) 作为公开署名，不公开真实姓名。原创代码采用 [MIT](LICENSE)，原创文档和数据采用 [CC BY 4.0](LICENSES/CC-BY-4.0.txt)，详见[许可范围](LICENSING.md)。

研究中使用本项目时，请按 [CITATION.cff](CITATION.cff) 和[引用说明](CITATION.md)引用项目，并注明研究阶段与版本。论文引用是明确的学术请求，不另加许可条件；MIT 的版权声明保留义务与 CC BY 的署名义务按各自许可执行。

仓库以[英文 README](README.md)及英文方法、结果、复现文档为主，另有中英文结论和[英文就绪报告](REPO_READY_REPORT.md)。历史中文协议保留原文，未冒称全部翻译。

## 仓库状态

仓库 [chumingyzx/mechanism-memory-studies](https://github.com/chumingyzx/mechanism-memory-studies) 当前设为 **private，供作者审查**；公开发布待作者审查后决定。引用元数据已填写真实仓库地址，未虚构 DOI 或正式发布版本。

## 本地集成状态

PI4、C1、B1、T1 Run02 四份独立源码快照已集成，含模型、模拟器、目标、配置、入口和测试。python scripts/verify_reference.py 核验来源；python scripts/check_reference_cpu.py 仅运行限定 CPU 单元测试。20 项标准库测试与 71 项限定 CPU 测试通过；本地集成没有训练或新增 GPU 实验；现已建立私有审查仓库。详见[就绪报告](REPO_READY_REPORT_CN.md)、[资产清单](ASSET_INVENTORY.json)、[来源清单](SOURCE_IMPORT_MANIFEST.json)、[脱敏记录](REDACTION_LOG.json)和[许可记录](LICENSE_REVIEW.md)。
