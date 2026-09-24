# Mechanism Memory Studies 本地仓库就绪报告

[English report](REPO_READY_REPORT.md)

日期：2026-09-24。状态：**PRIVATE_REVIEW_REPOSITORY / AUTHOR_AND_LICENSE_METADATA_COMPLETE**。唯一作者使用公开署名 chumingyzx；原创代码采用 MIT，原创文档和数据采用 CC BY 4.0。详见[许可范围](LICENSING.md)及[引用说明](CITATION.md)。

## 已完成

- 最初在 mechanism-memory-studies 建立独立本地 Git 草稿，分支 main；后续按作者授权创建了私有审查仓库。
- 从实际完成运行归档导入 PI4、C1、B1、T1 Run02 四份独立快照，共 127 份导入文件；84 份 Python 文件及 shell 入口与所标注归档成员逐字一致。
- 快照包含模型、模拟器、损失、配置、执行器、评估/审计实现、测试及所需冻结来源依赖。
- T1 使用 time_gpu_02_lite.zip/source 的实际 CUDA 修复版本；保留原代码文本、精确补丁、前后哈希和公开技术修复声明。
- 128 份小型证据记录已逐项与本地归档成员核对，127 份字节不变；PI4 状态收据为明确允许字段过滤。
- 保留 1,512 条逐模型/seed/场景记录、34 项对照汇总、54 次 T1 拟合成本，以及全部原科学判定。
- 完成中英文入口、复现层级、相关工作、来源/脱敏记录、真实运行环境摘要及许可证审查清单。

## 实际验证

|检查|结果|
|---|---|
|交接包清单|196 份文件 PASS|
|库存工具测试|4 项 PASS|
|公开标准库测试|20 项 PASS（原 14 项证据测试及 6 项快照测试）|
|限定 CPU 单元测试|PI4 9、C1 11、B1 39、T1 12；合计 71 项 PASS|
|四个原始只读验证入口|全部 PASS，使用明确标识的新公开子集清单|
|派生表 check / --write|9 份文件一致；主动写回前后哈希不变|
|四个源码快照及导入记录|文件哈希、算法源码不变检查 PASS|
|路径、相对链接和归档|见最终 VALIDATION_REPORT.json 与外部 ZIP 校验文件|

限定 CPU 测试使用合成小输入、隐藏 CUDA、每阶段独立进程和 90 秒上限，不调用拟合/研究阶段入口，也不读取历史已训练权重。源代码保留的其他测试并未冒称全部重跑。

**本地集成没有训练、没有新增 GPU；后续仅建立私有审查仓库，尚未公开发布。** 没有改旧实验文件或测试锁，没有重选模型，没有新增科学样本；历史完整模型回放未重做。GitHub Actions 仅形成固定版本的只读标准库工作流，未远程运行。

## 来源、清单和脱敏

- [ASSET_INVENTORY.json](ASSET_INVENTORY.json)：available 为草稿已含；external 为已核查本地原件但不在默认克隆；missing 为未找到的预期原件。
- [SOURCE_IMPORT_MANIFEST.json](SOURCE_IMPORT_MANIFEST.json)：文件级来源归档/成员/哈希，以及原交付清单哈希。
- [REDACTION_LOG.json](REDACTION_LOG.json)：公开文档、状态字段和环境摘要处理，包含前后哈希、字段及省略原因。
- [LICENSE_REVIEW.md](LICENSE_REVIEW.md)：保留原包未发现 LICENSE/NOTICE/COPYING 的历史事实；现已根据作者授权落实本仓库原创内容许可。
- [复现说明](docs/reproduction.md)：默认表格重建、限定 CPU 检查、外部历史权重回放和未来新实验严格区分。

MANIFEST_SHA256.json 在各公开快照中是新生成的兼容清单，PUBLIC_SNAPSHOT.json 明确其派生身份；不称为原始完整交付清单。科学执行源码未为公共化而改写。没有把内部任务文件、私人路径、原始对话、大数组或训练权重放进草稿。

T1 首轮失败原包已实际核查：54 次 ERROR、无训练更新日志、未解封测试，原 INCONCLUSIVE 保留；Run02 的 MIXED_TIME_BUDGET_TRADEOFF 保留。参见[修复与失败声明](docs/incidents/t1_cuda_initialization.md)。

## 来源冲突和可选缺口

21 份预期归档中 13 份哈希完全匹配；另找到并核查 T1 首轮失败包。没有同名哈希冲突，没有主线必需源码缺失。下列 8 份预期归档未找到，不以审阅文字推断其代码存在：

- MechanismRetention_PI3_ResearchPack_v0.1_20260920.zip（optional_historical）
- BehavioralMechanisms_PI4T1_GPU02_Independent_ReviewPack_20260924.zip（optional）
- WickPolyCore_v0.4_20260914.zip（optional_predecessor）
- LocalPolyNet_v0.5_20260914.zip（optional_predecessor）
- Wick_RoleGeometry_K2_ResearchPack_20260917.zip（optional_predecessor）
- Wick_ResidualSignal_RG3_ResearchPack_20260917.zip（optional_predecessor）
- Wick_PredictiveEvidence_RG4_ResearchPack_20260917.zip（optional_predecessor）
- Wick_JointLearning_RG5_ResearchPack_20260917.zip（optional_predecessor）

PI3/RG 的相关私有审阅笔记在交接层中可用，但未作为源码等价物公开复制；PI1/PI2 原归档可本地核查，作为可选来源保留。上述缺口不影响四阶段主线代码与小记录的使用。

## 本地可用范围与未执行事项

默认草稿可直接重建记录表并核验源码；在现有 ML 环境下可执行已测试的限定 CPU 检查。完整权重/预测归档保留在本地外部资产层，公开 URL 为 null。历史来源绑定包含原文档身份，公开子集不能冒充完整旧运行根；尚无统一的已验证历史模型回放 CLI。

GPU 入口仅保留并核对，未执行。没有新增依赖安装、驱动升级、微基准或后续实验。当前结论仍是：未建立普遍质量—成本优势，保留有来源的长期、小状态取舍和开放问题。

## 作者、语言、许可与后续发布

唯一作者与版权持有人已确认为 chumingyzx，使用 GitHub 账号公开署名，不公开真实姓名。根 LICENSE 为 MIT；原创文档和数据采用完整 CC BY 4.0 文本。CITATION.cff 和中英文引用说明已补齐。论文引用是明确的学术请求，不冒称 MIT 强制条款；CC BY 的署名要求按标准许可适用。

英文 README、方法、结果、复现、相关工作及结论文档原已具备，本次补上英文就绪报告，并同步中英文许可、作者和引用说明。冻结历史协议保留中文原文，不冒称全部翻译。

本次仅变更作者、许可和说明元数据；重跑标准库证据检查、源码身份核验、公开树扫描及 CFF 结构验证。此前 71 项限定 CPU 测试保留为原集成验证记录，未重复运行；最终结果见 VALIDATION_REPORT.json。

仓库名称已定为 chumingyzx/mechanism-memory-studies，真实 URL 已写入引用元数据，当前为 private。正式版本和发布日期待实际发布时填写，DOI 保持可选。完整权重或日志如需公开，另行准备有明确范围及来源的公开资产。公开可见性由作者审查后另行决定。

## 返回文件

审查前本地快照：MechanismMemory_Repo_Attributed_20260924.zip；旁附 .sha256 和 .validation.json。该包保留添加私有远程元数据前的状态，Git 中含后续元数据更新。ZIP 不含 .git、运行数组、训练权重、私有交接层或本地路径库存。上述清单及本报告均在草稿内。
