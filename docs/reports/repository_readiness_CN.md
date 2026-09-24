# 仓库内容与验证报告

[English report](repository_readiness.md) · [文档索引](../README.md)

仓库：[dethronedname/mechanism-memory-studies](https://github.com/dethronedname/mechanism-memory-studies)。唯一作者以 **dethronedname** 作为公开署名。原创代码采用 MIT，原创文档和数据采用 CC BY 4.0。详见[许可范围](../project/licensing.md)、[引用说明](../project/citation.md)和[验证记录](validation.json)。

## 内容与目录

- PI4、C1、B1、T1 Run02 四份独立快照共导入 127 份文件，其中 84 份 Python 文件；Python 和 shell 源码与标注的执行归档成员逐字一致。
- 包含模型、模拟器、目标、配置、执行入口、评估与测试；T1 保留 CUDA 初始化修复声明、精确补丁及原始来源哈希。
- 128 份小型证据文件保留 1,512 条逐模型/seed/场景记录、34 项对照汇总与 54 次 T1 拟合成本。127 份证据逐字保留，PI4 状态收据采用明确的允许字段过滤。
- 英文为主要文档语言，另有中文 README 和双语结论。历史中文协议保留原文。

根目录保留中英文 README、LICENSE 和 CITATION.cff。研究文档位于 docs/，项目说明位于 docs/project/，验证报告位于 docs/reports/，机器可读清单位于 catalog/；贡献指南位于 .github/。重复资产清单已合并。

## 验证与科学边界

初次集成通过 20 项标准库测试、71 项限定 CPU 测试（PI4 9、C1 11、B1 39、T1 12）、四个原始只读验证入口和九份派生文件一致性检查。

公开整理核对迁移后的清单路径、Markdown 链接、作者和引用元数据、文件哈希及科学内容不变性；标准库测试和源码核验在本地及 [CI](../project/ci_scope.md) 执行。71 项 ML 相关 CPU 测试保留为此前集成记录，不冒称本次全部重跑。

本次目录整理与发布没有训练、没有新增 GPU 实验、没有神经模型回放。算法、数值证据、科学判定、失败记录和测试锁保持不变。

## 来源和可用性

- [资产清单](../../catalog/assets.json)：available 为仓库已含；external 为在克隆之外核查的原件；missing 为未找到的预期原件。
- [源码来源](../../catalog/source_imports.json)：归档成员、原始交付清单哈希及明确标识的新公开兼容清单。
- [脱敏记录](../../catalog/redactions.json)：说明文档改写、状态字段过滤和环境摘要。
- [整理变更](../../catalog/curation_changes.json)与[文件清单](../../catalog/file_manifest.json)：目录变化、交接差异及当前文件哈希。
- [许可记录](../project/license_review.md)：原包扫描事实及作者授权。

21 份预期归档中 13 份完全匹配，另核查了 T1 首轮失败包；没有同名哈希冲突或主线必需源码缺口。八份可选归档仍未找到，确切文件名保留在资产清单中；相关审阅笔记不作为源码替代物。

T1 首轮仍为 54 次启动 ERROR、零训练更新和 INCONCLUSIVE；Run02 仍为 MIXED_TIME_BUDGET_TRADEOFF。见[失败及修复记录](../incidents/t1_cuda_initialization.md)。

## 公开范围

本仓库公开代码与小型证据。完整权重、预测数组和原始运行归档仍为外部资产，尚无公开下载链接；也没有新增统一历史神经回放 CLI。仓库公开不等于已创建正式软件 Release、论文 DOI 或获得同行评审。

当前结论仍是：未建立普遍质量—成本优势，保留有来源的长期、小状态取舍与开放问题。
