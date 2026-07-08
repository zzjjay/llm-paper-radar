# MLSys 会议论文趋势分析 — Design Spec

**Date**: 2026-07-08
**Author**: zhaolin (with Claude)
**Status**: Draft (awaiting review)

## 1. Goal & Non-Goals

### Goal

给定一个学术会议(**首个目标: MLSys 2026**),一次性拉取该会议全部**已录用**论文,用 LLM 判断哪些属于 **LLM 推理部署优化**方向(比 `llm-paper-radar` 现有的量化/压缩范围更广,涵盖 KV cache 优化、量化、投机解码、调度与批处理、MoE 推理、长上下文/PD 分离、多卡异构部署、编译器与算子融合等),按子方向分组,并对每个子方向生成研究趋势总结,最终产出一份 Markdown 报告。

这是 `llm-paper-radar` 仓库内的**新增能力**,与现有的每日滚动增量 pipeline(`scripts/daily.sh`)完全独立,不修改其行为。

### Non-Goals (v1)

- 不做跨会议通用化——只硬编码支持 MLSys 2026 一个会议,字段名/decision 解析逻辑可以写死。通用化留到后续迭代。
- 不做全文精读——只用 title + abstract 做判分和趋势总结。
- 不做历年同比趋势(没有历史基线数据),只做"当届横切面"总结。
- 不做子方向分类体系的强枚举——预设一份初始锚点列表供 LLM 参考,允许模型在"其它"里给出新方向名,人工后续迭代时再决定是否固化。

### Success Criteria

- 完整拉取 MLSys 2026 全部 accept 论文,论文数不低于粗略下限(如 <20 篇视为抓取不完整,报错而非静默产出)。
- 判分阶段每篇论文都有可诊断的 hard_gate / subfield / reason,LLM 调用失败不静默丢论文(沿用 `pipeline/filter.py` 的 `judge unavailable` 记录方式)。
- 产出的报告包含:各子方向论文数量分布、每个子方向的趋势总结(核心问题、代表性论文)、全量论文列表。

---

## 2. 架构与数据流

新增文件(均为新增,不改动现有 daily/weekly pipeline):

```
sources/openreview_venue.py     # 按会议批量拉取全部 submission + decision，过滤 accept
prompts/inference_relevance.md  # LLM 判分 prompt：是否属于 LLM 推理部署优化 + 子方向
pipeline/venue_filter.py        # 判分阶段（复用 pipeline/filter.py 的 LLM 调用框架/风格）
pipeline/venue_group.py         # 按 subfield 分组统计（纯代码）
scripts/venue_report.sh         # CLI 入口
workflows/venue_trend_report.js # Workflow 脚本：并行趋势分析 + 综合报告
```

### 数据流

```
1. 拉取 (sources/openreview_venue.py)
   fetch_venue(venue="MLSys.org/2026/Conference")
   → 翻页拉取全部 /-/Submission
   → 关联 decision（具体字段名待 OpenReview API 可用时探测确认；
      按当前观察，403 挑战是间歇性的，不代表 API 结构变化）
   → 只保留 accept，复用现有 Paper 模型
   → 落盘 data/raw/mlsys-2026/accepted.json
   → 翻页过程中增量落盘每页原始 note（data/raw/mlsys-2026/openreview_pages/），
     支持中途失败后从断点续拉

2. 判分 (pipeline/venue_filter.py)
   新 prompt: prompts/inference_relevance.md
   - hard_gate: 是否属于"LLM 推理部署优化"
   - subfield: 自由文本，prompt 内给出预设锚点列表：
     KV cache 优化、量化、投机解码、调度与批处理、MoE 推理、
     长上下文/PD 分离、多卡异构部署、编译器与算子融合、其它（需说明新方向名）
   → data/scored/mlsys-2026.json
   （单篇 LLM 调用失败：沿用 filter.py 的 judge-unavailable 记录方式，
     hard_gate=True + 可诊断 reason，不静默丢论文）

3. 分组 (pipeline/venue_group.py, 纯代码)
   按 subfield 分组统计数量、列出论文
   → data/scored/mlsys-2026-grouped.json

4-5. 趋势分析 + 综合报告 (Workflow: workflows/venue_trend_report.js)
   pipeline(subfield_groups,
     group => agent("总结该方向的核心问题/代表性论文/方法共性",
                     {schema: TREND_SCHEMA}))
   → 各 subfield 相互独立，并行执行
   → synthesis agent 汇总所有方向总结 + 论文数量分布
   → 写 digests/mlsys-2026-report.md
```

---

## 3. 错误处理

- **拉取阶段（step 1）**:
  - 403 挑战与现有 429 同等对待——指数退避重试（如 5s/10s/20s/40s，多给几次机会，观察到挑战会自行过去，非永久封禁）。
  - 重试耗尽后**不写入"当作完成"的空/半成品结果**，脚本非零退出，报错提示用户稍后手动重跑。
  - accept 论文总数低于粗略下限（如 20）→ 视为抓取不完整，报错而非继续往下跑分析。这与仓库 `AGENTS.md` "不要静默跳过昂贵步骤"的原则一致，同样适用于"不要静默接受不完整抓取结果"。
- **判分阶段（step 2）**: 单篇失败沿用 `pipeline/filter.py` 现有套路，不中断整体流程。
- **趋势分析阶段（step 4）**: 某个 subfield 的 agent 失败 → 该 subfield 在报告中标注"分析失败"，仍列出论文列表；不影响其它 subfield。

---

## 4. 测试

- **拉取阶段**: 网络依赖，不写单元测试；手动跑一次验证 decision 字段解析正确。
- **判分/分组阶段**: 纯逻辑 + prompt，构造几个 fixture 论文（明显相关 / 明显不相关 / 边界案例）跑 `venue_filter.py`，断言 hard_gate 与 subfield 输出符合预期。
- **趋势分析/报告阶段**: 不写自动化测试，人工检查报告可读性与分类是否合理。

---

## 5. 已知风险

- OpenReview API 自 2026-06-30 起出现间歇性 403（Cloudflare 挑战），已通过重试处理，但如果挑战持续时间变长（超过重试窗口），拉取阶段会整体失败，需要人工重跑。
- MLSys 在 OpenReview 上的 decision 字段格式尚未实测确认（本次设计期间 API 持续 403，未能探测），实现阶段第一步就是探测该结构并调整 `openreview_venue.py` 的解析逻辑。
