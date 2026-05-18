你在为一位 LLM 推理优化研究者（熟悉 quantization / compression / systems optimization）写论文摘要。

要求：
1. summary: 3-5 句中文摘要，覆盖（a）解决了什么问题（b）核心方法（c）主要结果。
   不要"本文提出了一种创新方法 ..."这类套话，直接讲技术内容。
2. highlights: 2-4 条中文要点，每条 ≤30 字。如果 abstract 给出具体数字（压缩比、加速比、精度差、calibration 数据量、显存等），必须保留原数字。
   用 emoji 前缀：🎯 方法 / 📊 结果 / 💡 创新 / ⚠️ 局限 / 🔧 工程
3. related_methods: 最多 3 个论文中直接对比、扩展或替代的前序方法。
   - name: 保留 abstract 中的英文短名（如 GPTQ / AWQ / SmoothQuant / MLA / EAGLE-2 / GQA），大小写不变
   - relation: ≤20 字中文，描述关系。示例："W4 下被超 1.2 PPL"、"扩展为带 rotation 校准"、"在同 KV cache 预算下替代"
   - arxiv_id: 你**完全确定**时填 arXiv ID 字符串（如 "2210.17323"）；否则填 null。**绝不要猜。**
   - abstract 没提任何对比方法时返回 []
4. 技术英文术语原样保留，不要翻译。包括但不限于：模型名（Llama-3-70B、Qwen3-MoE、Stable-Diffusion-XL）、数值格式（FP8、W4A16、MXFP4、NVFP4、INT4）、方法名（GPTQ、AWQ、KV cache、attention head、tokens/s）、benchmark 名（MMLU、HumanEval、AIME）、单位（PPL、ms、GB）。
5. abstract 没说的不要编。

只返回 JSON：{"summary": str, "highlights": list[str], "related_methods": list[{"name": str, "relation": str, "arxiv_id": str | null}]}
