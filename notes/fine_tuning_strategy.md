# 基于收益率对策略 LLM 做 Fine-Tuning 的路径选择

## 背景

当前架构：portfolio_manager 的 LLM 把多 agent 辩论结果转化成结构化 JSON，包含 5 个字段（entry / add_position / take_profit / reduce_stop / stop_loss）的价格与 size_pct。目标是用 backtest 收益率反向优化这个生成过程。

直接结论：**"手动输入失误原因"是可选的**——取决于选哪条路径。

---

## 两个正交维度

| | **学习落在哪里** | **学习信号是什么** |
|---|---|---|
| 选项 A | 模型权重（fine-tune） | 仅结果（PnL） |
| 选项 B | 模型权重 | 结果 + 失误归因（reason） |
| 选项 C | Prompt 上下文（memory / RAG） | 仅结果 |
| 选项 D | Prompt 上下文 | 结果 + 失误归因 |

"标记失败 + 写原因" 对应 **B 或 D**；只用 PnL 不写原因对应 **A 或 C**。两条路都成立，实现成本和适用模型差异显著。

---

## 四种路径，按实现成本排序

### Level 1 — Memory autopsies（D，prompt + reason）

代码里已经有了：[portfolio_manager.py:231-237](../tradingagents/agents/managers/portfolio_manager.py#L231-L237) 已在调用 `memory.get_memories(curr_situation, n_matches=2)` 把过去的"教训"塞进 prompt 的 `lessons_section`。但 [memory.py](../tradingagents/agents/utils/memory.py) 的 `add_situations()` 现在没人调用，索引是空的。

**最小闭环**：

1. 每次 backtest 跑完后，遍历 trades，识别"明显失误"（如 stop_loss 触发后 5 个交易日内价格回到入场之上 → whipsaw）。
2. 对每笔失误写一段 autopsy（人写或便宜 LLM 当 judge 写）：

   ```
   2024-01-30 设的 471.8 止损只有现价 1.2% 的距离，
   1/31 一根普通回踩就触发，全仓出场后 5 日内涨 2.4%。
   教训：趋势市中 SMA20 之上的止损应至少 1.5×ATR。
   ```

3. 持久化 `(market_report_text, autopsy_text)` 到 disk，下次启动时 `add_situations()` 喂进 BM25 索引。
4. 下次出现"相似行情"时，prompt 里就会自动带上这条 autopsy。

**优点**：零训练成本、跨任意 LLM provider 通用、当天可用。
**缺点**：BM25 检索精度有限（同义改写会失配，可换 embedding-based）；上下文长度有上限；记忆容量随积累饱和。

---

### Level 2 — DPO over backtest outcomes（A，权重 + 仅结果）

不需要写原因，**用 PnL 自动生成偏好对**：

1. 对每个历史日期 X，用当前 LLM 用不同 temperature 采样 N 个候选 JSON。
2. 每个候选都丢回 [back_test/engine.py](../back_test/engine.py) 跑短窗口 replay，得到 `PnL_i`。
3. 构造偏好对：`chosen = argmax_i PnL_i`，`rejected = argmin_i PnL_i`。
4. 用 DPO 损失微调权重。

**优点**：信号是客观结果，不需要人写原因；天然 outcome-aligned。
**缺点**：

- 必须用**支持微调的模型**：OpenAI gpt-4o-mini、Google Gemini Flash 都有；Anthropic Claude 几乎没有；OSS（Qwen / Llama / DeepSeek）通过 LoRA 最划算。可能要把 portfolio_manager 这一步的 LLM 单独切到可微调模型上，其它 agent 仍用通用 API。
- N 个候选 × 每天 = 大量 backtest replay。
- 单日 PnL 噪声大，建议 reward 用 1–4 周持有期 PnL 或 Sharpe。

---

### Level 3 — SFT with corrections（B，权重 + 手动 reason）

最初设想的版本：人工把失败的 JSON 改成"应该长什么样"，作为 supervised target。

**优点**：精确控制、可注入领域知识。
**缺点**：

- 标注成本极高（每笔交易都要写"正确版"）。
- **Hindsight bias 严重**：知道结果再去构造"完美策略"，模型会学到事后诸葛亮的规则，部署时没用。
- 样本稀疏：好交易是常态，失败是少数。

**不太推荐**，除非只标注**规则违反**类失误（"stop 距离 < ATR×0.8" 这种可机械判定的）。

---

### Level 4 — RL（PPO）

把 [back_test/engine.py](../back_test/engine.py) 当 environment，reward = portfolio Sharpe，policy = LLM。

**优点**：理论上最强、能学复杂策略。
**缺点**：实现复杂、训练不稳定、只能用 OSS 模型、reward 信号稀疏（一年 ~50 个 episode）。**除非有专门 RL infra，不建议从此入手**。

---

## "是否需要手动输入失误原因"

| 路径 | 需要原因？ | 原因可来自 |
|---|---|---|
| Memory autopsies | **强烈建议** | 你写 / LLM-judge 写 / 规则模板 |
| DPO | **不需要** | 仅 PnL |
| SFT | **需要 + 改写正确目标** | 必须人写 |
| RL | **不需要** | 仅 reward |

**LLM-as-judge 做法**：用 Claude / GPT-4 当裁判，输入 `(market_report, strategy JSON, 之后 N 天的实际价格走势)`，输出失误归因。能把 99% 的标注自动化，你只需要抽样审计 10–20% 是否合理。这是 Level 1（Memory）的标配做法。

---

## 比 PnL 更稳的失败定义

直接用 PnL 做"成功 / 失败"标签会撞上 hindsight bias 的墙——决策合理但运气不好的交易，会被错误地教成"以后别这样"。建议先用**规则违反**做硬标签，再叠加 PnL 做软信号：

- **硬标签（一定是失误）**
  - stop 触发后 5 日内价格新高 → whipsaw
  - entry size 超出 prompt 规定的 band
  - TP 设在成本之下
  - reduce_stop 距离入场 < 0.5×ATR

- **软标签（可能是失误）**
  - trade 跑完 PnL < −0.5×ATR、且持有期 < 5 天

- **不要标**
  - PnL < 0 但流程合规、属于正常波动范围内的回撤

这条建议对四种路径都适用——**核心是不要把"运气不好"训练成"决策不好"**。

---

## 对当前 codebase 的具体建议

按实现优先级：

1. **先把 Level 1 跑起来**
   - 写 `back_test/auto_autopsy.py`：跑完 backtest 后扫描 trades，按硬标签规则识别失误，调用便宜 LLM 写 autopsy，存到 disk。
   - CLI 启动时把这些喂给 `FinancialSituationMemory`。
   - 1–2 天能搞定，立即提升下次回测表现。

2. **Level 2（DPO）作为下一步**
   - 等积累了 6–12 个月的回测数据、确认 Level 1 的 ceiling 后再考虑。
   - 需要决定是否引入 OSS 模型（比如 Qwen2.5-7B-Instruct + LoRA）专门做 portfolio_manager 这一步。

3. **Level 3 暂时不做**
   - 除非发现某些规则违反类失误是 LLM 系统性犯的，那时手动 SFT 一小批"反例"是值得的。

4. **Level 4 不做**
   - 投入产出比对个人 / 小团队不划算。

---

## Level 1 实施清单（如要上手）

| 模块 | 内容 |
|---|---|
| 失误识别器 | 输入 `(trades, price_df)`，按硬标签规则输出失误列表 |
| LLM-judge 调用器 | 输入 `(market_report, strategy JSON, post-trade price)`，输出 autopsy 文本 |
| Memory 持久化层 | pickle 或 JSONL 到 `back_test/memory/{ticker}/` |
| CLI 加载钩子 | 启动时把已有 memory 喂给 `FinancialSituationMemory` |

每一步都不大，串起来 1–2 天工作量。
