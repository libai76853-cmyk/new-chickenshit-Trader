# Laya 搭配什么预测模型：推荐方案

日期：2026-09-21
前提：美股、日频、中低频选股（个人资金、IBKR 账户、Mac 本机算力）。高频不在讨论范围。

## 一句话结论

搭配的"预测模型"应该是**两层**，不是一个：

1. **主模型：LightGBM（走 Qlib 框架）**。横截面收益排名预测，中低频选股的行业标配，CPU 就能跑，能吃任何特征。Laya 的事件概率就是作为特征塞进去。
2. **副模型：Kronos（金融 K 线基础模型）**。用金融数据从头预训练，MIT 开源，输出未来 K 线的采样路径，从中提炼几个数值特征，也喂给 LightGBM。

Chronos-2、TimesFM-2.5、Moirai-2 这些通用时序基础模型**不要用来预测方向**，只能考虑用来预测波动率、决定仓位大小。

## 分工表

| 层 | 模型 | 干什么 | 为什么是它 |
|---|---|---|---|
| 文本侧 | Laya（用金融文本微调） | 每票每日的事件概率：利好/利空、并购、业绩预警、监管、高管变动 | 校准概率、100+ 语种、一次前向多问、免费自托管 |
| 数值侧：基础模型 | Kronos-small 或 Kronos-base | 读最近 512 根 K 线，采样未来路径，提炼：预期收益、预测波动、上涨概率 | 唯一开源的金融专用 K 线预训练模型；学术证据表明通用模型在金融上不行，金融数据从头预训练的才行 |
| 数值侧：主模型 | LightGBM | 把因子、Kronos 特征、Laya 特征全部融合，预测 T+5 或 T+10 收益排名 | 表格数据最稳的模型，可解释，Qlib 里现成，Alpha158 因子库现成 |
| 波动率 | GARCH（首选）或 Chronos-2 | 预测未来波动，做波动率目标仓位 | 时序基础模型在波动率上的表现明显好于在方向上 |

## 数据流（每日收盘后跑一遍）

```
日线 OHLCV ──┬─→ Alpha158 因子（Qlib 内置）─────────────────┐
             └─→ Kronos 采样 N 条路径 → 3 个特征 ──────────┤
                                                            ↓
新闻/公告/8-K → Laya → 每票每日事件概率 → 按票聚合并衰减 ──→ LightGBM 预测收益排名
                                                            ↓
                              top-k 多头（或多空），波动率目标仓位 → IBKR 下单
```

Qlib 是公共底座：LightGBM 基线、Alpha158 因子、回测框架，以及 Kronos 官方微调流程的数据准备，全都建在 Qlib 上。

## 候选模型逐个说明

### Kronos（推荐作为数值侧副模型）

- 出处：AAAI 2026 论文，GitHub shiyu-coder/Kronos，约 39k stars，MIT 许可。
- 训练：约 120 亿根 K 线，覆盖 45 个交易所。
- 开源权重：Kronos-mini 4.1M 参数（上下文 2048）、Kronos-small 24.7M（512）、Kronos-base 102M（512）。Kronos-large 499M 未开源。
- 输入：open/high/low/close，可选 volume/amount 的 DataFrame。输出：自回归采样的未来 K 线，可调温度和 top-p，多路径平均。
- 微调：官方有完整微调流程，数据准备依赖 Qlib，多卡训练。
- 自称：RankIC 比最强通用时序基础模型高 93%。这是项目方自己的数字。
- 上下文 512 在日线上约 2 年历史。
- README 自己声明："不是生产级量化交易系统"，原始信号需要组合优化和风险因子中性化。
- 独立评价（Jonathan Kinlay，2026 年 2 月）："有前景的研究方向，不是生产级 alpha 引擎"。他的核心提醒：下一根 K 线 MSE 提升 5%，可能只对应 IC 0.01，扣掉买卖价差就归零。

### LightGBM / Qlib（推荐作为主模型）

- Qlib 是微软的开源量化研究框架，模型库里 LightGBM、XGBoost、CatBoost、MLP、LSTM、Transformer、TRA、HIST 等都有，标准数据集 Alpha158 / Alpha360。
- 美股数据通过 Qlib 的 yahoo 数据采集脚本获取。
- 2026 年一篇论文用带市场状态识别的 LightGBM 在纳斯达克 100 成分股上做 walk-forward 回测，Sharpe 1.18。这个数字仅供参考，个人复现通常打折。

### 通用时序基础模型（不推荐预测方向）

- GIFT-Eval 榜单：Chronos-2 第一，TimesFM-2.5 第二，Moirai-2、TiRex、Toto 在三到五位。
- 但在金融上，2026 年 6 月论文《Pretrained Time-Series Foundation Models for Financial Return Forecasting》在 5 只美股上对比随机游走，结论是增益"小而稀疏"，多数情况统计上不显著。
- 兰卡斯特大学 2025 年 11 月论文《Re(Visiting) Time Series Foundation Models in Finance》：现成的通用时序基础模型零样本和微调都表现差，用金融数据从头预训练的模型才有实质提升。这正是选 Kronos 而不选 Chronos-2 的依据。
- 2026 年 7 月有论文专门对比时序基础模型和 GARCH 类模型预测已实现波动率，基础模型有竞争力。所以它们的合理位置是波动率，不是方向。

### 其他不选的

- FinCast（2025 年 8 月论文）：自称零样本强，但未见开源权重信息，暂不考虑。
- TimeGPT：闭源付费 API，金融上没有证据显示优势。
- Qlib 里的深度模型（TRA、HIST、PatchTST、iTransformer）：数据量和调参要求高，先跑通 LightGBM 基线再说。
- 强化学习交易 agent：跳过。

## Laya 特征怎么进模型

对每条新闻或公告，问 Laya 几个问题，拿到概率：

- p_利好、p_利空（choice 三选一）
- p_并购、p_业绩预警、p_监管处罚、p_高管变动（各一个 noul）
- 重要度 0-4（score）

按票、按日聚合：当日求和或取最大，再做 3 到 5 日指数衰减。得到每票每日约 8 个特征，直接拼进 LightGBM 的特征矩阵。

前提是要先微调 Laya：用大模型在几千条中英文财经新闻上打标，再训练。零样本不可用，这一点上一份评估已经写明。

## 硬件与数据成本

- LightGBM 和 Kronos-small：Mac CPU 可跑。Kronos-base 建议有 GPU。
- Laya 推理：CPU 可跑，日频新闻量完全够用；微调需要 GPU，租一台 T4 数小时即可。
- 行情数据：IBKR API 或 yahoo 免费日线。
- 新闻数据：SEC EDGAR 8-K 免费；IBKR API 自带部分新闻源；Polygon、Benzinga、Finnhub 付费。文本侧的成本主要在这里。

## 分阶段路线

1. **第 1 到 2 周**：Qlib 装好，美股日线入库，Alpha158 + LightGBM 跑通基线，看 RankIC 和扣费后的回测。这一步没有任何 AI 模型，但决定后面所有增量怎么衡量。
2. **第 3 到 4 周**：加 Kronos-small 的 3 个特征，看 RankIC 增量。没增量就删掉，不要恋战。
3. **第 5 到 8 周**：接新闻源，标注几千条，微调 Laya，加事件特征，看增量。
4. **之后**：IBKR 模拟账户跑 1 到 3 个月，再上实盘。

## 现实提醒

- 模型选择只占胜负的一小部分。walk-forward 验证、扣手续费和滑点、防未来函数、样本外测试，这些决定你是赚是亏。
- 三个模型叠加不保证更好。每加一层都要看 IC 增量，增量不显著就砍。
- 任何公开模型的信号都是拥挤的。Kronos 39k stars 意味着很多人在用同样的东西。

## 参考链接

- https://github.com/shiyu-coder/Kronos
- https://huggingface.co/NeoQuasar/Kronos-small
- https://github.com/microsoft/qlib
- https://arxiv.org/abs/2606.27100 （通用时序基础模型预测美股收益，2026-06）
- https://arxiv.org/abs/2511.18578 （Re(Visiting) TSFMs in Finance，2025-11）
- https://arxiv.org/pdf/2607.05291 （时序基础模型预测已实现波动率，2026-07）
- https://arxiv.org/pdf/2510.15821 （Chronos-2）
- https://jonathankinlay.com/2026/02/time-series-foundation-models-for-financial-markets-kronos-and-the-rise-of-pre-trained-market-models/
