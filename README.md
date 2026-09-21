# new-chickenshit-Trader（靓鸡屎交易员）

美股日频量化研究仓库。第一阶段目标：把「行情 → 因子 → 模型 → 样本外评估 → 扣费回测」这条基线跑通并拿到基线数字，之后每加一层（Kronos 序列特征、Laya 文本事件特征）都只看相对基线的增量。方案见 [docs/02_Laya搭配预测模型方案.md](docs/02_Laya搭配预测模型方案.md)。

```
日线 OHLCV ──┬─→ Alpha158 因子（Qlib 内置）─────────────────┐
             └─→ Kronos 采样 → 序列特征（阶段 2）──────────┤
                                                            ↓
新闻/公告/8-K → Laya → 事件概率（阶段 3）──────────────────→ LightGBM 预测收益排名
                                                            ↓
                              top-k 多头，波动率目标仓位 → IBKR（模拟盘先行）
```

## 现状（阶段 1：基线）

| 环节 | 实现 |
|---|---|
| 股票池 | 当前标普 500 成分股（Wikipedia），基准 SPY。**有幸存者偏差**，见报告 |
| 行情 | Yahoo Finance 日线，2010 年至今；按 Qlib 官方 yahoo collector 的规则归一化（复权因子、异常修复） |
| 存储 | Qlib 二进制格式 `data/qlib/us_sp500`（用 vendored 的 `scripts/dump_bin.py` 生成） |
| 特征 | Qlib Alpha158（158 个价量因子） |
| 标签 | 5 日前向收益，t+1 收盘进、t+6 收盘出 |
| 模型 | LightGBM（Qlib 公开基准参数），验证集早停 |
| 评估 | 测试期日度 IC / RankIC / ICIR；TopkDropout 多头回测，买卖各 5 bp 成本，对比 SPY |
| 输出 | `reports/baseline_lgb_<日期>.md` + `.json` + 曲线图 |

## 环境

macOS Apple Silicon，Python 3.12（pyqlib 0.9.7 只有到 3.12 的轮子）。

```bash
brew install libomp            # LightGBM 需要 OpenMP
make venv                      # 建 .venv 并安装 requirements.txt
```

## 运行

```bash
make data        # 下载 + 归一化 + 转 Qlib 格式（约 5 分钟；Yahoo 限流时会自动退避）
make baseline    # 训练 + 评估 + 回测，写 reports/
make test        # 归一化逻辑的单元测试
```

两个脚本都读 `configs/baseline_lgb.yaml`；改标签、切分、模型参数、成本都在那里。

## 目录

```
configs/     baseline_lgb.yaml        全部参数
ljs/         universe.py              标普 500 名单（Wikipedia，带缓存）
             data_yahoo.py            Yahoo 下载（限流退避、Stooq 兜底）+ Qlib 风格归一化
             dump_qlib.py             调 dump_bin、写股票池文件、写未来交易日历
             baseline_lgb.py          Alpha158 + LightGBM + IC + 回测 + 报告
scripts/     01_build_data.py         数据构建入口
             02_run_baseline.py       基线入口
             dump_bin.py              vendored from microsoft/qlib（MIT）
docs/        01_ 模型评估、02_ 方案
reports/     基线报告（提交到仓库）
data/        原始/归一化/Qlib 数据、模型产物（不提交，make data 可重建）
tests/       单元测试
```

## 已知坑

- Qlib 在 macOS 上用多进程算因子，**入口脚本必须有 `if __name__ == "__main__":` 保护**，否则子进程重新导入主模块导致死锁。
- mlflow ≥ 3.16 默认禁用文件后端，Qlib 训练时会报错；代码里已设 `MLFLOW_ALLOW_FILE_STORE=true`。
- Qlib 回测需要 `calendars/day_future.txt`（交易日历延伸到未来），否则在数据最后一天越界；`make data` 会自动生成。
- Yahoo 对并发请求限流（HTTP 429）；下载已改为串行、每批 20 只、遇限流指数退避。

## 免责

研究代码，不是投资建议，不保证任何收益。任何回测数字在上实盘前都应默认打折。
