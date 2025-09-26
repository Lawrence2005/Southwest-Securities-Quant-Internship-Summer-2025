# 📓 Alpha Strategy Log

## vwap_strat1
- **Idea**: 比较最后收盘价 vs 当日 VWAP  
- **Formula**:  
  signal = -1 if Close > VWAP else +1
- **Data Used**: 当日最后一笔 Close，当日 VWAP 
- **Performance**:  
  - Sharpe(raw): 1.25
  - Win Rate: 53.5%
- **Verdict**: ✅ Good  
- **Notes**: ___  

---

## vwap_strat2
- **Idea**: 几何均值 √(High*Low) vs 当日 VWAP 
- **Formula**:  
  signal = -1 if 几何均值 > VWAP else +1 
- **Data Used**: 当日 High, Low, VWAP
- **Performance**:  
  - Sharpe(raw): 0.1
  - Win Rate: 52.5%
- **Verdict**: ❌ Bad  
- **Notes**: ___  

---

## rank_strat
- **Idea**: 首个开盘价 vs 最后收盘价 cross-sectional rank 的反转逻辑
- **Formula**:
  factor = Open / Close - 1 \
  signal = -1 if 因子值排名低 else +1 
- **Data Used**: 当日 Open, Close（多只股票横截面） 
- **Performance**:  
  - Sharpe(raw): 0.65
  - Win Rate: 50.2%  
- **Verdict**: ➖ Mid
- **Notes**: ___  

---

## correlation_strat
- **Idea**: 过去10日 Open 与 Volume 的相关性，乘 -1  
- **Formula**:  
  factor_t = -Corr(Open_{t-9..t}, Vol_{t-9..t}) \
  signal = -1 if factor_t > 0 else +1
- **Performance**:  
  - Sharpe(raw): 0.18
  - Win Rate: 51.8%
- **Verdict**: ❌ Bad  
- **Notes**: ___  

---

## amplitude_strat
- **Idea**: 当日振幅 (High - Low) / Open
- **Formula**:  
  signal = -1 if 当日振幅 > 中位振幅 else +1
- **Performance**:  
  - Sharpe(raw): 0.84
  - Win Rate: 51.9%
- **Verdict**: 👍 Ok
- **Notes**: 真实运用需要实时更新中位值

---

## close_pos_strat
- **Idea**: 收盘在当日振幅区间中的位置  
- **Formula**:  
  Pos = (Close - Low) / (High - Low) \
  signal = -1 if 收盘价位于当日振幅前40% else +1
- **Performance**:  
  - Sharpe(raw): 1.15
  - Win Rate: 52.9%
- **Verdict**: ✅ Good  
- **Notes**: ___

---

## volume_strat1
- **Idea**: 当日总成交量 vs 过去 N 日均值  
- **Formula**:  
  ratio = Vol_t / MA_N(Vol) \
  signal = -1 if ratio > 1.5 else +1
- **Performance**:  
  - Sharpe(raw): 0.1
  - Win Rate: 52.9%
- **Verdict**: ❌ Bad 
- **Notes**: 真实运用需要实时跟踪总成交量

---

## reversal_strat
- **Idea**: 当日涨跌幅 (Close - Open) 反转
- **Formula**:  
  signal = -1 if 当日涨了 else +1
- **Performance**:  
  - Sharpe(raw): 0.97
  - Win Rate: 52.6%
- **Verdict**: 👍 Ok 
- **Notes**: ___

---

## rank_reversal_strat
- **Idea**: 当日涨跌幅 cross-sectional rank 的反转逻辑
- **Formula**:  
  signal = -1 if 当日涨幅排名高 else +1
- **Performance**:  
  - Sharpe(raw): 0.5
  - Win Rate: 51.7%
- **Verdict**: ➖ Mid  
- **Notes**: ___

---

## momentum_strat
- **Idea**: ln(Open / Close) → 趋势方向
- **Formula**:  
  signal = -1 if 当日ln值小于昨日ln值 else +1
- **Performance**:  
  - Sharpe(raw): 0.74
  - Win Rate: 51.7%
- **Verdict**: 👍 Ok
- **Notes**: ___

---

## volume_strat2
- **Idea**: 当日主要交易量价格 vs VWAP
- **Formula**:  
  signal = -1 if 当日大部分交易量价格小于VWAP else +1
- **Performance**:  
  - Sharpe(raw): 0.25
  - Win Rate: 51.7%
- **Verdict**: ❌ Bad
- **Notes**: ___  

---

## volume_strat3
- **Idea**: 在日中和日末RVOL攀升 反转（潜在利润实现点）
- **Formula**:  
  signal = -1 if 在任意时间点RVOL极大且价格上涨 else +1
- **Performance**:  
  - Sharpe(raw): 0.7
  - Win Rate: 51.85%
- **Verdict**: 👍 Ok
- **Notes**: ___

---

## return_strat
- **Idea**: 当日收益正负不对称性 反转（潜在利润实现点）
- **Formula**:  
  signal = -1 if 正收益显著大于负收益 else +1
- **Performance**:  
  - Sharpe(raw): 0.59
  - Win Rate: 52.1%
- **Verdict**: ➖ Mid
- **Notes**: ___

---

## spread_strat
- **Idea**: 当日首开盘价与末收盘价差 vs 当日振幅 趋势方向
- **Formula**:  
  signal = -1 if 净价差相较振幅小且盘末跌 或净价差相较振幅大且盘末涨 else +1
- **Performance**:  
  - Sharpe(raw): 0.32
  - Win Rate: 50.6%
- **Verdict**: ❌ Bad
- **Notes**: ___

---

## momentum_strat2
- **Idea**: 当日盘末收益 反转
- **Formula**:  
  signal = -1 if 盘末见涨 else +1
- **Performance**:  
  - Sharpe(raw): 1.26
  - Win Rate: 52.2%
- **Verdict**: ✅ Good 
- **Notes**: ___  

---

## liquidity_strat
- **Idea**: 当日成交量骤降却价格波动 反转
- **Formula**:  
  signal = -1 if 成交量小且价格涨幅>2% 或成交量正常且价格涨了 else +1
- **Performance**:  
  - Sharpe(raw): 0.97
  - Win Rate: 52.6%
- **Verdict**: 👍 Ok
- **Notes**: ___  

---

## momentum_strat3
- **Idea**: 当日盘末 vs 其他期价格走势 反转
- **Formula**:  
  signal = -1 if 盘末价格走势与其他期价格走势相反 或 走势相同价格且涨了 else +1
- **Performance**:  
  - Sharpe(raw): 0.38
  - Win Rate: 53.4%
- **Verdict**: ❌ Bad
- **Notes**: ___  

---

## volume_strat4
- **Idea**: 高价与成交量相关性和价格波动性 
- **Formula**:  
  signal = -1 if 价格走势与成交量趋势不符 异或 市场波动性小 else +1
- **Performance**:  
  - Sharpe(raw): 0.2
  - Win Rate: 50.4%
- **Verdict**: ❌ Bad
- **Notes**: ___  

---

## vwap_strat3
- **Idea**: 收益、VWAP、收盘回落 反转
- **Formula**:  
  signal = +1 if 收益为负，流动性高，且收盘回落大 else -1
- **Performance**:  
  - Sharpe(raw): 1.1
  - Win Rate: 51.9%
- **Verdict**: ✅ Good
- **Notes**: ___  

---

## momentum_strat4
- **Idea**: 成交量 vs 价格波动
- **Formula**:  
  signal = +1 if 成交量高但价格相对稳定 else -1
- **Performance**:  
  - Sharpe(raw): 0.68
  - Win Rate: 51.8%
- **Verdict**: 👍 Ok
- **Notes**: ___  

---

## momentum_strat5
- **Idea**: 收盘价走势与日内价格走势
- **Formula**:  
  signal = +1 if 收盘价见大涨且日内也大涨 else -1
- **Performance**:  
  - Sharpe(raw): 0.65
  - Win Rate: 53.2%
- **Verdict**: 👍 Ok
- **Notes**: ___  

---

## close_pos_strat2
- **Idea**: 买卖趋势 vs 价格波动
- **Formula**:  
  signal = +1 if 价格回落 或 无明显最终趋势 else -1
- **Performance**:  
  - Sharpe(raw): 0.65
  - Win Rate: 53.2%
- **Verdict**: 👍 Ok
- **Notes**: ___  

---

## 🗂 Summary Table

| Strategy            | Sharpe(raw) | Win Rate       | Verdict | Notes  |
|---------------------|-------------|----------------|---------|--------|
| vwap_strat1         | 1.25        | 53.5%          | ✅ Good | ___    |
| vwap_strat2         | 0.1         | 52.5%          | ❌ Bad  | ___    |
| rank_strat          | 0.65        | 50.2%          | ➖ Mid  | ___    |
| correlation_strat   | 0.18        | 51.8%          | ❌ Bad  | ___    |
| amplitude_strat     | 0.84        | 51.9%          | 👍 Ok   | ___    |
| close_pos_strat1    | 1.15        | 52.9%          | ✅ Good | ___    |
| volume_strat1       | 0.1         | 52.9%          | ❌ Bad  | ___    |
| reversal_strat      | 0.97        | 52.6%          | 👍 Ok   | ___    |
| rank_reversal_strat | 0.5         | 51.7%          | ➖ Mid  | ___    |
| momentum_strat      | 0.74        | 51.7%          | 👍 Ok   | ___    |
| volume_strat2       | 0.25        | 51.7%          | ❌ Bad  | ___    |
| volume_strat3       | 0.7         | 51.85%         | 👍 Ok   | ___    |
| return_strat        | 0.59        | 52.1%          | ➖ Mid  | ___    |
| spread_strat        | 0.32        | 50.6%          | ❌ Bad  | ___    |
| momentum_strat2     | 1.26        | 52.2%          | ✅ Good | ___    |
| liquidity_strat     | 0.97        | 52.6%          | 👍 Ok   | ___    |
| momentum_strat3     | 0.38        | 53.4%          | ❌ Bad  | ___    |
| volume_strat4       | 0.2         | 50.4%          | ❌ Bad  | ___    |
| vwap_strat3         | 1.1         | 51.9%          | ✅ Good | ___    |
| momentum_strat4     | 0.68        | 51.8%          | 👍 Ok   | ___    |
| momentum_strat5     | 0.65        | 53.2%          | 👍 Ok   | ___    |