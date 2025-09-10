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

## rank_st
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

## correlation_st
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

## amplitude_st
- **Idea**: 当日振幅 (High - Low) / Open
- **Formula**:  
  signal = -1 if 当日振幅 > 中位振幅 else +1
- **Performance**:  
  - Sharpe(raw): 0.84
  - Win Rate: 51.9%
- **Verdict**: 👍 Ok
- **Notes**: 真实运用需要实时更新中位值

---

## close_pos_st
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

## volume_st
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

## reversal_st
- **Idea**: 当日涨跌幅 (Close - Open) 反转
- **Formula**:  
  signal = -1 if 当日涨了 else +1
- **Performance**:  
  - Sharpe(raw): 0.97
  - Win Rate: 52.6%
- **Verdict**: 👍 Ok 
- **Notes**: ___

---

## rank_reversal_st
- **Idea**: 当日涨跌幅 cross-sectional rank 的反转逻辑
- **Formula**:  
  signal = -1 if 当日涨幅排名高 else +1
- **Performance**:  
  - Sharpe(raw): 0.5
  - Win Rate: 51.7%
- **Verdict**: ➖ Mid  
- **Notes**: ___  

---

## momentum_st
- **Idea**: ln(Open / Close) → 趋势方向
- **Formula**:  
  signal = -1 if 当日ln值小于昨日ln值 else +1
- **Performance**:  
  - Sharpe(raw): 0.74
  - Win Rate: 51.7%
- **Verdict**: 👍 Ok
- **Notes**: ___  

---

## 🗂 Summary Table

| Strategy         | Sharpe(raw) | Win Rate       | Verdict | Notes  |
|------------------|-------------|----------------|---------|--------|
| vwap_strat1      | 1.25        | ___            | ✅ Good | ___    |
| vwap_strat2      | 0.1         | -              | ❌ Bad  | ___    |
| rank_st          | ___         | -              | ❌ Bad  | ___    |
| correlation_st   | ___         | ___            | ➖ Mid  | ___    |
| amplitude_st     | ___         | -              | ➖ Mid  | ___    |
| close_pos_st     | ___         | ___            | ✅ Good | ___    |
| volume_st        | ___         | -              | ➖ Mid  | ___    |
| reversal_st      | ___         | ___            | ➖ OK   | ___    |
| rank_reversal_st | ___         | -              | ❌ Bad  | ___    |
| momentum_st      | ___         | -              | ❌ Bad  | ___    |
