# 2llmchat
2個LLM 對話 一個當評估
患者：我已經做很多事情了 但我的老闆扔乃不滿意 我實在不知道該怎麼辦

B LLM分析結果：{
  "emotion": "挫折",
  "question": "您覺得老闆的期待和您的理解有什麼不同嗎？"
}

A LLM回覆：聽起來您現在感到有些挫折呢。不知道是否方便聊聊，您覺得老闆的期待和您實際完成的工作之間，可能存在哪些認知上的差異呢？

# 心理諮商對話系統 (Psychological Counseling Dialogue System)

這是一個基於 LLM 的心理諮商對話系統，通過分析案主的分享，提供專業的諮商回應和評估。系統採用多層次回應策略，包含情緒反映、同理支持和探索引導等元素。

## 功能特點

- 🎯 多層次回應生成
  - 情緒反映
  - 同理支持
  - 深度探索
  - 整合回應

- 🤖 專業督導評估
  - 回應方式分析
  - 策略建議
  - 詳細評估說明

- 💡 智能分析
  - 情緒識別
  - 需求評估
  - 回應策略優化

## 系統需求

- Python 3.8+
- Ollama API
- 相關 Python 套件

## 安裝步驟

1. 克隆專案
```bash
git clone https://github.com/yourusername/counseling-dialogue-system.git
cd counseling-dialogue-system
```

2. 安裝依賴
```bash
pip install -r requirements.txt
```

3. 確保 Ollama 服務運行中
```bash
# 檢查 Ollama 狀態
ollama list
```

## 使用方法

1. 運行系統
```bash
python twollmchat.py
```

2. 開始對話：
```
歡迎進行心理諮商對話 (輸入 'exit' 或 'quit' 結束對話)

案主：最近工作壓力很大...

【循序漸進回應】
A. 情緒回應：我聽到你感到很疲憊和挫折。
B. 情緒+同理：這段時間的工作壓力和失眠，確實讓人感到很無力。
...
```

## 系統架構

### 核心組件

1. ResponseAnalyzer 類
   - 回應解析
   - 格式驗證
   - 預設值處理

2. LLM 分析器
   - 情緒分析
   - 回應生成
   - 策略評估

### 回應層次

1. 循序漸進回應
   - 情緒回應
   - 情緒+同理
   - 情緒+同理+支持
   - 完整回應

2. 整合回應
   - 綜合性理解
   - 問題探索

## 自定義配置

可以通過修改以下參數來自定義系統行為：

1. 回應長度
```python
MAX_RESPONSE_LENGTH = 40  # 單個回應最大字數
```

2. 評估內容
```python
DEFAULT_EVALUATION = {
    'better_approach': '循序漸進',
    'reason': '...'
}
```

## 開發指南

### 添加新功能

1. 在 ResponseAnalyzer 類中添加新方法
```python
def new_analysis_method(self, input_data):
    # 實現新的分析方法
    pass
```

2. 更新主程序
```python
analysis_result = analyzer.new_analysis_method(input_data)
```

### 修改回應策略

在 system_prompt 中調整提示詞：
```python
system_prompt = """
修改回應策略的提示詞...
"""
```

## 常見問題

1. Q: 系統回應出現英文怎麼辦？
   A: 檢查 system_prompt 的語言設定，確保指定使用繁體中文。

2. Q: 評估部分內容不完整？
   A: 系統會自動補充預設的評估內容，確保回應完整性。

