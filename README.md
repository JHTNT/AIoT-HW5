# AI vs Human Detector

簡單的 Streamlit 應用，利用 Hugging Face `openai-community/roberta-base-openai-detector` 對輸入文本估計 AI 生成 / 人類撰寫的可能性。

## 功能

- 即時輸入文本並輸出 AI% / Human% 與結論。
- 顯示模型原始分數、文字統計（字元數、單詞數、平均單詞長度）。
- 完全在 CPU 上推論，方便一般環境快速試用。

## 安裝

```bash
pip install -r requirements.txt
```

## 執行

```bash
streamlit run app.py
```

啟動後依指示開啟本機網址（通常 <http://localhost:8501）。>

## 使用

1. 貼上或輸入要判斷的文本。
2. 點擊「判斷」，查看 AI / Human 概率、進度條與推測結論。
3. 如需重新輸入，按「清空」。

## 其他

- 模型載入時若看到 `Some weights ... were not used` 警告屬正常，因為 pooler 權重未使用於分類頭。
- 若想靜音 transformers 訊息，可在 `app.py` 開頭加入：

  ```python
  from transformers import logging
  logging.set_verbosity_error()
  ```
