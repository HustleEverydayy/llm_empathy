import ollama
import json
import logging

# 設定日誌，只記錄 ERROR 級別以上的訊息到檔案
logging.basicConfig(
    level=logging.ERROR,
    filename='dialogue_system.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(console)

# 初始化 Ollama 客戶端
client = ollama.Client()

def llm_b_analyzer(user_input):
    """B LLM 負責情緒辨識和蘇格拉底式提問指導"""
    system_prompt = """你是一個資深心理諮詢督導，請針對病患的回答執行兩個任務：
1. 用精準的文字描述病患的情緒狀態 (格式: Emotion: {情緒描述})
2. 設計一個蘇格拉底式提問，幫助病患深入思考 (格式: Question: {問題})

回應格式示例：
Emotion: 感到挫折與無力
Question: 您提到老闆不滿意，能具體說說是哪些事情讓您覺得特別困擾嗎？"""

    try:
        response = client.chat(model='llama3:8b', messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_input}
        ])
        return parse_b_response(response['message']['content'])
    except Exception as e:
        logging.error(f"B LLM 分析過程發生錯誤: {str(e)}")
        return {
            'emotion': '情緒不明確',
            'question': '能否多分享一些您的想法？'
        }

def parse_b_response(response):
    """解析 B LLM 的回應"""
    emotion = "情緒不明確"
    question = "能否多分享一些您的想法？"
    
    try:
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Emotion:'):
                parts = line.split(':', 1)
                if len(parts) > 1:
                    emotion = parts[1].strip()
            elif line.startswith('Question:'):
                parts = line.split(':', 1)
                if len(parts) > 1:
                    question = parts[1].strip()
        
        # 驗證輸出
        if not emotion or emotion.isspace():
            emotion = "情緒不明確"
        if not question or question.isspace():
            question = "能否多分享一些您的想法？"
            
    except Exception as e:
        logging.error(f"解析回應時發生錯誤: {str(e)}")
        logging.error(f"原始回應: {response}")
    
    return {'emotion': emotion, 'question': question}

def llm_a_response(user_input, analysis):
    """A LLM 根據 B 的分析生成最終回應"""
    system_prompt = f"""你是一位專業的心理諮詢師，請根據以下資訊組織回應：

1. 患者情緒狀態：{analysis['emotion']}
2. 引導性問題：{analysis['question']}

請注意：
- 首先要表達對患者情緒的理解和同理
- 然後自然地引入引導性問題
- 保持溫暖專業的語氣
- 回應長度控制在100字以內
- 避免生硬的公式化回答"""

    try:
        response = client.chat(model='phi3:14b', messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_input}
        ])
        return response['message']['content']
    except Exception as e:
        logging.error(f"A LLM 回應過程發生錯誤: {str(e)}")
        return "我能感受到您的困擾。您願意多跟我分享一些嗎？這樣我能更好地理解和協助您。"

def main():
    print("歡迎使用心理諮詢助手 (輸入 'exit' 或 'quit' 結束對話)\n")
    
    while True:
        try:
            user_input = input("患者：").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input:
                print("請說出您的困擾。")
                continue
                
            # B LLM 分析階段
            print("\n正在分析回應...\n")
            b_analysis = llm_b_analyzer(user_input)
            print("B LLM 分析結果：")
            print(f"情緒：{b_analysis['emotion']}")
            print(f"引導問題：{b_analysis['question']}\n")
            
            # A LLM 生成回應
            print("正在生成回應...\n")
            a_response = llm_a_response(user_input, b_analysis)
            print(f"諮詢師：{a_response}\n")
            
        except KeyboardInterrupt:
            print("\n對話已結束")
            break
        except Exception as e:
            logging.error(f"主程序發生錯誤: {str(e)}")
            print("\n系統發生錯誤，請重試。")

if __name__ == "__main__":
    main()