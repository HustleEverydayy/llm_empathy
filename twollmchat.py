import ollama
import json
import logging

# 設定日誌
logging.basicConfig(
    level=logging.ERROR,
    filename='counseling_system.log',
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
    """B LLM 作為專業督導，分析案主狀態並提供諮商方向"""
    system_prompt = """你是一位資深的心理諮商督導，擁有豐富的臨床經驗。請針對案主的陳述，從以下四個面向進行專業分析：

1. 覺察案主的情緒狀態 (格式: Emotion: {描述})
- 辨識核心情緒和次要情緒
- 關注情緒的強度和變化
- 觀察情緒背後的需求

2. 提供專業的同理回應 (格式: Empathy: {回應})
請依據專業諮商原則：
- 準確反映（reflect）案主的感受，避免解釋或評斷
- 同理案主的處境和情緒體驗
- 展現真誠的理解和接納
- 留意案主未明說的需求
- 避免過早給建議或安慰

3. 掌握本次對話重點 (格式: Summary: {摘要})
- 摘要案主的核心議題
- 注意反覆出現的主題
- 不超過15字

4. 設計探索性問題 (格式: Question: {問題})
根據專業諮商技巧：
- 使用開放式問題
- 協助案主深入自我探索
- 聚焦在案主的主觀經驗
- 避免引導或暗示性問題

回應示例：
Emotion: 經歷深層的無力感和挫折，伴隨著對自我價值的懷疑

Empathy: 我聽到您在工作中投入了許多心力，但這些努力似乎沒有得到預期的回應，讓您感到很受傷。在這樣反覆付出卻得不到認可的過程中，您一定感到很孤單和無助。

Summary: 工作努力未獲肯定，影響自我價值感

Question: 能請您分享，在這些努力的過程中，對您來說最具挑戰的部分是什麼？"""

    try:
        response = client.chat(model='llama3:8b', messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_input}
        ])
        return parse_b_response(response['message']['content'])
    except Exception as e:
        logging.error(f"B LLM 分析過程發生錯誤: {str(e)}")
        return {
            'emotion': '需要進一步了解案主狀態',
            'empathy': '我感受到您正在經歷一些困擾',
            'summary': '需要深入了解案主狀況',
            'question': '能和我多分享一些您的想法嗎？'
        }

def parse_b_response(response):
    """解析督導的專業分析"""
    result = {
        'emotion': "需要進一步了解案主狀態",
        'empathy': "我感受到您正在經歷一些困擾",
        'summary': "需要深入了解案主狀況",
        'question': "能和我多分享一些您的想法嗎？"
    }
    
    try:
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Emotion:'):
                result['emotion'] = line.split(':', 1)[1].strip()
            elif line.startswith('Empathy:'):
                result['empathy'] = line.split(':', 1)[1].strip()
            elif line.startswith('Summary:'):
                result['summary'] = line.split(':', 1)[1].strip()
            elif line.startswith('Question:'):
                result['question'] = line.split(':', 1)[1].strip()
        
        # 驗證所有欄位都有值
        for key, value in result.items():
            if not value or value.isspace():
                result[key] = result[key]  # 使用預設值
            
    except Exception as e:
        logging.error(f"解析回應時發生錯誤: {str(e)}")
        logging.error(f"原始回應: {response}")
    
    return result

def llm_a_response(user_input, analysis):
    """A LLM 作為諮商師，根據督導分析提供專業回應"""
    system_prompt = f"""你是一位專業的心理諮商師，請根據以下分析結果，提供專業的諮商回應：

案主狀態分析：
1. 情緒狀態：{analysis['emotion']}
2. 同理要點：{analysis['empathy']}
3. 核心議題：{analysis['summary']}
4. 待探索方向：{analysis['question']}

諮商回應準則：
1. 展現真誠的同理心和接納
2. 避免過早給建議或解決方案
3. 使用合適的諮商技巧：
   - 準確反映（reflect）案主的感受
   - 摘要重述案主的經驗
   - 同理案主的處境
   - 適時使用開放性問題深化探索
4. 回應長度控制在100字以內
5. 保持專業、溫和且支持的語氣
6. 避免：
   - 評斷或解釋案主的感受
   - 過早提供建議
   - 使用安慰性或說教性的語言
   - 轉移話題或忽視案主的情緒
"""

    try:
        response = client.chat(model='phi3:14b', messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_input}
        ])
        return response['message']['content']
    except Exception as e:
        logging.error(f"A LLM 回應過程發生錯誤: {str(e)}")
        return "我聽到您正在經歷一些困擾。能否和我多分享一些您的想法和感受？"

def main():
    print("歡迎進行心理諮商對話 (輸入 'exit' 或 'quit' 結束對話)\n")
    
    while True:
        try:
            user_input = input("\n案主：").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input:
                print("請分享您的想法。")
                continue
                
            # 督導分析階段
            print("\n正在理解您的分享...\n")
            b_analysis = llm_b_analyzer(user_input)
            print("督導分析：")
            print(f"情緒狀態：{b_analysis['emotion']}")
            print(f"同理重點：{b_analysis['empathy']}")
            print(f"核心議題：{b_analysis['summary']}")
            print(f"探索方向：{b_analysis['question']}\n")
            
            # 諮商師回應階段
            print("正在組織回應...\n")
            a_response = llm_a_response(user_input, b_analysis)
            print(f"諮商師：{a_response}\n")
            
        except KeyboardInterrupt:
            print("\n諮商對話已結束")
            break
        except Exception as e:
            logging.error(f"主程序發生錯誤: {str(e)}")
            print("\n系統發生錯誤，請重試。")

if __name__ == "__main__":
    main()
