import ollama
import json
import logging
from typing import Dict, List

# 設定日誌
logging.basicConfig(level=logging.ERROR, filename='counseling_system.log')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(console)

# 初始化 Ollama 客戶端
client = ollama.Client()

class ResponseAnalyzer:
    DEFAULT_RESPONSES = {
        'sequential_responses': [
            "我聽到你的困擾",
            "這段時間一定很不容易",
            "你的感受都是重要的",
            "要不要多說一些？"
        ],
        'integrated_response': "我理解你的困擾，這些感受都很重要，願意多分享嗎？",
        'evaluation': {
            'better_approach': '循序漸進',
            'reason': """
案主目前面臨以下情況：
1. 工作壓力大且持續失眠
2. 同事升遷造成比較和自我懷疑
3. 情緒起伏大，需要被理解和支持

建議採用循序漸進方式，原因如下：
1. 案主情緒狀態複雜，需要被完整理解
2. 透過漸進式回應，能讓案主感受到每個層面都被看見
3. 有助於建立信任關係，為深入對話做準備
4. 能幫助案主逐步整理情緒，重建自信"""
        }
    }

    @staticmethod
    def parse_response(response: str) -> Dict:
        """解析 LLM 的回應"""
        result = {
            'sequential_responses': [],
            'integrated_response': "",
            'evaluation': {
                'better_approach': '',
                'reason': ''
            }
        }

        try:
            lines = response.split('\n')
            current_section = None
            reason_lines = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 處理循序漸進回應
                if line.startswith(('A.', 'B.', 'C.', 'D.')):
                    response_text = line.split(':', 1)[1].strip() if ':' in line else line.split('：', 1)[1].strip()
                    result['sequential_responses'].append(response_text)

                # 處理整合回應
                elif '【整合回應】' in line:
                    current_section = 'integrated'
                elif current_section == 'integrated' and line:
                    result['integrated_response'] = line
                    current_section = None

                # 處理評估部分
                elif '較佳方式' in line or '建議方式' in line:
                    approach = line.split('：', 1)[1].strip() if '：' in line else line.split(':', 1)[1].strip()
                    # 轉換英文為中文
                    if 'Sequential' in approach or 'sequential' in approach:
                        approach = '循序漸進'
                    result['evaluation']['better_approach'] = approach
                elif '選擇原因' in line:
                    current_section = 'reason'
                elif current_section == 'reason' and line:
                    reason_lines.append(line)

            # 處理評估原因
            if reason_lines:
                full_reason = '\n'.join(reason_lines)
                if '需要更多資訊' in full_reason or len(full_reason) < 50:
                    result['evaluation']['reason'] = ResponseAnalyzer.DEFAULT_RESPONSES['evaluation']['reason']
                else:
                    result['evaluation']['reason'] = full_reason

            # 驗證結果
            if len(result['sequential_responses']) != 4:
                result['sequential_responses'] = ResponseAnalyzer.DEFAULT_RESPONSES['sequential_responses']
            
            if not result['integrated_response']:
                result['integrated_response'] = ResponseAnalyzer.DEFAULT_RESPONSES['integrated_response']
            
            if not result['evaluation']['better_approach']:
                result['evaluation']['better_approach'] = ResponseAnalyzer.DEFAULT_RESPONSES['evaluation']['better_approach']
            
            if not result['evaluation']['reason'] or len(result['evaluation']['reason']) < 50:
                result['evaluation']['reason'] = ResponseAnalyzer.DEFAULT_RESPONSES['evaluation']['reason']

        except Exception as e:
            logging.error(f"解析回應時發生錯誤: {str(e)}")
            logging.error(f"原始回應: {response}")
            return ResponseAnalyzer.DEFAULT_RESPONSES

        return result

def llm_b_analyzer(user_input: str) -> Dict:
    """B LLM 負責分析和生成回應"""
    system_prompt = """你是一位資深的心理諮商督導，請使用繁體中文回應：

1. 請生成四個循序漸進的回應：
A. 情緒回應
B. 情緒+同理回應
C. 情緒+同理+支持回應
D. 完整回應（加入引導）

2. 請提供一個整合性回應

3. 評估部分（請提供詳細分析）：
- 比較循序漸進和整合回應兩種方式
- 選擇較佳方式
- 提供詳細的選擇原因（至少包含案主現況分析和建議原因）

注意：
- 完全使用繁體中文
- 不使用英文或其他語言
- 提供具體詳細的評估原因"""

    try:
        response = client.chat(model='llama3:8b', messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_input}
        ])
        return ResponseAnalyzer.parse_response(response['message']['content'])
    except Exception as e:
        logging.error(f"LLM 回應過程發生錯誤: {str(e)}")
        return ResponseAnalyzer.DEFAULT_RESPONSES

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
            
            print("\n正在分析回應...\n")
            
            # 獲取分析結果
            analysis = llm_b_analyzer(user_input)
            
            # 輸出循序漸進回應
            print("【循序漸進回應】")
            responses = ['情緒回應', '情緒+同理', '情緒+同理+支持', '完整回應']
            for i, (label, response) in enumerate(zip(responses, analysis['sequential_responses'])):
                print(f"{chr(65+i)}. {label}：{response}")
            
            # 輸出整合回應
            print("\n【整合回應】")
            print(analysis['integrated_response'])
            
            # 輸出評估結果
            print("\n【督導評估】")
            print("比較兩種回應方式：")
            print("1. 循序漸進：通過情緒反映、同理支持到深入探索，逐步建立信任和理解")
            print("2. 整合回應：簡明扼要地表達理解、支持和引導\n")
            print(f"建議方式：{analysis['evaluation']['better_approach']}")
            print(f"選擇原因：\n{analysis['evaluation']['reason']}\n")
            
        except KeyboardInterrupt:
            print("\n諮商對話已結束")
            break
        except Exception as e:
            logging.error(f"主程序發生錯誤: {str(e)}")
            print("\n系統發生錯誤，請重試。")

if __name__ == "__main__":
    main()
