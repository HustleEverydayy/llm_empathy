import ollama
import logging

# 設定日誌
logging.basicConfig(level=logging.ERROR, filename='counseling_system.log')
logger = logging.getLogger(__name__)

# 初始化 Ollama 客戶端
client = ollama.Client()

def get_step_one(user_input: str) -> str:
    """第一步：情緒辨識"""
    prompt = """你是專業心理諮商師，請只用繁體中文回應。
    
第一步 - 情緒辨識：
請僅反映案主當前的情緒狀態，不要加入其他內容。

範例：「我聽到你現在感到很疲憊和無力」

限制：
- 只做情緒辨識
- 不要提供建議
- 不要進行評價
- 字數限制在30字內"""

    try:
        response = client.chat(model='llama3:8b', messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': user_input}
        ])
        return response['message']['content']
    except Exception as e:
        logger.error(f"第一步回應錯誤: {str(e)}")
        return "系統暫時無法回應"

def get_step_two(step_one: str, user_input: str) -> str:
    """第二步：情緒辨識 + 同理"""
    prompt = f"""你是專業心理諮商師，請只用繁體中文回應。
    
第二步 - 在第一步的基礎上，加入同理：
前一步回應：{step_one}

請在這個基礎上，加入對案主處境的理解。

範例：
第一步：「我聽到你現在感到很疲憊和無力」
第二步：「我聽到你現在感到很疲憊和無力，這種持續工作卻看不到成果的感受確實讓人感到挫折」

限制：
- 必須包含第一步的內容
- 加入對處境的理解
- 自然地銜接兩個部分"""

    try:
        response = client.chat(model='llama3:8b', messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': user_input}
        ])
        return response['message']['content']
    except Exception as e:
        logger.error(f"第二步回應錯誤: {str(e)}")
        return step_one

def get_step_three(step_two: str, user_input: str) -> str:
    """第三步：情緒辨識 + 同理 + 支持"""
    prompt = f"""你是專業心理諮商師，請只用繁體中文回應。
    
第三步 - 在第二步的基礎上，加入支持：
前一步回應：{step_two}

請在這個基礎上，加入支持性的回應。

範例：
第二步：「我聽到你現在感到很疲憊和無力，這種持續工作卻看不到成果的感受確實讓人感到挫折」
第三步：「我聽到你現在感到很疲憊和無力，這種持續工作卻看不到成果的感受確實讓人感到挫折。你的這些感受都是合理的，每個人在這樣的處境下都會感到困擾」

限制：
- 必須包含第二步的內容
- 加入支持性話語
- 自然地銜接各個部分"""

    try:
        response = client.chat(model='llama3:8b', messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': user_input}
        ])
        return response['message']['content']
    except Exception as e:
        logger.error(f"第三步回應錯誤: {str(e)}")
        return step_two

def get_step_four(step_three: str, user_input: str) -> str:
    """第四步：完整回應 + 蘇格拉底式提問"""
    prompt = f"""你是專業心理諮商師，請只用繁體中文回應。
    
第四步 - 在第三步的基礎上，加入蘇格拉底式提問：
前一步回應：{step_three}

請在這個基礎上，加入一個探索性的蘇格拉底式提問。

蘇格拉底式提問的特點：
- 開放性：鼓勵深入思考
- 引導性：幫助發現核心議題
- 不帶評判：避免暗示或引導特定答案

提問示例：
- 「你覺得是什麼讓這些壓力特別難以承受？」
- 「當你提到不夠好的時候，你心中的標準是什麼？」
- 「你能分享更多關於這些感受的想法嗎？」

限制：
- 必須包含第三步的內容
- 結尾加入一個探索性提問
- 自然地銜接所有部分"""

    try:
        response = client.chat(model='llama3:8b', messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': user_input}
        ])
        return response['message']['content']
    except Exception as e:
        logger.error(f"第四步回應錯誤: {str(e)}")
        return step_three

def get_integrated_response(user_input: str) -> str:
    """獲取整合性回應"""
    prompt = """你是專業心理諮商師，請只用繁體中文，提供一個整合性回應。

回應需要包含：
1. 情緒辨識：準確反映案主的情緒
2. 同理支持：表達對案主處境的理解
3. 蘇格拉底式提問：引導案主深入思考

範例：
「我理解你現在感到非常疲憊和無力，這種持續付出卻看不到回報的感受確實讓人感到挫折。面對這樣的狀況感到困擾都是很自然的。你覺得是什麼讓這些壓力特別讓你難以承受呢？」

限制：
- 只使用繁體中文
- 回應要流暢自然
- 必須以蘇格拉底式提問結束
- 總字數不超過100字"""

    try:
        response = client.chat(model='llama3:8b', messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': user_input}
        ])
        return response['message']['content']
    except Exception as e:
        logger.error(f"獲取整合回應錯誤: {str(e)}")
        return "系統暫時無法回應"

def evaluate_responses(progressive: str, integrated: str) -> str:
    """評估兩種回應方式"""
    prompt = f"""你是資深心理諮商督導，請只用繁體中文評估以下兩種回應方式：

循序漸進回應：
{progressive}

整合回應：
{integrated}

請說明：
1. 哪種方式更適合案主
2. 選擇的原因

評估示例：
「建議採用循序漸進方式，因為案主情緒複雜，需要被逐步理解。這種方式能幫助案主感受到被完整理解，同時也讓諮商師能更好地掌握案主的需求變化。」

限制：
- 只使用繁體中文
- 評估要具體明確
- 總字數不超過100字
- 避免任何英文字詞"""

    try:
        response = client.chat(model='llama3:8b', messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': "請評估上述兩種回應方式"}
        ])
        return response['message']['content']
    except Exception as e:
        logger.error(f"評估回應錯誤: {str(e)}")
        return "系統暫時無法進行評估"

def main():
    print("歡迎進行心理諮商對話（輸入 'exit' 或 'quit' 結束對話）\n")
    
    while True:
        try:
            user_input = input("\n案主：").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input:
                print("請分享您的想法。")
                continue
            
            print("\n正在分析回應...\n")
            
            # 獲取循序漸進回應
            print("【循序漸進回應（漸進累積）】")
            step_one = get_step_one(user_input)
            print(f"步驟一：{step_one}")
            
            step_two = get_step_two(step_one, user_input)
            print(f"\n步驟二：{step_two}")
            
            step_three = get_step_three(step_two, user_input)
            print(f"\n步驟三：{step_three}")
            
            step_four = get_step_four(step_three, user_input)
            print(f"\n步驟四：{step_four}")
            
            # 獲取整合回應
            integrated = get_integrated_response(user_input)
            print(f"\n【整合回應（一次性）】\n{integrated}")
            
            # 獲取評估
            evaluation = evaluate_responses(step_four, integrated)
            print(f"\n【督導評估】\n{evaluation}")
            print()
            
        except KeyboardInterrupt:
            print("\n諮商對話已結束")
            break
        except Exception as e:
            logger.error(f"主程序錯誤: {str(e)}")
            print("\n系統發生錯誤，請重試。")

if __name__ == "__main__":
    main()
