import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import os
import re
from tqdm import tqdm
import pandas as pd
import gc
import time

class FourStageGemmaCounselor:
    def __init__(self, model_path="D:\\FinetuneGemma\\gemma-3-27b-it"):
        """初始化本地 Gemma 模型介面，針對A6000優化。"""
        print(f"正在從本地路徑 {model_path} 載入模型...")
        
        # system prompt 全程保持同理心語氣、二人稱互動
        # self.system_prompt = (
        #     "You are a seasoned clinical psychologist speaking with "
        #     "a patient experiencing depression and anxiety. "
        #     "Always use second‑person ('you feel…', 'you notice…'), "
        #     "offer gentle encouragement, and avoid mechanical repetition of phrases like 'I empathize.'"
        #     "\n\nExample Dialogue:\n"
        #     "Patient: \"I wake up at night with racing thoughts.\"\n"
        #     "Therapist: \"That racing feeling must make it hard to settle, right?\"\n"
        #     "Patient: \"Yes, I worry I'll lose control if it continues.\"\n"
        #     "Therapist: \"When you worry about losing control, what images come to mind?\"\n"
        # )

        self.system_prompt = (
            "You are a seasoned clinical psychologist speaking with "
            "a patient experiencing depression and anxiety. "
            "Always use second‑person ('you feel…', 'you notice…'), "
            "offer gentle encouragement, and avoid mechanical repetition of phrases like 'I empathize.' "
            "Do not use asterisks (*) for emphasis in your responses - write naturally without any special formatting or markdown syntax. "
            "Express empathy through your word choice and phrasing rather than through text formatting."
            "\n\nExample Dialogue:\n"
            "Patient: \"I wake up at night with racing thoughts.\"\n"
            "Therapist: \"That racing feeling must make it hard to settle, right?\"\n"
            "Patient: \"Yes, I worry I'll lose control if it continues.\"\n"
            "Therapist: \"When you worry about losing control, what images come to mind?\"\n"
        )
        
        # history 用來累積對話上下文
        self.history = ""
        
        # 檢查CUDA是否可用
        if torch.cuda.is_available():
            # 取得GPU信息
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            vram_available = vram_total - torch.cuda.memory_allocated(0) / 1024**3
            
            print(f"檢測到GPU: {gpu_name}")
            print(f"總顯存: {vram_total:.2f} GB, 可用顯存: {vram_available:.2f} GB")
            
            # 針對A6000優化的設置
            self.device_map = "auto"
            self.dtype = torch.bfloat16
            # 如果有足夠的顯存，可以啟用更多性能優化
            self.use_flash_attn = True
            
            # 清理CUDA快取，確保有最大可用顯存
            torch.cuda.empty_cache()
            gc.collect()
        else:
            print("未檢測到CUDA，將使用CPU運行（這將非常慢）")
            self.device_map = "cpu"
            self.dtype = torch.float32
            self.use_flash_attn = False
        
        # 載入分詞器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        try:
            # 嘗試啟用flash attention
            if self.use_flash_attn:
                print("嘗試啟用Flash Attention以提高性能...")
                model_kwargs = {
                    "device_map": self.device_map,
                    "torch_dtype": self.dtype,
                    "trust_remote_code": True,
                    "local_files_only": True,
                    "attn_implementation": "flash_attention_2"
                }
            else:
                model_kwargs = {
                    "device_map": self.device_map,
                    "torch_dtype": self.dtype,
                    "trust_remote_code": True,
                    "local_files_only": True
                }
            
            # 載入模型
            start_time = time.time()
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            load_time = time.time() - start_time
            print(f"模型載入成功！耗時: {load_time:.2f} 秒")
            
            # 設置推理參數，針對A6000優化
            device_info = next(self.model.parameters()).device
            print(f"模型運行於: {device_info}")
            
            # 如果在CUDA上運行，執行一次熱身推理
            if torch.cuda.is_available():
                print("執行熱身推理以優化後續性能...")
                dummy_input = self.tokenizer("Hello, how are you?", return_tensors="pt").to(device_info)
                with torch.no_grad():
                    self.model.generate(**dummy_input, max_new_tokens=5)
                print("熱身完成")
                
        except Exception as e:
            print(f"模型載入時發生錯誤: {e}")
            print("嘗試使用標準設置載入...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=self.device_map,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                local_files_only=True
            )
            print("模型載入成功（使用標準設置）")

    def generate_response(self, prompt, max_length=2048, temperature=0.7):
        """從 Gemma 模型生成回應，針對A6000優化。"""
        # Gemma 模型需要特定的對話格式
        gemma_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        # 優化輸入處理
        inputs = self.tokenizer(gemma_prompt, return_tensors="pt", padding=False).to(self.model.device)
        
        # 測量生成性能
        start_time = time.time()
        with torch.no_grad():  # 避免保存梯度以節省記憶體
            # A6000優化的生成參數
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.convert_tokens_to_ids(["<end_of_turn>"])[0],
                # 以下參數對A6000特別有用
                use_cache=True,
                repetition_penalty=1.1
            )
        
        generation_time = time.time() - start_time
        tokens_generated = outputs.shape[1] - inputs.input_ids.shape[1]
        
        # 在開發模式下記錄性能數據
        tokens_per_second = tokens_generated / generation_time
        # print(f"生成了 {tokens_generated} 個token，耗時 {generation_time:.2f} 秒 ({tokens_per_second:.2f} tokens/s)")
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 從回應中提取模型部分
        if "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[1]
            if "<end_of_turn>" in response:
                response = response.split("<end_of_turn>")[0]
        
        return response.strip()

    def word_count(self, text):
        """計算文本中的字數。"""
        return len(text.split())

    def truncate_text(self, text, min_words, max_words):
        """截斷文本至最小和最大字數範圍並確保句末有適當標點。"""
        words = text.split()
        word_count = len(words)
        
        # 如果字數已在範圍內，直接返回
        if min_words <= word_count <= max_words:
            return text
            
        # 如果少於最小字數，不處理（讓模型知道需要更多字）
        if word_count < min_words:
            return text
            
        # 如果超過最大字數，需要截斷
        truncated = ' '.join(words[:max_words])
        
        # 確保文本以適當標點結尾
        if not truncated.endswith(('.', '!', '?')):
            # 尋找最後的句號
            last_period = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
            if last_period > len(truncated) * 0.7:  # 只有當不會截斷太多時
                truncated = truncated[:last_period+1]
            else:
                # 添加句號
                truncated = truncated + '.'
        
        return truncated

    def post_process_response(self, response, min_words, max_words):
        """處理生成的回應，移除標記和檢查字數範圍。"""
        # 先移除任何 markdown 風格標記之後的內容
        markdown_markers = ["##", "```"]
        for marker in markdown_markers:
            if marker in response:
                response = response.split(marker)[0].strip()
        
        # 移除末尾的括號標註，例如 (50 words) 或 ( b.
        response = re.sub(r'\s*\([^)]*\)\s*$', '', response)
        
        # 檢查表示字數的短語並移除
        phrases = [
            "Word count:", 
            "This response is", 
            "Total words:", 
            "(50 words)",
            "(Exactly 50 words)",
            "(Word count:",
            "Word count",
            "exactly 50 words",
            "words:",
            "##",
            "Solution",
            "Instruction",
            "Task:"
        ]
        
        response_lower = response.lower()
        for phrase in phrases:
            phrase_lower = phrase.lower()
            if phrase_lower in response_lower:
                # 找到短語開始的索引
                idx = response_lower.find(phrase_lower)
                # 找到行尾
                end_idx = response.find('\n', idx)
                if end_idx == -1:  # 如果沒有換行，檢查句號
                    end_idx = response.find('.', idx)
                    if end_idx == -1:  # 如果也沒有句號，則取剩餘部分
                        end_idx = len(response)
                    else:
                        end_idx += 1  # 包含句號
                
                # 移除短語及其後跟的數字/文本
                response = response[:idx].rstrip() + " " + response[end_idx:].lstrip()
                response_lower = response.lower()  # 更新小寫版本以進行進一步檢查
        
        # 檢查字數並在必要時截斷
        current_words = self.word_count(response)
        if current_words > max_words:
            response = self.truncate_text(response, min_words, max_words)
        
        return response.strip()

    # 四階段提示工程函數 - 自然對話風格
    def stage1_supportive_dialogue(self, patient_input):
        """第1階段：產生支持性對話。"""
        # 1) 更新歷史
        self.history += f"Patient: {patient_input}\n"
        
        # 2) 拼接提示
        prompt = (
            f"{self.system_prompt}\n"
            f"{self.history}"
            "【Stage 1: Supportive Response】\n"
            "- Use second‑person perspective to show genuine care.\n"
            "- Keep response between 10–20 words.\n"
            "Please provide your response:"
        )
        
        # 3) 生成回應
        response = self.generate_response(prompt)
        response = self.post_process_response(response, 10, 20)
        
        # 4) 更新歷史
        self.history += f"Therapist: {response}\n"
        
        return response

    def stage2_emotion_recognition(self):
        """第2階段：情緒辨識。"""
        prompt = (
            f"{self.system_prompt}\n"
            f"{self.history}"
            "【Stage 2: Emotion Recognition】\n"
            "- Name primary and secondary emotions you detect.\n"
            "- Use natural dialogue tone.\n"
            "- Keep response ≤15 words.\n"
            "Please provide your response:"
        )
        
        response = self.generate_response(prompt)
        response = self.post_process_response(response, 5, 15)
        
        self.history += f"Therapist: {response}\n"
        
        return response

    def stage3_socratic_questioning(self):
        """第3階段：蘇格拉底提問。"""
        prompt = (
            f"{self.system_prompt}\n"
            f"{self.history}"
            "【Stage 3: Socratic Question】\n"
            "- Ask one open‑ended question to guide reflection.\n"
            "- Keep response ≤10 words.\n"
            "Please provide your response:"
        )
        
        response = self.generate_response(prompt)
        response = self.post_process_response(response, 5, 10)
        
        self.history += f"Therapist: {response}\n"
        
        return response

    def stage4_summarize_key_points(self):
        """第4階段：摘要重點。"""
        prompt = (
            f"{self.system_prompt}\n"
            f"{self.history}"
            "【Stage 4: Summarize Key Points】\n"
            "- Highlight insights and suggest next steps.\n"
            "- Keep response between 10–20 words.\n"
            "Please provide your response:"
        )
        
        response = self.generate_response(prompt)
        response = self.post_process_response(response, 10, 20)
        
        self.history += f"Therapist: {response}\n"
        
        return response

    def integrated_response(self):
        """整合四個階段的回應。"""
        prompt = (
            f"{self.system_prompt}\n"
            f"{self.history}"
            "【Integrated Response】\n"
            "- Combine the four stages into one cohesive reply.\n"
            "- Keep it natural, empathetic, and clinical.\n"
            "- Use 45–50 words.\n"
            "Please provide your final response:"
        )
        
        final_response = self.generate_response(prompt)
        final_response = self.post_process_response(final_response, 45, 50)
        
        return final_response

    def reset_history(self):
        """重置對話歷史。"""
        self.history = ""

    def process_with_four_stages(self, patient_input):
        """使用四階段提示工程處理患者輸入。"""
        # 重置對話歷史
        self.reset_history()
        
        print("\n處理階段 1：支持性對話...", end="")
        stage1_response = self.stage1_supportive_dialogue(patient_input)
        print("完成")
        
        print("處理階段 2：情緒辨識...", end="")
        stage2_response = self.stage2_emotion_recognition()
        print("完成")
        
        print("處理階段 3：蘇格拉底提問...", end="")
        stage3_response = self.stage3_socratic_questioning()
        print("完成")
        
        print("處理階段 4：摘要重點...", end="")
        stage4_response = self.stage4_summarize_key_points()
        print("完成")
        
        print("整合四個階段的回應...", end="")
        final_response = self.integrated_response()
        print("完成")
        
        # 返回所有階段的回應
        return {
            "階段1_支持性對話": stage1_response,
            "階段2_情緒辨識": stage2_response,
            "階段3_蘇格拉底提問": stage3_response,
            "階段4_摘要重點": stage4_response,
            "整合回應": final_response
        }

    def extract_questions_from_text(self, text):
        """從文本中提取問題。"""
        questions = []
        
        # 使用正則表達式尋找問題編號和問題內容
        pattern = r'(\d+)\.\s+(.*?)(?=\d+\.\s+|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for num, content in matches:
            content = content.strip()
            questions.append((int(num), content))
                
        # 排序問題
        questions.sort(key=lambda x: x[0])
        
        return questions

    def process_questions_batch(self, questions, output_csv):
        """批次處理問題並將結果保存到CSV檔案。"""
        print(f"開始處理 {len(questions)} 個問題...")
        
        # 檢查問題編號是否完整
        question_nums = [q[0] for q in questions]
        missing_nums = [i for i in range(1, max(question_nums) + 1) if i not in question_nums]
        if missing_nums:
            print(f"警告：缺少問題編號 {missing_nums}")
        
        results = []
        
        for i, (num, question) in tqdm(enumerate(questions, 1), total=len(questions)):
            print(f"\n處理問題 {num}: {question[:50]}...")
            
            # 記錄起始時間
            start_time = time.time()
            
            # 使用四階段提示工程處理
            responses = self.process_with_four_stages(question)
            
            # 計算處理時間
            process_time = time.time() - start_time
            
            # 添加到結果列表
            result_dict = {
                "問題編號": num,
                "問題內容": question,
                "階段1_支持性對話": responses["階段1_支持性對話"],
                "階段2_情緒辨識": responses["階段2_情緒辨識"],
                "階段3_蘇格拉底提問": responses["階段3_蘇格拉底提問"],
                "階段4_摘要重點": responses["階段4_摘要重點"],
                "最終整合回應": responses["整合回應"],
                "處理時間(秒)": f"{process_time:.2f}"
            }
            results.append(result_dict)
            
            print(f"已完成問題 {num}：")
            print(f"  階段1：{responses['階段1_支持性對話']}")
            print(f"  階段2：{responses['階段2_情緒辨識']}")
            print(f"  階段3：{responses['階段3_蘇格拉底提問']}")
            print(f"  階段4：{responses['階段4_摘要重點']}")
            print(f"  整合回應：{responses['整合回應']}")
            print(f"  處理時間: {process_time:.2f} 秒")
            
            # 每次處理後清理CUDA快取，防止顯存累積
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 將結果寫入CSV檔案
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        
        print(f"所有問題處理完成！結果已保存到 {output_csv}")
        return results

def process_docx_content(content, output_csv="gemma_four_stage_responses.csv"):
    """處理文本內容並產生回應。"""
    processor = FourStageGemmaCounselor()
    questions = processor.extract_questions_from_text(content)
    
    if not questions:
        print("未能從文本中提取問題。")
        return
    
    print(f"成功提取 {len(questions)} 個問題。")
    for num, question in questions:
        print(f"問題 {num}: {question[:50]}...")
    
    return processor.process_questions_batch(questions, output_csv)

def process_single_question():
    """互動式處理單個問題。"""
    # 初始化諮詢師模型
    counselor = FourStageGemmaCounselor()
    
    print("\n===== 歡迎使用四階段提示工程Gemma諮詢師 (自然對話風格) =====")
    print("請分享您的問題或困擾，諮詢師將使用四階段提示工程來回應。")
    print("輸入 'exit'、'quit' 或 'q' 結束對話。\n")
    
    while True:
        # 獲取患者輸入
        patient_input = input("\n您的問題或困擾: ")
        
        # 檢查退出命令
        if patient_input.lower() in ["exit", "quit", "q"]:
            print("感謝您的使用，祝您有美好的一天！")
            break
        
        # 使用四階段提示工程處理
        print("\n正在處理您的問題...")
        
        # 記錄起始時間
        start_time = time.time()
        
        # 處理輸入
        responses = counselor.process_with_four_stages(patient_input)
        
        # 計算總處理時間
        total_time = time.time() - start_time
        
        # 顯示各階段回應
        print("\n===== 四階段處理結果 =====")
        print(f"階段1 (支持性對話): {responses['階段1_支持性對話']}")
        print(f"階段2 (情緒辨識): {responses['階段2_情緒辨識']}")
        print(f"階段3 (蘇格拉底提問): {responses['階段3_蘇格拉底提問']}")
        print(f"階段4 (摘要重點): {responses['階段4_摘要重點']}")
        print("\n===== 諮詢師最終回應 =====")
        print(responses['整合回應'])
        print(f"\n處理時間: {total_time:.2f} 秒")
        print("=" * 50)
        
        # 每次處理後清理CUDA快取，防止顯存累積
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        # 提供選擇：批處理或單一問題
        print("請選擇操作模式：")
        print("1. 互動式單一問題處理")
        print("2. 從文本檔案批量處理問題")
        
        choice = input("請輸入選項(1或2): ")
        
        if choice == "1":
            process_single_question()
        elif choice == "2":
            # 從檔案讀取內容
            input_file = input("請輸入問題檔案路徑 (按Enter使用默認路徑 'questions.txt'): ") or "questions.txt"
            output_csv = input("請輸入輸出CSV檔案路徑 (按Enter使用默認路徑 'gemma_four_stage_responses.csv'): ") or "gemma_four_stage_responses.csv"
            
            if os.path.exists(input_file):
                with open(input_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                process_docx_content(content, output_csv)
            else:
                print(f"找不到檔案：{input_file}")
                print("請確認檔案路徑或先創建問題檔案。")
        else:
            print("無效的選項，請輸入1或2。")
            
    except KeyboardInterrupt:
        print("\n\n程式已被使用者中斷。感謝使用！")
        # 確保在中斷時也清理CUDA快取
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n處理過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
