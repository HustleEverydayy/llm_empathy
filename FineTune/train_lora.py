import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import pandas as pd
import logging
import gc
import os

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clear_gpu_memory():
    """清理 GPU 記憶體"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU 記憶體已清理")

def prepare_training_data(tokenizer, data_path: str):
    """準備訓練資料"""
    try:
        # 載入訓練數據
        with open(data_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        # 準備數據列表
        processed_data = []
        
        for example in training_data['training_data']:
            for step in ['step1', 'step2', 'step3', 'step4']:
                # 創建提示和回應
                prompt = f"案主：{example['input']}\n回應："
                full_text = prompt + example['output'][step]
                
                # 編碼文本
                encoded = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=256,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                item_dict = {
                    'input_ids': encoded['input_ids'][0].tolist(),
                    'attention_mask': encoded['attention_mask'][0].tolist(),
                    'labels': encoded['input_ids'][0].tolist()
                }
                processed_data.append(item_dict)
        
        dataset = Dataset.from_pandas(pd.DataFrame(processed_data))
        logger.info(f"已準備 {len(dataset)} 筆訓練資料")
        return dataset
        
    except Exception as e:
        logger.error(f"準備訓練資料時發生錯誤: {str(e)}")
        raise

def train_counseling_model(data_path: str, output_dir: str):
    """訓練心理諮商模型"""
    try:
        # 清理 GPU 記憶體
        clear_gpu_memory()
        
        # 載入分詞器
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-4",
            trust_remote_code=True
        )
        
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 準備訓練資料
        dataset = prepare_training_data(tokenizer, data_path)
        
        # 載入模型
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-4",
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False
        )
        
        # LoRA 配置
        lora_config = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=[
                "k_proj", "q_proj", "v_proj", "o_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False
        )
        
        # 準備模型
        model = get_peft_model(model, lora_config)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        
        # 訓練參數
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            warmup_steps=50,
            learning_rate=1e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            remove_unused_columns=False,
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            eval_strategy="no",
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            report_to="none"
        )
        
        # 定義數據整理器
        def data_collator(features):
            batch = {}
            for key in features[0].keys():
                values = [f[key] for f in features]
                tensor = torch.tensor(values)
                batch[key] = tensor
            return batch
        
        # 訓練器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )
        
        # 開始訓練
        trainer.train()
        
        # 儲存模型
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info("訓練完成，模型已儲存")
        return True
        
    except Exception as e:
        logger.error(f"訓練過程發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        train_counseling_model(
            data_path="counseling_dataset.json",
            output_dir="./phi4_counseling_model"
        )
    except Exception as e:
        logger.error(f"程序執行錯誤: {str(e)}")
