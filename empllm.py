import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import os
import json
import re
from datetime import datetime

# Load model and tokenizer
def load_model():
    print("Loading Phi-3.5-mini-instruct model...")
    
    # Please replace with your local model path
    model_path = "D:/Finelora/empathetic同理心提示工程/phi-3.5-mini-instruct"  # Replace with your actual local path
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Set pad token to eos token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set pad_token to eos_token")
        
        print("Loading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        print("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Create a prompt format for Phi-3.5-mini-instruct
def create_phi_prompt(system_message, user_message):
    """Format prompt with system message and user message for Phi model"""
    prompt = f"<|system|>\n{system_message.strip()}\n\n<|user|>\n{user_message.strip()}\n\n<|assistant|>\n"
    return prompt

# Clean response function with enhanced processing
def clean_response(text):
    """Clean the response to remove unwanted patterns"""
    # Remove any text after these markers
    cutoff_markers = [
        "Follow Up", "Elaborat", "Note:", "Response:", "Therapeutic", 
        "Example", "Let me", "Here is", "Remember", "---", "***", "###",
        "[End]", "(Stop)", "STOP", "<stop>", "End response", "END", 
        "Hope this helps", "I hope", "In summary", "To summarize"
    ]
    
    # Check for cutoff markers and truncate text
    for marker in cutoff_markers:
        if marker in text:
            parts = text.split(marker)
            text = parts[0].strip()
    
    # Clean any trailing incomplete sentences or phrases
    trailing_fragments = ["...", ".", ",", ":", ";", "-", "–", "—"]
    for fragment in trailing_fragments:
        if text.endswith(fragment) and text.count(".") > 1:
            # Find the last complete sentence
            sentences = re.split(r'(?<=[.!?])\s+', text)
            if len(sentences) > 1 and not sentences[-1].strip().endswith("."):
                text = " ".join(sentences[:-1]).strip()
                if not text.endswith("."):
                    text += "."
    
    # Remove redundant period at the end
    if text.endswith(".."):
        text = text[:-1]
    
    return text.strip()

# Generate response with the model
def generate_with_model(model, tokenizer, prompt, max_new_tokens=200, temperature=0.7):
    """Generate a response using the model with specified parameters"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    stop_token_ids = [tokenizer.eos_token_id]
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.92,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_token_ids
        )
    
    # Decode the generated response
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # Apply response cleaning
    response = clean_response(response)
    
    return response

# Generate empathetic response using the four-stage counseling approach
def generate_empathetic_response(model, tokenizer, patient_input):
    # Create a log file for recording responses
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"response_log_{timestamp}.txt"
    detailed_log = f"detailed_log_{timestamp}.txt"
    
    # Log the input
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(f"===== New Response ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) =====\n")
        f.write(f"Input: {patient_input}\n\n")
    
    with open(detailed_log, "a", encoding="utf-8") as f:
        f.write(f"===== Detailed Response Log ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) =====\n\n")
        f.write(f"User Input:\n{patient_input}\n\n")
    
    try:
        # Stage 1: Supportive Dialogue (A1)
        supportive_system = """
As a compassionate listener, your task is to acknowledge the person's experience and validate their feelings.

Create a brief supportive response that:
1. Reflects back what you heard them share
2. Shows understanding without judgment
3. Validates that their feelings make sense

Requirements:
- Keep it to 2-3 short, simple sentences
- Use warm, everyday language
- Focus only on empathetic understanding
- Do not give advice or solutions

Example: "It sounds like you're dealing with a lot of pressure right now. Working so hard without recognition while seeing others advance must feel incredibly frustrating. Your feelings of doubt are completely understandable in this situation."
"""
        
        supportive_prompt = create_phi_prompt(supportive_system, patient_input)
        
        # Log the prompt
        with open(detailed_log, "a", encoding="utf-8") as f:
            f.write(f"Stage 1 (Supportive Dialogue) Prompt:\n{supportive_prompt}\n\n")
        
        # Generate the response
        print("Generating supportive dialogue response...")
        supportive_response = generate_with_model(model, tokenizer, supportive_prompt, max_new_tokens=75, temperature=0.5)
        print(f"Stage 1 response generated: {supportive_response[:50]}...")
        
        # Log the response
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(f"Stage 1 (Supportive Dialogue): {supportive_response}\n\n")
        with open(detailed_log, "a", encoding="utf-8") as f:
            f.write(f"Stage 1 (Supportive Dialogue) Response:\n{supportive_response}\n\n")
        
        # Stage 2: Emotion Recognition (A2)
        emotion_system = """
Based on what the person shared, identify 1-2 core emotions they're likely experiencing.

Your task:
1. Name specific emotions (like frustration, sadness, self-doubt, anxiety)
2. Connect these emotions to their specific situation
3. Frame as a gentle question to confirm your understanding

Requirements:
- Be specific about the emotions
- Keep to 2 short sentences maximum
- End with a simple question

Example: "I'm hearing feelings of disappointment and self-doubt as you compare yourself to promoted colleagues. Does that capture some of what you're experiencing?"
"""
        
        emotion_user = f"""
Person's message:
{patient_input}

Your previous supportive response:
{supportive_response}

Based on their message, identify 1-2 specific emotions they might be feeling and ask a simple question to confirm.
"""
        
        emotion_prompt = create_phi_prompt(emotion_system, emotion_user)
        
        # Log the prompt
        with open(detailed_log, "a", encoding="utf-8") as f:
            f.write(f"Stage 2 (Emotion Recognition) Prompt:\n{emotion_prompt}\n\n")
        
        # Generate the response
        print("Generating emotion recognition response...")
        emotion_response = generate_with_model(model, tokenizer, emotion_prompt, max_new_tokens=60, temperature=0.5)
        print(f"Stage 2 response generated: {emotion_response[:50]}...")
        
        # Log the response
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(f"Stage 2 (Emotion Recognition): {emotion_response}\n\n")
        with open(detailed_log, "a", encoding="utf-8") as f:
            f.write(f"Stage 2 (Emotion Recognition) Response:\n{emotion_response}\n\n")
        
        # Stage 3: Socratic Questioning (A3)
        socratic_system = """
Ask ONE simple question that helps the person explore their situation more deeply.

Your question should:
- Be open-ended (not answerable with yes/no)
- Use simple language (under 15 words)
- Focus on self-exploration
- Not contain advice or suggestions

Good examples:
"What would success look like for you right now?"
"How do you measure your value at work?"
"What aspects of your work still bring you satisfaction?"

Bad examples:
"Don't you think you should talk to your manager?" (contains advice)
"In what ways might this situation reflect deeper patterns?" (too complex)
"Why not focus on your strengths instead?" (leading)

Write ONLY the question. No introduction or follow-up.
"""
        
        socratic_user = f"""
Person's message:
{patient_input}

Based on their situation, ask ONE simple, open-ended question to help them reflect more deeply. Just write the question.
"""
        
        socratic_prompt = create_phi_prompt(socratic_system, socratic_user)
        
        # Log the prompt
        with open(detailed_log, "a", encoding="utf-8") as f:
            f.write(f"Stage 3 (Socratic Questioning) Prompt:\n{socratic_prompt}\n\n")
        
        # Generate the response
        print("Generating Socratic questioning response...")
        socratic_response = generate_with_model(model, tokenizer, socratic_prompt, max_new_tokens=30, temperature=0.4)
        print(f"Stage 3 response generated: {socratic_response[:50]}...")
        
        # Log the response
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(f"Stage 3 (Socratic Questioning): {socratic_response}\n\n")
        with open(detailed_log, "a", encoding="utf-8") as f:
            f.write(f"Stage 3 (Socratic Questioning) Response:\n{socratic_response}\n\n")
        
        # Stage 4: Summarizing Key Points (A4)
        summary_system = """
Offer a brief perspective that helps reframe the person's situation in a constructive way.

Your response should include exactly:
1. One sentence summarizing their core challenge
2. One sentence validating that their feelings are natural
3. One or two sentences offering an alternative perspective

Use cognitive restructuring to help them see their situation from a different angle without dismissing their feelings.

Example: "You're dealing with exhaustion and self-doubt while seeing colleagues advance. These feelings are completely natural in your situation. Consider that growth sometimes happens invisibly at first, and this challenging period may be building strengths that aren't yet recognized but will be valuable in the future."
"""
        
        summary_user = f"""
Person's message:
{patient_input}

Briefly summarize their situation and offer an alternative perspective in 3-4 short sentences.
"""
        
        summary_prompt = create_phi_prompt(summary_system, summary_user)
        
        # Log the prompt
        with open(detailed_log, "a", encoding="utf-8") as f:
            f.write(f"Stage 4 (Summarizing Key Points) Prompt:\n{summary_prompt}\n\n")
        
        # Generate the response
        print("Generating summary response...")
        summary_response = generate_with_model(model, tokenizer, summary_prompt, max_new_tokens=75, temperature=0.5)
        print(f"Stage 4 response generated: {summary_response[:50]}...")
        
        # Log the response
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(f"Stage 4 (Summarizing Key Points): {summary_response}\n\n")
        with open(detailed_log, "a", encoding="utf-8") as f:
            f.write(f"Stage 4 (Summarizing Key Points) Response:\n{summary_response}\n\n")
        
        # Final integrated response
        final_system = """
Create one unified, supportive response that combines four elements:

1. Acknowledge their situation (based on your supportive dialogue)
2. Name the specific emotions they might be feeling (based on your emotion recognition)
3. Include your reflective question to help them explore further
4. Offer a brief alternative perspective (based on your summary)

Important requirements:
- 100-120 words total
- Use everyday language and short sentences
- Make it flow naturally from one idea to the next
- No headings, bullet points, or markers
- No explanations of what you're doing
- No introductions to each element (like "Here's a question for you")
- End naturally without trailing off

The response should read as ONE unified message from a supportive friend.
"""
        
        final_user = f"""
Original message:
{patient_input}

Integrate these elements into ONE unified supportive response (no markers or extra text):

Supportive acknowledgment: {supportive_response}

Emotion recognition: {emotion_response}

Reflective question: {socratic_response}

Alternative perspective: {summary_response}
"""
        
        final_prompt = create_phi_prompt(final_system, final_user)
        
        # Log the prompt
        with open(detailed_log, "a", encoding="utf-8") as f:
            f.write(f"Final Integrated Response Prompt:\n{final_prompt}\n\n")
        
        # Generate the final response with stricter controls
        print("Generating final integrated response...")
        final_response = generate_with_model(model, tokenizer, final_prompt, max_new_tokens=120, temperature=0.4)
        print(f"Final response generated: {final_response[:50]}...")
        
        # Additional safety check for final response
        final_response = clean_response(final_response)
        
        # Log the response
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(f"Final Response: {final_response}\n")
            f.write("====================================\n")
        with open(detailed_log, "a", encoding="utf-8") as f:
            f.write(f"Final Response:\n{final_response}\n\n")
            f.write("====================================\n")
        
        # Return all responses
        return {
            "stage1_supportive": supportive_response,
            "stage2_emotion": emotion_response,
            "stage3_socratic": socratic_response,
            "stage4_summary": summary_response,
            "final_response": final_response
        }
    except Exception as e:
        print(f"Error generating response: {e}")
        import traceback
        traceback.print_exc()
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(f"Error: {str(e)}\n")
            f.write("====================================\n")
        raise e

# Gradio Interface
def create_interface(model, tokenizer):
    def process_input(patient_input, show_intermediate):
        try:
            print(f"\nProcessing input: {patient_input[:50]}...")
            
            # Generate responses
            all_steps = generate_empathetic_response(model, tokenizer, patient_input)
            
            # Format response for display
            display_response = all_steps["final_response"]
            if not display_response.strip():
                display_response = "The generated response was empty. Please try again or use different input."
            
            # Print for debugging
            print(f"\nFinal response length: {len(display_response)}")
            
            # Return results
            if show_intermediate:
                result = [
                    display_response,
                    all_steps["stage1_supportive"],
                    all_steps["stage2_emotion"],
                    all_steps["stage3_socratic"],
                    all_steps["stage4_summary"],
                    ""  # No error
                ]
                print(f"Returning with intermediate steps: {len(result)} items")
                return result
            else:
                result = [
                    display_response,
                    "", "", "", "",  # Empty intermediate steps
                    ""  # No error
                ]
                print(f"Returning without intermediate steps: {len(result)} items")
                return result
                
        except Exception as e:
            error_message = f"Error occurred: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return ["", "", "", "", "", error_message]  # Return with error message
    
    with gr.Blocks(title="Empathetic Conversation Assistant") as demo:
        gr.Markdown("# 💬 同理心對話助理")
        gr.Markdown("此AI助理使用四階段提示方法提供具有同理心的回應。")
        
        with gr.Row():
            with gr.Column(scale=2):
                patient_input = gr.Textbox(
                    label="分享你想說的",
                    placeholder="例如：最近工作壓力很大...",
                    lines=5
                )
                
                with gr.Row():
                    show_steps = gr.Checkbox(label="顯示詳細步驟", value=True)
                    submit_btn = gr.Button("獲取回應", variant="primary")
                
            with gr.Column(scale=3):
                final_output = gr.Textbox(label="AI回應", lines=8)
                error_output = gr.Textbox(label="錯誤訊息", lines=2)
                
                with gr.Accordion("查看詳細步驟", open=True):
                    supportive_output = gr.Textbox(label="步驟1: 支持性對話", lines=3)
                    emotion_output = gr.Textbox(label="步驟2: 辨識情緒", lines=3)
                    reflective_output = gr.Textbox(label="步驟3: 蘇格拉底提問", lines=3)
                    summary_output = gr.Textbox(label="步驟4: 摘要重點", lines=3)
        
        submit_btn.click(
            process_input,
            inputs=[patient_input, show_steps],
            outputs=[final_output, supportive_output, emotion_output, reflective_output, summary_output, error_output]
        )
        
        gr.Markdown("""
        ## 工作原理
        
        此助理使用四階段方法提供具有同理心的回應：
        
        1. **支持性對話**：確認你的感受並表達理解
        2. **辨識情緒**：識別你可能正在經歷的情緒
        3. **蘇格拉底提問**：提供深思熟慮的問題，幫助你探索自身處境
        4. **摘要重點**：提供對你處境的有益視角
        
        這些元素被融合成一個連貫、支持性的回應，旨在幫助你感到被傾聽和理解。
        """)
        
    return demo

def main():
    try:
        # Load model
        model, tokenizer = load_model()
        
        # Create and launch interface
        demo = create_interface(model, tokenizer)
        demo.launch(share=True)
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        # Create a simple error interface
        with gr.Blocks() as error_demo:
            gr.Markdown("# ⚠️ 錯誤")
            gr.Markdown(f"""
            啟動過程中發生錯誤: {str(e)}
            
            可能的解決方案:
            1. 檢查模型路徑是否正確 (D:/Finelora/empathetic同理心提示工程/phi-3.5-mini-instruct)
            2. 確保已安裝所有必要的套件: `pip install torch transformers gradio`
            3. 檢查是否有足夠的 GPU 記憶體
            
            詳細錯誤資訊:
            ```
            {str(e)}
            ```
            """)
        error_demo.launch(share=True)

if __name__ == "__main__":
    main()
