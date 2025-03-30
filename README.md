# **蘇邑洋論文進度**

# **1/16 ~ 1/13 進度**

1.預計完成
* 建立基於蘇格拉底式對話法的訓練環境。
* 使用 Phi3 作為 Data Generation Pipeline 來生成訓練數據。
* 採用 Phi4 作為訓練過程中的評估 LLM（Judge LLM）。
* 使用 DPO（Direct Preference Optimization）訓練方法，並記錄完整的訓練時間。
* 訓練完成的模型將用於後續的 Fine-Tune 階段。

2. 已經完成
* 成功搭建基於蘇格拉底對話法的訓練環境。
* 完成 Data Generation Pipeline，並透過 Phi3 生成訓練數據。
* 以 Phi4 作為 Judge LLM 來評估訓練效果。
* 運用 DPO 方法訓練模型，並詳細記錄訓練時間。

3. 目前問題
* 先前的研究主要使用 GPT-4o 作為 Judge LLM，因此原始程式碼針對 GPT 系列進行了優化，未出現明顯 Bug。然而，在改用 Ollama 模型時，產生了多種相容性問題與錯誤。
* 由 Data Generation Pipeline 生成的數據用於 Fine-Tune 可能也會導致潛在錯誤。

4.本週嘗試解決方法
* 由於 Ollama 模型與原始 GPT-4o 版本存在架構差異，每次進行 Data Generation Pipeline 訓練時，都需要手動修改對應的程式碼。
* 由於訓練時間較長，且評估分數影響最終模型表現，因此 Debug 過程無法直接以少量測試來確認問題，導致時間成本增加。
* 嘗試修正 Ollama 相關 Bug，以確保能夠正常訓練 Data Generation Pipeline 生成的模型，並進行後續 Fine-Tune。



# **1/13 ~ 1/20 進度**

1.預計完成
* 生成訓練數據並訓練出測試模型，以驗證 Fine-Tune 過程的可行性。
* 評估測試模型的效能與準確性。
* 若測試結果符合預期，則移除測試指令，進行正式訓練。


2. 已經完成
* 成功生成測試模型，並完成初步 Fine-Tune 過程。
* 驗證與評估測試模型的效能。


3. 目前問題
* 使用 Ollama LLM 需要進行特定參數的調整，因不同版本的 LLM 可能在訓練過程中出現不相容問題。
* 目前遇到錯誤：'list' object has no attribute 'chat'，可能與 Judge LLM 的程式碼設計相關。


4.本週嘗試解決方法
* 修改 Judge LLM 的 action 設計，確保返回的實例能夠被正確處理。
* 目前已經能夠大致運行，但仍需進一步檢查潛在 Bug。


# **1/20 ~ 1/27 進度**

1.預計完成
* 移除test訓練完成正式的生成和fine tine訓練，並評估和驗證
* 完成之後對所現有的dataset進行二階段的生成和微調訓練



2. 已經完成
* 移除test訓練完成正式的生成和fine tine訓練，並評估和驗證
* 完成之後對所現有的dataset進行二階段的生成和微調訓練



3. 目前問題
* 過程中還是有一些bug，但大致完成。



4.本週嘗試解決方法
* 嘗試修改使用JudgeLLM action並保持其返回的實例。大致ok，並繼續檢查有啥bug。

# **1/27 ~ 2/12 進度**

Github：https://github.com/HustleEverydayy/2llmchat

1.預計完成
* (1)設計兩個llm的互動 A這個llm專門面對患者 b這個LLM專門在指導這個A的LLM怎麼回答(假設B的LLM判斷這段句子覺得ok 要辨識情感和者蘇格拉底的回答，然後A這個LLM接到這個動作以後組這兩件事情的辨識清單和蘇格拉底的詢問 (B的LLM就是接收A的LLM句子給情緒回饋和用蘇格拉底對話式的方法) 像是給A的LLM(我已經做很多事情了 但我的老闆扔乃不滿意 我實在不知道該怎麼辦)然後丟給B的LLM (B的LLM只要做兩件事情
* (2)增加重點簡述語句 同理同情心
* (3)評估循序回答和單一整合的差別
* (4)新增了5個患者的問題(input)
* (5)第一個特徵(辨識情感) 得到第一個輸出 第一個原本輸出+上第二個(同情同理心的回應) 得到 第二個輸出+加上第三個(蘇格拉底式的回答)得到第三個輸出+上短述語句得到第四個輸出(而第四個輸出就是循續的最後一個完整答案就是前面累積相加再判斷我說的這些的) 然後再和一次性的輸出做比較 問phi4同樣的使用者輸入 這兩句哪 一個比較有同理心對患者是友好的
2. 已經完成
* (1)設計兩個llm的互動 A這個llm專門面對患者 b這個LLM專門在指導這個A的LLM怎麼回答(假設B的LLM判斷這段句子覺得ok 要辨識情感和者蘇格拉底的回答，然後A這個LLM接到這個動作以後組這兩件事情的辨識清單和蘇格拉底的詢問 (B的LLM就是接收A的LLM句子給情緒回饋和用蘇格拉底對話式的方法) 像是給A的LLM(我已經做很多事情了 但我的老闆扔乃不滿意 我實在不知道該怎麼辦)然後丟給B的LLM (B的LLM只要做兩件事情
* (2)增加重點簡述語句 同理同情心
* (3)評估循序回答和單一整合的差別
* (4)新增了5個患者的問題(input)
* (5)第一個特徵(辨識情感) 得到第一個輸出 第一個原本輸出+上第二個(同情同理心的回應) 得到 第二個輸出+加上第三個(蘇格拉底式的回答)得到第三個輸出+上短述語句得到第四個輸出(而第四個輸出就是循續的最後一個完整答案就是前面累積相加再判斷我說的這些的) 然後再和一次性的輸出做比較 問phi4同樣的使用者輸入 這兩句哪 一個比較有同理心對患者是友好的
3. 目前問題
* (1)在轉換過程中會遇到翻譯問題
* (2)處理回應時有時候會太冗長或者遇到許多llm再回應時會遇到衝突和bug
4.本週嘗試解決方法
* (1)讓llm在回應時都統一用繁體中文
* (2)使用更準確得提示讓llm正常去回應，並且避免英文所導致的衝突。


起初架構：
![image](https://hackmd.io/_uploads/SJ7SLoVFJl.png)

(2)起初的回應範例：
![image](https://hackmd.io/_uploads/rJrD8jEtyl.png)

(3):增加了同理同情心
![image](https://hackmd.io/_uploads/SyB_Lj4t1x.png)


新增的5個患者問題：
1. 工作壓力與自我懷疑：
```
最近工作壓力真的很大，每天都在加班，已經連續一個月都睡不好了。同事都升遷了，只有我還在原地踏步。我開始懷疑自己是不是真的不夠好，可能真的比不上別人吧...有時候坐在辦公室，看著電腦螢幕就會突然想哭。
```
![image](https://hackmd.io/_uploads/rkT2C1_Yyl.png)
![image](https://hackmd.io/_uploads/HJ_AR1_Kyl.png)



2. 人際關係的疏離：
```
我發現自己越來越不會跟人相處了。每次朋友約出去，我總是找藉口推掉。就連家人打電話來關心，我也常常敷衍了事。我知道這樣不對，但就是提不起勁，感覺跟這個世界有一層隔閡。
```
![image](https://hackmd.io/_uploads/SJLeyxuFye.png)



3. 感情困擾與失落：
```
交往三年的感情就這樣結束了。他說我們不合適，要去國外發展。我以為我們的感情很穩定，還規劃著未來...現在每天起床第一件事就是看手機，希望他能傳訊息來，但我知道不可能了。
```
![image](https://hackmd.io/_uploads/HkobJe_YJg.png)



4. 生涯迷惘：
```
研究所快畢業了，但我完全不知道接下來要做什麼。爸媽希望我找個穩定的工作，但我對自己的興趣和專長都沒有把握。看著同學們都規劃得很好，我卻像是迷失在十字路口，不知道該往哪個方向走。
```
![image](https://hackmd.io/_uploads/HkJUJx_tJg.png)



5. 家庭角色衝突：
```
自從生了小孩之後，我感覺自己快要崩潰了。白天要上班，晚上要照顧孩子，婆婆還常說我把小孩照顧得不好。老公總說他工作很累，需要休息...但我呢？我也很累啊！有時候真的很想把一切都放下逃走。
```

![image](https://hackmd.io/_uploads/rk8XygOKkg.png)




---


# **2/12 ~ 2/19 進度**

Github：https://github.com/HustleEverydayy/2llmchat
1. 預計完成
* (1)設計兩個llm的互動 A這個llm專門面對患者 b這個LLM專門在指導這個A的LLM怎麼回答(假設B的LLM判斷這段句子覺得ok 要辨識情感和者蘇格拉底的回答，然後A這個LLM接到這個動作以後組這兩件事情的辨識清單和蘇格拉底的詢問 (B的LLM就是接收A的LLM句子給情緒回饋和用蘇格拉底對話式的方法) 像是給A的LLM(我已經做很多事情了 但我的老闆扔乃不滿意 我實在不知道該怎麼辦)然後丟給B的LLM (B的LLM只要做兩件事情
* (2)增加重點簡述語句 同理同情心
* (3)評估循序回答和單一整合的差別
* (4)新增了5個患者的問題(input)
* (5)第一個特徵(辨識情感) 得到第一個輸出 第一個原本輸出+上第二個(同情同理心的回應) 得到 第二個輸出+加上第三個(蘇格拉底式的回答)得到第三個輸出+上短述語句得到第四個輸出(而第四個輸出就是循續的最後一個完整答案就是前面累積相加再判斷我說的這些的) 然後再和一次性的輸出做比較 問phi4同樣的使用者輸入 這兩句哪 一個比較有同理心對患者是友好的
* (6)對B的llm (phi4)進行微調訓練，基於現有的5個案例資料，讓模型能更好地理解和執行循序漸進的諮商過程
* (7)將原本的提示工程設計應用在微調後的模型上，確保保留原有的專業諮商架構

2. 已經完成
* (1)設計兩個llm的互動 A這個llm專門面對患者 b這個LLM專門在指導這個A的LLM怎麼回答
* (2)增加重點簡述語句 同理同情心
* (3)評估循序回答和單一整合的差別
* (4)新增了5個患者的問題(input)
* (5)實現循序漸進的回應邏輯和累積式輸出
* (6)完成 LoRA 微調的程式代碼架構
* (7)設計好微調後模型的提示工程架構

3. 目前問題
* (1)在轉換過程中會遇到翻譯問題
* (2)處理回應時有時候會太冗長或者遇到許多llm再回應時會遇到衝突和bug 
* (3)微調過程中需要考慮 A6000 的記憶體限制
* (4)需要確保微調後的模型仍能保持原有的提示工程效果

4. 本週嘗試解決方法
* (1)讓llm在回應時都統一用繁體中文
* (2)使用更準確得提示讓llm正常去回應，並且避免英文所導致的衝突
* (3)優化 LoRA 訓練參數以適應 A6000：
   - 使用較小的批次大小（batch_size=2）
   - 增加梯度累積步數（gradient_accumulation_steps=8）
   - 啟用梯度檢查點
* (4)將原有的提示工程架構整合到微調流程中，確保模型訓練符合原有設計理念

使用lora fine tune 完成截圖

![image](https://hackmd.io/_uploads/rysU2yQ91e.png)

![image](https://hackmd.io/_uploads/Hk1dnkm9Jl.png)


# **2/19 ~ 2/26 進度**


1. 預計完成
* (1)找到一個專業並且非常成熟的Dataset來Fine tune：CounselChat Dataset 
* (2)把兩個llm的chatbot，在B的LLM使用finetune過後的LLM拿來再做之前的提示工程
* (3)再來拿一個成熟的llm去評估兩個回答

2. 已經完成
* (1)找到一個專業並且非常成熟的Dataset來Fine tune
* (2)把兩個llm的chatbot，在B的LLM使用finetune過後的LLM拿來再做之前的提示工程

3. 目前問題
* (1)因為Finetune一次大型的Dataset需要大量時間，因此我先嘗試縮減並當作test train.

4. 本週嘗試解決方法
* (1)使用簡短的dataset先進尋finetune完成之後，處理原本的提示工程


# **2/26 ~ 3/5 進度**

1. 預計完成
* (1)把原本拿來test縮減的Dataset的方式引進專業的CounselChat Dataset使可以FineTune
* (2)使用另一個專業的Dataset查看Finetune效果跟 CounselChat Dataset哪個優
* (3)並且再加入原本提示工程的方法在這個llm，並且回答的內容拿另一個成熟的llm來評估分數回答內容
* (4)增加更多的患者回應查看另一個LLM評估分數的聚集

2. 已經完成
* (1)把原本拿來test縮減的Dataset的方式引進原本正常的Dataset使可以FineTune

3. 目前問題
* (1)FineTune之後的LLM在嘗試使用提示工程會導致些bug，並且評估分數也有很大的差距問題

4. 本週嘗試解決方法
* (1)因為FineTune後的時間成本太過於長久，因此先使用測試縮短的Dataset能夠完美的加上提示工程，並且讓另一個成熟的LLM能夠評估分數
* (2)解決提示工程在FineTune過後的LLM使可以正常回復


# **3/1 ~ 3/9 進度**

預計完成
(1)製作一個新的方向的流程圖
(2)重新使用乾淨的llama3 8b去做我找到的dataset做lora finetune
(3)微調完成之後使用乾淨的llama3 8b去找gpt做評估分數
已經完成
(1)製作一個新的方向的流程圖
(2)重新使用乾淨的llama3 8b去做我找到的dataset做lora finetune
目前問題
(1)因為之前的方法雜亂重新訓練一個新的，並且開始根據上禮拜說的流程開始重做(比較有效率) 
(2)嘗試寫出一個完整工作的腳本code
本週嘗試解決方法
(1)重新寫了一個流程的腳本，並可以拆分每個步驟的code

流程圖：
![image](https://hackmd.io/_uploads/B13OKC9oye.png)


# **3/9 ~ 3/16 進度**


預計完成
(1)更新流程圖邏輯
(2)製作一個有同理心的llama3，使用提示工程
(3)乾淨的和有同理心的給chatgpt評估分數(1-10)分
已經完成
(1)更新流程圖邏輯
(2)
目前問題
(1)大問題
本週嘗試解決方法
(1)等補充

流程圖：
![image](https://hackmd.io/_uploads/S1fgVRM2kx.png)

回應圖：
I hear how overwhelmed you're feeling at work with the constant overtime and lack of sleep. It's completely understandable that you'd feel frustrated and upset seeing your colleagues being promoted while you remain in the same position. Those moments of feeling like crying at your desk show just how deeply this is affecting you. The self-doubt you're experiencing is a natural response to this situation, but it doesn't reflect your true capabilities. Have you been able to identify what specific aspects of your current situation are most difficult for you? And have you considered talking to someone at work who might be able to provide some perspective on the promotion process?
![螢幕擷取畫面 2025-03-16 202546](https://hackmd.io/_uploads/Byo5QBVnJe.jpg)


# **3/16 ~ 3/23 進度**


預計完成
(1)llm同理心提示工程優化回答
已經完成
(1)無
目前問題
(1)回應的內容非常不理想，持續優化
本週嘗試解決方法
(1)嘗試使用最簡單的提示先慢慢試出回應，因為提示工程在phi和llama3會有些小問題，導致沒辦法有理想的回答，因為不是拿成熟的llm所以也不能每次的回應都非常理想，持續測試中。


# **3/23 ~ 3/30 進度**

預計完成
(1)新方法：先使用A經過提示工程的llm產生的內容經過B的LLM檢查答案，如果適合在反回去給A LLM回答，像是反思的概念。
(2)回答限制50字

已經完成
(1)無

目前問題
(1)嘗試經過這設計去設計一個新的提示工程，需要設計一個框架讓A模型生成內容後由B模型檢查品質，不符合標準則反饋給A進行優化。

本週嘗試解決方法
(1)研究反思機制設計，實現兩個LLM間的協作流程，確保A模型能根據B模型的反饋進行有效調整，同時保證最終回答控制在50字內。