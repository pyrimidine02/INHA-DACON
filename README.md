# 2024 INHA DACON

---
[**2024 인하 인공지능 챌린지**](https://dacon.io/competitions/official/236291/overview/description)
---
---
## **1. Structure of the file**

```
├── CSV_to_jsonl.ipynb               # Convert CSV to JSONL
├── Lora.ipynb                       # Fine-tuning the model(Not Used!)
├── Lora_v2.ipynb                    # Fine-tuning the model(Recent)
├── Modelfile                        # ollama custom model setting(Not Used!)
├── README.md
├── baseline.ipynb                   # Base line(Not Used!)
├── data
│   ├── INHA-DACON.jsonl
│   ├── dataset_info.json
│   ├── sample_submission.csv
│   ├── state.json
│   ├── test.csv
│   └── train.csv
├── main.ipynb                      # Make submission.csv
├── requirements.txt
└── result
    └── submission_0705.csv         # Best score

```
---
## **2. Install**

###  **a. Libraries**


The standard of the python version is 3.10
```
$ pip3 install -r requirements.txt
```
###  **b. Models and Lora**
Use Huggingface 🤗 Transformers to download Models and Lora
```
sonhy02/INHA-DACON-2024-8B-4BIT     #Quantizationed Model
sonhy02/INHA-DACON-2024-Lora        #Lora
```
---

## **3. LLM Model**

###  **a. About the Models**
- [~~llama3 8B~~](https://ollama.com/library/llama3:8b)  (Not Used!)
- [~~EEVE-Korean-Instruct-10.8B-v1.0-GGUF~~](https://huggingface.co/heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF/tree/main) (Not Used!)
- [~~beomi/Llama-3-Open-Ko-8B~~](https://huggingface.co/beomi/Llama-3-Open-Ko-8B)  (Not Used!)
- [MLP-KTLim/llama-3-Korean-Bllossom-8B](https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B)
- [sonhy02/INHA-DACON-2024-8B-4BIT](https://huggingface.co/sonhy02/INHA-DACON-2024-8B-4BIT)

###  **b. How to use**
Change the `model_id` to your fine-tuned model's path.
```python
# load the model.
model_id = "./models/20240703" # <-- CHANGE HERE TO YOUR MODEL PATH
model = AutoModelForCausalLM.from_pretrained(model_id,
                                            torch_dtype="auto", load_in_4bit=True)
```
---

## **3. History**

### 2024.07.05.
**Lora**
```python
lora_config = LoraConfig(
     task_type=TaskType.CAUSAL_LM,
     r=1,
     target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
     lora_alpha = 2,
     lora_dropout=0.05,
     #modules_to_save=['embed_tokens','lm_head']
)

training_args = TrainingArguments(
    # torch_compile = True,
    output_dir = './results',
    num_train_epochs = 1,
    fp16=True,
    per_device_train_batch_size=1,
    #per_device_eval_batch_size=1,
    gradient_accumulation_steps=5,
    save_strategy='epoch',
    #evaluation_strategy='epoch',
    save_total_limit=1,
    optim='adamw_bnb_8bit',
    #load_best_model_at_end=True,
    save_only_model=True,
    logging_strategy='steps',
    logging_steps=100,
    label_names=['labels'],
    report_to='tensorboard',
)
```
**Prompt**
```python
question_prompt = f"너는 주어진 Context를 토대로 Question에 답하는 챗봇이야. \
                                Question에 대한 답변만 가급적 한 단어로 최대한 간결하게 답변하도록 해. \
                                Context: {context} Question: {question}\n Answer:"
```


**Result**
```
Answer for question: 강원도 전 지역에서 91%의 수출을 점한 지역이 어디야 : 원주 홍천 동해 춘천 강릉
Processed count: 1
Answer for question: 전항일 사장이 이베이재팬 대표로 취임한 연도 : 2018년
Processed count: 2
Answer for question: 샤르르의 편곡에 참가한 사람이 누구야 : doko 도코
Processed count: 3
Answer for question: 예비사회적기업 일자리창출사업 공모에 신청한 기업의 수 : 52개 기업
Processed count: 4
Answer for question: 어디서 우수 청년 창업자를 발굴해 다양한 지원을 해주는 사업을 시행해 : 청년창업사관학교
Processed count: 5
Answer for question: 어디에서 국내 우주개발 정책 수립의 걸림돌이 되는 추격형 연구를 탈피한 새로운 연구를 하는 거야 : 한국항공우주연구원
Processed count: 6
Answer for question: 한국새농민충북도회가 취약계층의 사람들을 위해 충북사회복지공동모금회에 전달한 건 뭐야 : 농산물 꾸러미 200상자
Processed count: 7
Answer for question: 어떤 사람들에게 금감원에서 맞춤형 지원을 한다는 거야 : 청년 자영업자에 대해선 지원을 특화
Processed count: 8
Answer for question: 충북지방중소기업벤처기업청에서 백년가게를 뽑아 돕는 이유는 뭐지 : 오랜 기간 노하우를 바탕으로 지속가
Processed count: 9
Answer for question: 충청지방통계청에 의하면 충북지역에서 일 년 동안 높아진 소비자물가지수는 얼마나 돼 : 260
Processed count: 10
```
**F1 Score**
```
{'f1': 62.530103995621246}
```
**Contest Score**
```
0.77511
```








