# LangChain - 모델 (Models) 개요

- 모델은 LLM 파이프라인을 중심 구성요소로, 입력된 프롬프트를 기반으로 텍스트를 생성하는 역할을 수행함
- LangChain에서는 **OpenAI, HuggingFace, Ollama** 등 다양한 LLM 제공자와 연동할 수 있음
- LLM의 추론 품질은 선택한 모델 종류와 설정된 하이퍼파라미터에 따라 달라짐

---

## 1. LLM 개요

- LLM (Large Language Model)은 대규모 텍스트 데이터를 학습하여, 자연어를 이해하고 생성할 수 있는 인공지능 모델임
- 문장 생성뿐 아니라, 요약, 번역, 질의응답, 추론 등 다양한 언어 기반 작업에 범용적으로 활용 가능함

**언어 모델 (Language Model, LM)**

- 언어 모델은 **주어진 문맥(Context)** 을 기반으로 다음 단어의 조건부 확률 분포를 계산함
- 이 확률 분포에서 단어를 선택하고, 이를 반복하여 문장을 생성하는 구조임
> **📓 단어 선택 방식**
> - 결정적 방법 (Deterministic) : 확률이 가장 높은 단어를 선택하는 방법으로, 항상 같은 결과가 나옴 *(Greedy Selection)*
> - 확률적 방법 (Probabilistic) : 단어 분포에서 확률에 따라 무작위로 단어를 샘플링하여, 각 실행마다 다른 결과가 나옴 *(Random Sampling)*

**LLM 작동 방식**

- LLM은 일반적으로 **사용자 입력 (Prompt) → 프롬프트 처리 → LLM 추론 → 응답 생성 (Completion)** 단계를 거쳐 작동함
- LLM은 대부분 Gemma, Llama, Qwen, Exaone과 같은 **대규모 사전 학습 모델 (Foundation Model)** 을 사용함
- 이전 단어들을 기반으로 다음 단어를 하나씩 순차적으로 예측하는 **Auto-Regressive 방식**임
- 각 단어는 이전까지의 모든 단어에 의존함
- 매 단계에서 가능한 모든 단어의 확률 분포를 계산하고, 이 분포에 따라 **하나의 단어를 확률적으로 선택**함

---

## 2. 주요 하이퍼파라미터

[🔗HuggingFace Blog](https://huggingface.co/blog/how-to-generate)

**max_tokens** : 생성될 최대 토큰 수
- 모델이 한 번에 생성할 수 있는 응답의 최대 토큰 길이를 제한하는 파라미터임

**temperature** : 토큰 다양성 제어 (0~2)
- 생성할 단어의 확률 분포를 얼마나 평탄하게 만들지 조절하는 파라미터임
- 값이 낮을수록 분포가 뾰족해져 확률이 높은 토큰에 집중하게 되어 예측 가능한 일관된 응답이 생성됨
- 값이 높을수록 분포가 평탄해져 확률이 낮은 토큰이 선택될 가능성이 높아짐 <br>- *예측 불가능한 창의적인 응답이 생성될 가능성 증가*
- `temperature = 0` : 가장 확률이 높은 단어만을 선택하며, 항상 동일한 응답을 출력함 <br>- *정확성과 일관성이 중요할 때 적합*
- `temperature = 1` : 확률 분포를 그대로 반영한 무작위 선택임
- `temperature = 2` : 확률 분포가 매우 평탄해져 거의 균등에 가까운 단어 선택이 이루어지며, 창의적이지만 응답의 일관성과 정확성은 크게 떨어질 수 있음 <br>- *실험적이거나 자유로운 텍스트 생성에 활용*

**top_p** : 누적 확률 기반 토큰 선택 (0~1)
- 누적 확률이 일정 기준을 넘는 상위 토큰 집합에서만 샘플링을 수행함
- 확률 분포의 꼬리 영역(낮은 확률 토큰)을 제거하여, 현실적이고 유의미한 선택을 유도함
- 파라미터 값이 낮을수록 의미 없는 단어나 비정상적인 결과를 줄이는 데 유리함
- `top_p = 0.9` : 전체 토큰 중 누적 확률이 90%에 도달할 때까지의 상위 토큰들만 고려하여 샘플링함 <br>- *나머지 10% 확률을 가진 토큰은 샘플링 제외*
- `top_p = 1` : 모든 토큰의 확률 분포를 그대로 반영하여 샘플링함

**frequency_penalty** : 단어 반복 억제 조절 (-2~2)
- 생성 중 이미 등장한 단어가 또 나올 경우 패널티를 부여해 반복을 줄이는 역할을 함
- 등장 횟수가 많을수록 점점 더 큰 패널티를 부여하며, 파라미터 값이 클수록 반복을 강하게 제한함
- 양수이면 더 다양한 단어가 생성되며, 음수이면 같은 표현이 자주 등장함
- `frequency_penalty > 0` : 같은 단어가 자주 나올수록 패널티를 부여하여 반복 단어가 감소함
- `frequency_penalty < 0` : 반복된 단어에 보상을 부여하여 단어 반복을 허용함

**presence_penalty** : 새 단어 등장 유도 (-2~2)
- 이전에 한 번이라도 등장한 단어에 대해 패널티를 부여하여 새로운 단어와 주제를 더 많이 사용하도록 유도하는 값임
- 파라미터 값이 클수록 더 다양한 내용이 생성됨
- 양수이면 더 다양한 화제를 유도하며, 음수이면 응답의 일관성이 증가함
- `presence_penalty > 0` : 등장한 적 있는 단어에 점수 패널티를 부여하여 새로운 단어와 주제가 등장하도록 장려함
- `presence_penalty < 0` : 이미 등장한 단어에 보상을 부여하여 기존 주제에 머무르려는 경향이 강화됨

**stream** : 응답 스트리밍 여부
- 모델이 응답을 한 번에 모두 반환할지, 아니면 토큰 단위로 실시간 전송할지 결정하는 옵션임
- `stream = False` : 완성된 응답을 한 번에 반환함
- `stream = True` : 생성된 토큰을 실시간으로 전송함 <br>- *대화형 UI나 채팅봇에서 더 자연스럽고 빠르게 느껴짐*

> **📓 temperature와 top_p 조합**
> - temperature와 top_p는 텍스트 생성의 다양성(변동성)과 일관성(안정성) 사이의 균형을 조절하는 핵심 하이퍼파라미터임
> - 서로 보완적으로 작동하며, 설정값에 따라 출력 스타일이 크게 달라짐
> - 출력 스타일은 모델마다 민감하게 달라질 수 있으므로, 목적에 맞게 여러 조합을 실험해보는 것이 가장 효과적임
> - 특히 창의성과 정확성 사이의 균형은 비즈니스 목적, 사용자 기대 수준에 따라 조정해야 함
>
> **✒️ 예시**
> - 안정성과 일관성이 중요한 경우 (요약, 설명, 문서 생성 등)<br>- temperature : 0 ~ 0.5<br>- top_p : 0.5 ~ 0.8 
> - 객관적 사실이 중요한 경우 (예: 정보 응답, 문서 분석 등)<br>- temperature : 0.7<br>- top_p : 0.5 ~ 0.8
> - 창의성이 중요한 경우 (예: 글쓰기, 브레인스토밍, 아이디어 생성 등)<br>- temperature : 1.0 ~ 1.5<br>- top_p : 0.9 ~ 1.0

---

## 3. LLM 제공자 연동

### 1\) OpenAI 모델 사용

[🔗Guide 문서](https://python.langchain.com/docs/integrations/chat/openai/) / [🔗API 문서](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html) / [🔗OpenAI Models](https://platform.openai.com/docs/pricing)

- OpenAI에서 제공하는 LLM을 LangChain에서 사용하려면 `langchain-openai` 패키지 설치가 필요함
- `ChatOpenAI` 클래스를 통해 모델 인스턴스를 생성하고, 이를 통해 LLM 응답을 처리할 수 있음
- OpenAI 계정에서 발급받은 **API Key**를 `OPENAI_API_KEY` 환경변수에 설정해야 하며, 외부에 노출되지 않도록 주의해야 함

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4.1-mini",
    # model="gpt-4.1-mini-2025-04-14", # 정확한 버전 표기
    temperature=0.4,
    top_p=0.7
)
```

**유의할 점**
- OpenAI 모델을 사용할 때 **정확한 학습 날짜가 포함된 버전**으로 선언하는 것이 좋음
    - 단순히 모델명만 지정하면, OpenAI의 자동 업데이트로 인해 모델이 변경될 수 있음
    - 동일한 모델명이라도 버전에 따라 응답 성향, 성능, 처리 방식이 달라질 수 있음
    - 따라서 실험 재현, 응답 일관성을 위해 명시적 버전 지정이 권장됨
- OpenAI의 API는 **모델의 종류와 사용량에 따라 비용이 발생**함
    - 고성능 모델은 응답 품질이 뛰어나지만, 단가가 높고 응답 속도도 느릴 수 있음
    - 따라서 사용 목적에 따라 모델 선택과 예산 계획을 사전에 조정하는 것이 필요함
    - 모델별로 지원하는 최대 컨텍스트 토큰 수도 다르므로 확인이 필요함
- API를 통해 전송하는 입/출력에는 **민감한 정보(개인 정보, 비밀번호 등)** 를 포함하지 않는 것이 좋음
    - OpenAI는 전송된 데이터를 학습에는 사용하지 않지만, 보안상 외부 유출 가능성을 최소화해야 함
    - 기업 환경에서는 내부 정책에 따라 OpenAI API 사용을 제한하거나 프록시 환경으로 구성하기도 함

### 2\) HuggingFace 모델 사용

[🔗Guide 문서](https://python.langchain.com/docs/integrations/chat/huggingface/)

- HuggingFace의 오픈소스 모델을 LangChain에서 사용하려면 `langchain_huggingface` 패키지 설치가 필요함 
- 모델 로드는 `HuggingFaceEndpoint` 또는 `HuggingFacePipeline` 클래스를 통해 수행하며, 이를 `ChatHuggingFace` 클래스에 전달해 LangChain에서 사용할 수 있음

**HuggingFaceEndpoint 사용** [🔗API 문서](https://python.langchain.com/api_reference/huggingface/llms/langchain_huggingface.llms.huggingface_endpoint.HuggingFaceEndpoint.html)
- `HuggingFaceEndpoint` 클래스는 `huggingface_hub` 패키지를 기반으로 작동되며, Hugging Face Hub의 API 엔드포인트를 통해 모델을 호출함
- 서버 없이 Hugging Face **Inference API**를 바로 사용하는 구조로, 빠르게 테스트 가능함 (단, API 호출 비용이 발생함)
- Hugging Face 계정에서 발급받은 **Access Token**을 `HUGGINGFACEHUB_API_TOKEN` 환경변수에 설정해야 함

```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)
chat_model = ChatHuggingFace(llm=llm)
```

**HuggingFacePipeline 사용** [🔗API 문서](https://python.langchain.com/api_reference/huggingface/llms/langchain_huggingface.llms.huggingface_pipeline.HuggingFacePipeline.html)
- `HuggingFacePipeline` 클래스는 `transformers` 패키지를 기반으로 작동되며, 직접 로컬 서버에서 모델을 로드해 사용함
- 자체 GPU 환경에서 모델 실행 가능하며, 비용 부담 없이 고정된 환경에서 테스트 가능함

```python
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)
chat_model = ChatHuggingFace(llm=llm)
```

### 3\) Ollama 모델 사용

[🔗Guide 문서](https://python.langchain.com/docs/integrations/chat/ollama/) / [🔗Ollama Github](https://github.com/ollama/ollama) / [🔗Ollama Models](https://ollama.com/search)

- Ollama는 로컬 실행에 최적화된 경량 LLM 플랫폼으로, GPU 없이도 쉽게 모델을 실행할 수 있음
- LangChain에서 사용하려면 `langchain_ollama` 패키지 설치가 필요함

**Ollama 모델 관리 기본 명령어**
```bash
ollama list                     # 설치된 모델 목록 확인
ollama pull gemma3:1b           # 모델 다운로드
ollama run gemma3:1b            # 모델 실행 (없으면 자동 다운로드)
```

**LangChain에서 Ollama 모델 사용** [🔗API 문서](https://python.langchain.com/api_reference/ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html)
- `ChatOllama` 클래스를 통해 LangChain에 통합 가능함
- 내부적으로 Ollama의 REST API (http://localhost:11434)를 호출함

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gemma3:1b",
    temperature=0
)
```

---

## 4. 오픈소스 LLM 사용 시 고려사항

**모델 구조 확인**
- 단순히 모델 이름만 보고 선택하면 안되며, `config.json` 또는 `model_card`는 필수로 확인해야 함
    - 파라미터 수 (예: 7B, 13B, 65B 등)
    - context length (지원 최대 토큰 수, 예: 2k, 4k, 16k)
    - hidden size / num attention heads 등 구조 정보
- 파라미터 수가 많다고 항상 고성능은 아님
    - Ollama나 llama.cpp 기반의 gguf 포맷 모델은 대부분 4bit 또는 8bit로 양자화(quantization) 되어 있음
    - 양자화 버전은 추론 속도 및 메모리 효율이 향상되지만, 정밀도 저하와 복잡한 논리 추론 성능 저하가 발생할 수 있음
    - 예: 동일한 LLaMA2-13B 모델이라도, full-precision(16/32bit) 버전과 4bit 양자화 버전은 응답의 일관성, 세밀한 추론 정확도 면에서 체감 성능 차이가 큼

**라이선스 확인**
- 일부 모델은 상업적 사용 금지임 (cc-by-nc, research only 등)
- 반드시 `LICENSE.txt` 또는 HuggingFace의 `model card` 하단을 확인해야 함

**토크나이저 호환성**
- 모델마다 토크나이저 구조가 달라, 다른 모델과 혼용 불가함
- 반드시 같은 모델의 토크나이저(`tokenizer_config.json`)를 사용해야 제대로 동작함

**프롬프트 형식 (prompt format)**
- 일부 모델은 채팅 특화(ChatML, instruction-tuned)되어 있어, 프롬프트 구성 방식이 다름
- LangChain에서 사용할 경우 `PromptTemplate` 구성에 주의가 필요함

**지원 task 여부 확인**
- text-generation만 지원하는지, chat, summarization, embedding, code generation 등 task별 최적화 여부를 확인해야 함

> **📓 LLM 실전 활용 팁**
> - 다양한 모델 x 파라미터 x 프롬프트 조합을 반복 실험하여 최적 구성을 찾아야 함
> - GPT 계열 모델의 보급으로 사용자 기대 수준이 매우 높아진 상황이며, 이에 따라 모델 선택과 응답 결과에 대한 설득력 있는 설명이 중요해짐
> - 간단한 작업(분류, 키워드 추출 등)은 경량 오픈 소스 LLM으로도 충분히 처리 가능함
> - 복잡한 응답 흐름에는 MCP, Agent 등의 고급 기능 활용하면 더 유연하게 구현할 수 있음