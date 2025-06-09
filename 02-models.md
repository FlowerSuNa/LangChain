# LangChain - 모델 (Model) 개요

## 1. LLM 개념

### 1\) 언어 모델 (Language Model, LM)

- 언어 모델은 **주어진 문맥(Context)** 을 기반으로 다음 단어의 조건부 확률 분포를 계산함
- 이 확률 분포에서 단어를 선택하고, 이를 반복하여 문장을 생성하는 구조임

```markdown
# 단어 선택 방식
- 결정적 방법 (Deterministic) : 확률이 가장 높은 단어를 선택하는 방법으로, 항상 같은 결과가 나옴 (Greedy Selection)
- 확률적 방법 (Probabilistic) : 단어 분포에서 확률에 따라 무작위로 단어를 샘플링하여, 각 실행마다 다른 결과가 나옴 (Random Sampling)
```

### 2\) LLM 작동 방식

- 입력 (Prompt) → 프롬프트 처리 → 모델 추론 (LLM) → 응답 생성 (Completion) 단계를 거침
- 대부분 **대규모 사전 학습 모델 (Foundation Model)** 을 사용함 - ex) Gemma, Lammma, Qwen, Exaone 등
- 이전 단어들을 기반으로 다음 단어를 하나씩 순차적으로 예측하는 방식임 (Auto-Regressive 방식)
- 각 단어는 이전까지의 모든 단어에 의존함
- 매 단계에서 가능한 모든 단어의 확률 분포를 계산하고, 이 분포에 따라 하나의 단어를 선택함 (확률적 예측)

### 3\) 주요 하이퍼파라미터

- `max_tokens` : 생성될 최대 토큰 수
- `temperature` : 토큰 다양성 제어 (0~2)
    - 생성할 단어의 확률 분포를 얼마나 평탄하게 만들 것인지를 결정하는 파라미터임
    - 낮을수록 분포가 뾰족해져 예측 가능하고 일관된 응답이 생성됨
    - 높을수록 분포가 평탄해져 창의적이지만 예측 불가능한 응답이 생성됨
    - `temperature=0` : 가장 확률이 높은 단어만을 선택하며, 항상 동일한 응답을 출력함 - 정확성과 일관성이 중요할 때 적합
    - `temperature=1` : 확률 분포를 그대로 반영한 무작위 선택임
    - `temperature=2` : 확률 분포가 매우 평탄해져 거의 균등에 가까운 단어 선택이 이루어지며, 창의적이지만 응답의 일관성과 정확성은 크게 떨어질 수 있음 - 실험적이거나 자유로운 텍스트 생성에 활용
- `top_p` : 누적 확률 기반 토큰 선택 (0~1)
    - 누적 확률이 일정 기준을 넘는 상위 토큰 집합에서만 샘플링을 수행함
    - 확률 분포의 꼬리 영역(낮은 확률 토큰)을 제거하여, 현실적이고 유의미한 선택을 유도함
    - 낮을수록 의미 없는 단어나 비정상적인 결과를 줄이는 데 유리함
    - `top_p=0.9` : 전체 토큰 중 누적 확률이 90%에 도달할 때까지의 상위 토큰들만 고려하여 샘플링함 - 나머지 10% 확률을 가진 토큰은 샘플링 제외
    - `top_p=1` : 모든 토큰의 확률 분포를 그대로 반영하여 샘플링함
- `frequency_penalty` : 단어 반복 억제 조절 (-2~2)
    - 생성 중 이미 등장한 단어가 또 나올 경우 패널티를 부여해 반복을 줄이는 역할을 함
    - 값이 클수록 반복을 강하게 제한함
    - `> 0`(양수) : 같은 단어가 자주 나올수록 패널티를 부여하여 반복 단어가 감소함 - 더 다양한 단어 생성
    - `< 0`(음수) : 반복된 단어에 보상을 부여하여 단어 반복을 허용함 - 같은 표현이 자주 등장함
- `presence penalty` : 새 단어 사용 장려 (-2~2)
    - 양수 : 새로운 주제 도입 장려
    - 음수 : 기존 주제 유지 선호
- `stream` : 응답을 스트리밍 방식으로 받을지 여부
    - False : 완성된 응답 반환
    - True : 토큰 단위 실시간 스트리밍

### 4\) LLM 활용 주의할 점

- 모델을 16bit float → 4bit로 양자화(Quantization) 하면 메모리 및 속도는 개선되지만, 정밀도는 다소 저하됨
- 다양한 모델, 파라미터, 프롬프트 조합을 반복 실험하여 최적 구성을 찾아야 함
- GPT 계열 모델의 등장으로 사용자의 기대 수준이 매우 높아진 상태임
- 간단한 작업(분류, 키워드 추출 등)은 경량 오픈 소스 LLM으로도 충분히 처리 가능함
- MCP, Agent 같은 고급 기능 활용하면 복잡한 작업도 더 유연하게 구현 가능함

---

## 2. OpenAI 사용법

- OpenAI에서 제공하는 LLM을 LangChain과 사용하려면 `langchain-openai` 라이브러리를 설치하여 사용할 수 있음
- `ChatOpenAI`는 인스턴스로 생성해 LLM 응답을 처리할 수 있음
    - [Document](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)
    - [Guide](https://python.langchain.com/docs/integrations/chat/openai/)
    ```python
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model="gpt-4.1-mini",
        # model="gpt-4.1-mini-2025-04-14", # 정확한 버전 표기
        temperature=0.4,
        top_p=0.7
    )
    ```
- 모델 버전은 학습 날짜를 명시해 정확히 선언하는 것이 좋음
    - [OpenAI Model Version](https://platform.openai.com/docs/pricing)
    - 단순히 모델명만 지정하면, OpenAI 측의 업데이트로 인해 모델 차이가 발생할 수 있음
    - 동일한 모델명이라도 버전에 따라 응답 성향이나 성능이 달라질 수 있음
- 모델 버전별 특징
    - gpt-4.1 : 높은 성능 및 비용
    - gpt-4.1-mini : 빠른 속도, 낮은 비용
    - o1 계열 : 복잡한 추론 가능

---

### 3. Ollama

---

### 4. HuggingFace