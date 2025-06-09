# LangChain - 모델 (Model) 개요

## 1. LLM 개념

### 1\) 언어 모델 (Language Model, LM)

- 언어 모델은 주어진 문맥(Context)을 기반으로 다음 단어의 조건부 확률을 계산함
- 이 확률 분포에 따라 다음 단어를 선택하고, 이를 반복해 문장을 생성함

**단어 선택 방식**
- **결정적 방법 (Deterministic)**은 가장 확률이 높은 단어를 선택하는 방법으로, 항상 같은 결과가 나옴 (Greedy Selection)
- **확률적 방법 (Probabilistic)**은 단어의 확률 분포에 따라 단어를 랜덤으로 선택하여, 각 실행마다 다른 결과가 나옴 (Random Sampling)

### 2\) LLM 작동 방식

- 입력 (Prompt) → 프롬프트 분석 및 처리 → 모델 추론 (LLM) → 응답 생성 (Completion) 단계를 거침
- 대부분 이미 사전 학습된 Foundation Model을 사용함 (Gemma, Lammma, Qwen, Exaone 등)
- 이전 단어들을 기반으로 다음 단어를 예측하는 방식임 (Auto-Regressive)
- 각 단어는 이전까지의 모든 단어에 의존하며, 순차적으로 생성됨
- 가능한 모든 단어의 확률의 분포를 계산하여 이 분포를 바탕으로 하나의 단어를 선택함 (확률적 예측)

### 3\) 주요 하이퍼파라미터

- `max_tokens` : 생성될 최대 토큰 수
- `temperature` : 토큰의 다양성 조절 (0 ~ 2)
    - 토큰의 확률 분포를 얼마나 평탄하게 만들 것인지 결정함
    - 낮을수록 예측 가능하고 일관된 응답이, 높을수록 창의적인 응답이 생성됨
    - `0` : 토큰 확률 분포가 중심에 몰린 형태로, 항상 가장 확률이 높은 토큰을 선택하여 일관된 응답을 함
    - `2` : 토큰 확률 분포가 평탄해져 다양한 루보 중에서 무작위성이 높아진 선택이 이루어짐<br>↳ 창의적인 응답을 유도할 수 있지만, 일관성과 정확성이 떨어질 수 있음


- `top_p` : 확률 누적 기반 상위 토큰에서 샘플링 (0~1)
    - 0.9 : 상위 90% 토큰의 확률 분포를 계산하여 샘플링 수행 ➡️ 하위 10% 토큰은 샘플링 제외
    - 1 : 모든 토큰의 확률 분포를 계산하여 샘플링 수행

- `frequency_penalty` : 반복된 단어에 패널티 부여 (-2~2)
    - 양수: 단어 반복 감소
    - 음수 : 단어 반복 허용
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