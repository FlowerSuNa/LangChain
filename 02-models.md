# LangChain - Models

## 1. LLM Overview

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
- `temperature` : 출력 다양성 조절 (0~2)
    - 0 : 토큰 확률 분포의 편차가 적음 (정밀도 높음) ➡️ 항상 가장 확률이 높은 토큰 선택 (결정적/일관된 응답)
    - 2 : 토큰 확률 분포의 편차가 커짐 ➡️ 매우 창의적이나 불안정한 응답 가성능 높음
- `top_p` : 상위 확률 토큰 선택 (0~1)
    - 0.9 : 상위 90% 토큰의 확률 분포를 계산하여 sampling 수행 ➡️ 하위 10% 토큰은 버리고 sampling 수행
    - 1 : 모든 토큰의 확률 분포를 계산하여 sampling 수행
- `frequency_penalty` : 이전에 사용된 토큰에 패널티 부여 (-2~2) ➡️ 반복 감소, 다양성 증가
    - 양수 : 단어 반복 감소
    - 음수 : 단어 반복 허용
- `presence penalty` : 새 단어 사용 장려 (-2~2)
    - 양수 : 새로운 주제 도입 장려
    - 음수 : 기존 주제 유지 선호
- `stream` : 응답을 스트리밍 방식으로 받을지 여부
    - False : 완성된 응답 반환
    - True : 토큰 단위 실시간 스트리밍

### 4\) LLM 활용 팁

- 모델 양자화 : 모델을 16비트 float에서 4비트로 경량화함 ➡️ 성능이 떨어짐
- LLM은 많이 실험해 보는 것이 중요함
- GPT로 눈높이가 높아져 있음
- 간단한 LLM 작업(분류 문제 등)은 오픈 소스를 써도 웬만하면 잘 작동됨
- MCP, Agent 기능 사용하면 좋음

---

## 2. OpenAI 사용법

- `openai` 패키지를 사용하여 API를 호출할 수 있음

    ```python
    from openai import OpenAI

    client = OpenAI(
        api_key = OPENAI_API_KEY # 환경 변수로 등록되어 있는 경우 입력하지 않아도 됨
    )

    # Completion 요청 (prompt -> completion)
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            # developer 역할 - 전반적인 동작 방식 정의
            {"role": "developer", "content": "You are a assistant."},
            # user 역할 - 실제 요청 내용
            {"role": "user", "content": "..."},
        ],
        temperature=0.7,
        max_tokens=1000,
    )
    ```

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

