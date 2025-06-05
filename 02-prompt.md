# Prompt

## 프롬프트 유형

### 1\) Question Prompts

- 정보 탐색이나 사실 기반 응답을 유도할 때 효과적임
- 간결한 질문을 통해 명확한 응답을 받을 수 있음

```python
from langchain.prompts import PromptTemplate

question_prompt = PromptTemplate.from_template(
    "다음 주제에 대해 무엇을 알고 있나요?: {topic}"
)
```

## 2\) Instruction Prompts

- 명확한 작업 지시나 단계별 처리 요청에 적합함
- 번역, 요약, 감성 분석 등 구조화된 태스크에서 유용함

```python
from langchain.prompts import PromptTemplate

question_prompt = PromptTemplate.from_template(
    "다음 텍스트를 한국어로 번역하세요:\n\n[텍스트]\n{text}"
)
```

```python
from langchain.prompts import PromptTemplate

step_prompt = PromptTemplate.from_template(
    """다음 텍스트에 대해서 작업을 순서대로 수행하세요:

[텍스트]
{text}

[작업 순서]
1. 텍스트를 1문장으로 요약
2. 핵심 키워드 3개 추출
3. 감정 분석 수행(긍정/부정/중립)

[작업 결과]"""
)
```

## 3\) Conversational Prompts

- 대화형 프롬프트는 사용자와의 자연스러운 상호작용을 통해 문맥을 지속적으로 유지할 수 있음
- 역할 기반 메시지 구조(system, human, AI)를 활용하여 대화 흐름에 맞는 응답 생성이 가능함

```python
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 고객 서비스 담당자입니다."),
    ("human", "{customer_message}"),
])
```

## 4\) Few-shot Prompts

- 예시 기반 학습을 통해 원하는 출력 형식을 유도할 수 있음
- 모델의 응답 품질과 일관성을 크게 높이는 방식임

```python
from langchain.prompts import PromptTemplate

few_shot_prompt = PromptTemplate.from_template(
    """다음은 텍스트를 요약하는 예시입니다:

원문: {example_input}
요약: {example_output}

이제 다음 텍스트를 같은 방식으로 50자 이내로 요약해주세요:
원문: {input_text}
요약:"""
)
```

## 5\) Conditional Prompts

- 입력의 유형에 따라 응답 로직을 다르게 구성할 수 있음
- 프롬프트 내 조건 분기를 통해 다양한 요청을 유연하게 처리 가능함

```python
from langchain.prompts import PromptTemplate

conditional_prompt = PromptTemplate.from_template(
    """입력 텍스트: {text}

주어진 텍스트가 질문인 경우: 명확한 답변을 제공
주어진 텍스트가 진술문인 경우: 진술문의 사실 여부를 검증
주어진 텍스트가 요청사항인 경우: 수행 방법을 단계별로 설명

응답은 다음 형식을 따라주세요:
유형: [질문/진술문/요청사항]
내용: [상세 응답]"""
)
```