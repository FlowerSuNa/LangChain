# LangChain - 프롬프트 (Prompt) 개요
- 프롬프트는 LLM의 응답 품질을 좌우하는 핵심 요소임
- LangChain에서는 목적에 맞는 적절한 프롬프트를 설계함으로써, 원하는 정보를 효과적으로 추출하거나 특정 작업을 정확하게 수행할 수 있도록 도와줌
- 단순한 질의부터 복잡한 지시, 예시 기반 학습까지, 프롬프트 유형에 따라 LLM의 응답 전략도 달라짐

---

## 1. 프롬프트 유형
- 프롬프트는 목적과 상황에 따라 다양한 방식으로 설계될 수 있음

### 1\) Question Prompts
- 간결한 질문 형태로 구성하여, 특정 정보에 대해 구체적인 응답을 유도함
- 정보 추출, 지식 확인, 대화의 방향 설정 등에 효과적임
- 예: "다음 주제에 대해 무엇을 알고 있나요?: {topic}"

### 2\) Instruction Prompts
- 명확한 작업 지시문을 통해 특정 작업을 수행하도록 유도함
- 번역, 요약, 감성 분석 등 정형화된 작업에 특히 적합함
- 예: "다음 문장을 한국어로 번역해 주세요: {sentence}"

### 3\) Conversational Prompts
- 사용자와의 자연스러운 상호작용을 통해 문맥을 지속적으로 유지할 수 있음
- 역할 기반 메시지 구조(system, human, AI)를 활용하여, 대화 흐름에 맞는 자연스러운 응답을 유도할 수 있음
- 주로 멀티턴 대화, 챗봇 응답, 시뮬레이션 기반 상호작용에 적합함

### 4\) Few-shot Prompts
- 샘플 입력-출력 쌍을 제시함으로써, 사용자가 원하는 응답 형식을 LLM이 따르도록 유도함
- 복잡하거나 구조화된 응답이 필요한 경우, 응답 품질과 일관성을 높이는 데 효과적임

```
⚠️ 프롬프트 엔지니어링 기법(Few-shot, Chain-of-Thought 등)에 대한 상세 내용은 별도 섹션에서 다룰 예정
```

---

## 2. 기본 프롬프트 구현

- [LangChain 문서](https://python.langchain.com/api_reference/core/prompts.html)

### 1\) PromptTemplate

- 입력 변수를 포함한 프롬프트 양식을 정의할 수 있는 LangChain의 기본 템플릿 클래스임
- 변수에 값을 채워 넣어 다양한 상황에 맞는 프롬프트를 동적으로 생성할 수 있음
- 반복되는 프롬프트 구조를 재사용 가능하게 설계할 때 유용함

**템플릿 생성**

```python
from langchain_core.prompts import PromptTemplate

# 템플릿 생성
prompt = PromptTemplate(
    template="다음 주제에 대해 {length}자 이내로 설명해주세요: {topic}의 {content}",
    input_variables=["length", "topic", "content"]
)

# 최종 프롬프트
formatted_prompt = prompt.format(
    length="500자",
    topic="인공지능",
    content="정의, 역사, 응용분야"
)
```

**`from_template` 메서드 사용** : 템플릿 문자열에서 입력 변수를 자동으로 추출해 `PromptTemplate`을 간편하게 생성함

```python
from langchain_core.prompts import PromptTemplate

# 템플릿 생성
prompt = PromptTemplate.from_template(
    "다음 주제에 대해 {length}자 이내로 설명해주세요: {topic}의 {content}"
)

# 최종 프롬프트
formatted_prompt = prompt.format(
    length="500자",
    topic="인공지능",
    content="정의, 역사, 응용분야"
)
```

**템플릿 부분 포맷팅** : 템플릿의 일부만 먼저 채워두고, 나머지를 나중에 입력해 반복 작업이나 유사 프롬프트 생성에 유용함

```python
from langchain_core.prompts import PromptTemplate

# 템플릿 생성 
prompt = PromptTemplate(
    template="다음 주제에 대해 {length}자 이내로 설명해주세요: {topic}의 {content}",
    input_variables=["length", "topic", "content"]
)

# 템플릿 부분 포맷팅
partial_prompt = template.partial(topic="인공지능")

# 최종 프롬프트
final_prompt1 = partial_prompt.format(content="정의", length="100자")
final_prompt2 = partial_prompt.format(content="역사", length="300자")
final_prompt3 = partial_prompt.format(content="응용분야", length="100자")
```

### 2\) ChatPromptTemplate
- 대화형 LLM과의 상호작용을 위해 설계된 템플릿 클래스임
- system, user, assistant 등 역할 기반 메시지를 구조화하여 대화 흐름을 정의함
- 멀티턴 대화에서 문맥을 유지하며 자연스럽고 일관된 응답을 유도할 수 있음

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

# 메시지 템플릿 생성
system_message = SystemMessagePromptTemplate.from_template(
    "당신은 {role} 전문가입니다."
)
human_message = HumanMessagePromptTemplate.from_template(
    "{question}"
)

# 템플릿 생성
chat_prompt = ChatPromptTemplate.from_messages([
    system_message,
    human_message
])

# 최종 프롬프트
formatted_prompt = chat_prompt.format(
    role="인공지능",
    question="인공지능의 정의를 설명해주세요."
)
```


