# LangChain - 프롬프트 (Prompts) 개요
- 프롬프트는 LLM의 응답 품질을 좌우하는 핵심 요소임
- LangChain에서는 목적에 맞는 적절한 프롬프트를 설계함으로써, 원하는 정보를 효과적으로 추출하거나 특정 작업을 정확하게 수행할 수 있도록 도와줌
- 단순한 질의부터 복잡한 지시, 예시 기반 학습까지, 프롬프트 유형에 따라 LLM의 응답 전략도 달라짐

---

## 1. 프롬프트 유형
- 프롬프트는 목적과 상황에 따라 다양한 방식으로 설계될 수 있음

**Question Prompts**
- 간결한 질문 형태로 구성하여, 특정 정보에 대해 구체적인 응답을 유도함
- 정보 추출, 지식 확인, 대화의 방향 설정 등에 효과적임
- 예: "다음 주제에 대해 무엇을 알고 있나요?: {topic}"

**Instruction Prompts**
- 명확한 작업 지시문을 통해 특정 작업을 수행하도록 유도함
- 번역, 요약, 감성 분석 등 정형화된 작업에 특히 적합함
- 예: "다음 문장을 한국어로 번역해 주세요: {sentence}"

**Conversational Prompts**
- 사용자와의 자연스러운 상호작용을 통해 문맥을 지속적으로 유지할 수 있음
- 역할 기반 메시지 구조(system, human, AI)를 활용하여, 대화 흐름에 맞는 자연스러운 응답을 유도할 수 있음
- 주로 멀티턴 대화, 챗봇 응답, 시뮬레이션 기반 상호작용에 적합함

**Few-shot Prompts**
- 샘플 입력-출력 쌍을 제시함으로써, 사용자가 원하는 응답 형식을 LLM이 따르도록 유도함
- 복잡하거나 구조화된 응답이 필요한 경우, 응답 품질과 일관성을 높이는 데 효과적임

**CoT (Chain Of Thought)**
- LLM이 복잡한 문제를 해결할 때 중간 사고 과정을 명시적으로 기술하도록 유도함
- LLM이 추론의 전 과정을 단계적으로 서술하게 하여, 문제 해결의 투명성과 정확성을 높일 수 있음

---

## 2. 프롬프트 설계 원칙

**명확성 (Clarity)**
- 프롬프트는 불필요한 수식어나 중의적 표현을 배제하고, 핵심 요구사항에 집중하여 작성해야 함
- LLM이 혼동 없이 작업을 수행할 수 있도록, 의도한 결과물에 대해 구체적이고 정확한 지시를 제공해야 함
- 예측 가능한 결과를 얻기 위해선, 문맥이 간결하면서도 목적이 분명한 문장으로 구성되어야 함

**맥락성 (Context)**
- 배경 정보, 목적, 대상 환경 등을 명확히 포함한 문맥 정보는 LLM이 사용자의 의도를 이해하고, 보다 정밀하게 목적에 부합하는 응답을 생성하는 데 도움을 줌
- 멀티턴 대화, 문서 요약, 코드 생성 등 복잡한 작업일수록 적절한 맥락 제공은 출력 품질과 정확도를 크게 향상 시킴

**구조화 (Structure)**
- 프롬프트는 일관된 입력-출력 형식을 유도할 수 있도록 구조화되어야 함
- 명확한 포맷 예시, 출력 템플릿, JSON 형식 등의 구조화를 적용하면 LLM의 응답 안정성을 높일 수 있음
- 구조화된 출력은 후처리 자동화, 오류 감지, 시스템 간 연동 등에 유리하며, 프롬프트의 재사용성과 유지보수 효율성도 함께 향상됨

---

## 3. 기본 프롬프트 구현

[🔗API 문서](https://python.langchain.com/api_reference/core/prompts.html)

### 1\) PromptTemplate

- 입력 변수를 포함한 프롬프트 양식을 정의할 수 있는 LangChain의 기본 템플릿 클래스임
- 변수에 값을 채워 넣어 다양한 상황에 맞는 프롬프트를 동적으로 생성할 수 있음
- 반복되는 프롬프트 구조를 재사용 가능하게 설계할 때 유용함

**템플릿 직접 생성**
- `PromptTemplate` 객체를 직접 생성하면 입력 변수 명을 명시적으로 지정할 수 있음
- `format` 메서드는 템플릿에 정의된 입력 변수에 값을 채워 최종 프롬프트 문자열을 생성함

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

**템플릿 간편 생성**
- `from_template` 메서드를 사용하면 템플릿 문자열 내 변수들을 자동 추출하여 간단히 생성 가능함
- 변수명을 명시하지 않아도 되므로 코드가 간결해짐

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

**템플릿 부분 포맷팅**
- `partial` 메서드를 사용하면 템플릿의 일부 입력값을 미리 고정해둘 수 있음
- 유사한 질의를 반복적으로 생성할 때 효율적임

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

---

## 4. 프롬프트 엔지니어링 (Prompt Engineering) 개념

- 프롬프트 엔지니어링은 LLM에게 명확하고 목적에 부합하는 지시를 제공함으로써, 원하는 결과를 안정적으로 얻어내는 기술임
- 단순 질의에서 고도화된 작업 지시까지, 프롬프트의 설계 방식에 따라 출력 품질이 크게 달라짐
- 잘 설계된 프롬프트 템플릿은 재사용 가능하며, 다양한 입력에 대해 일관된 응답 품질과 정확도를 확보할 수 있음

```
⚠️ 프롬프트 엔지니어링 기법(Few-shot, CoT 등)에 대한 상세 내용은 별도 섹션에서 다룰 예정
```