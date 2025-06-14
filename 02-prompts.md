# LangChain - 프롬프트 (Prompts) 개요
- 프롬프트는 모델의 응답 품질을 좌우하는 핵심 요소임
- LangChain에서는 목적에 맞는 적절한 프롬프트를 설계함으로써, 원하는 정보를 효과적으로 추출하거나 특정 작업을 정확하게 수행할 수 있도록 도와줌
- 단순한 질의부터 복잡한 지시, 예시 기반 학습까지, 프롬프트 구성에 따라 LLM의 응답 전략도 달라짐

---

## 1. 프롬프트 구성 방식
- 프롬프트는 목적과 상황에 따라 다양한 방식으로 설계될 수 있음

### 1\) 요청 형식

**질문형 (Question)**
- 간결한 질문 형태로 구성하여, 특정 정보에 대해 구체적인 응답을 유도함
- 정보 추출, 지식 확인, 대화의 방향 설정 등에 효과적임
- 예: "다음 주제에 대해 무엇을 알고 있나요?: {topic}"

**지시형 (Instruction)**
- 명확한 작업 지시문을 통해 특정 작업을 수행하도록 유도함
- 번역, 요약, 감성 분석 등 정형화된 작업에 특히 적합함
- 예: "다음 문장을 한국어로 번역해 주세요: {sentence}"

**대화형 (Conversational)**
- 사용자와의 자연스러운 상호작용을 통해 문맥을 지속적으로 유지할 수 있음
- 역할 기반 메시지 구조(System, Human, AI)를 활용하여, 대화 흐름에 맞는 자연스러운 응답을 유도할 수 있음
- 주로 멀티턴 대화, 챗봇 응답, 시뮬레이션 기반 상호작용에 적합함

### 2\) 예시 활용 방식

**제로샷 (Zero-shot)**
- 예시 없이 직접 LLM에게 작업을 지시함
- 단순하고 직관적인 작업에 적합함
- 다만 작업 문맥이 복잡하거나 응답 형식이 명확하지 않을 경우, 일관성이 떨어질 수 있음

**원샷 (One-shot)**
- 하나의 입출력 예시를 제시한 뒤, 유사한 작업을 지시함
- LLM이 예시의 형식, 출력 구조를 모방하도록 유도함
- 일정한 포맷을 따르는 작업에서 효과적이나, 예시 하나에 과적합될 수 있음

**퓨샷 (Few-shot)**
- 2-5개의 예시 입력-출력 쌍을 제공하여, LLM이 작업 패턴을 일반화하도록 유도함
- 복잡한 문장 구조, 고정된 출력 포맷, 응답 일관성이 요구되는 작업에 효과적임
- 입력 문장과 유사한 예시만을 동적으로 선택하여 높은 응답 품질을 유지할 수 있음
- 다만 프롬프트가 길어져 토큰 비용이 증가하고, LLM 입력 한도를 초과할 수도 있음

### 3\) 추론 유도 기법

**CoT (Chain Of Thought)**
- LLM이 복잡한 문제를 해결할 때 중간 사고 과정을 명시적으로 기술하도록 유도함
- 정답만 제시하는 방식보다 사고 흐름이 명확하게 드러나므로 신뢰도와 설명 가능성(explainability)이 높아짐

**Self-consistency**
- 모델이 하나의 문제에 대해 다양한 사고 경로를 따라 여러 번 추론하도록 유도하는 기법임
- 각 경로에서 도출된 결과 간 일관성을 비교하여 정확한 답변을 선택함
- 다양한 접근을 통해 얻은 결과가 일치하는지 확인함으로써 오류 가능성을 낮출 수 있음

**PAL (Program-Aided Language)**
- 자연어 문제를 프로그래밍적 사고방식으로 구조화하여 해결하도록 유도하는 기법임
- 문제를 코드 또는 의사코드 형태로 변환하여 처리함

**Reflexion**
- AI가 자신의 이전 응답을 검토하고 평가한 후 개선하도록 유도하는 메타인지적 기법임
- 자기 피드백(self-critique)을 통해 답변의 완성도를 높이는 방식임
- 반복적인 자기 점검을 통해 점진적으로 더 나은 결과물을 생성할 수 있음

> ⚠️ 추론 기법에 대한 상세 내용은 별도 글에서 다룰 예정임

---

## 2. 프롬프트 엔지니어링 (Prompt Engineering)

### 1\) 개념
- 프롬프트 엔지니어링은 LLM에게 명확하고 목적에 부합하는 지시를 제공함으로써, 원하는 결과를 안정적으로 얻어내는 기술임
- 단순 질의에서 고도화된 작업 지시까지, 프롬프트의 설계 방식에 따라 출력 품질이 크게 달라짐
- 잘 설계된 프롬프트 템플릿은 재사용 가능하며, 다양한 입력에 대해 일관된 응답 품질과 정확도를 확보할 수 있음

### 2\) 프롬프트 설계 원칙

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

### 1\) PromptTemplate

- 입력 변수를 포함한 프롬프트 양식을 정의할 수 있는 LangChain의 기본 템플릿 클래스임
- 변수에 값을 채워 넣어 다양한 상황에 맞는 프롬프트를 동적으로 생성할 수 있음
- 반복적으로 사용하는 프롬프트 구조를 템플릿화함으로써, 재사용성과 유지보수 효율을 높일 수 있음

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
    length="500",
    topic="인공지능",
    content="정의, 역사, 응용분야"
)
print(formatted_prompt)
"""
'다음 주제에 대해 500자 이내로 설명해주세요: 인공지능의 정의, 역사, 응용분야'
"""
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
    length="500",
    topic="인공지능",
    content="정의, 역사, 응용분야"
)
print(formatted_prompt)
"""
'다음 주제에 대해 500자 이내로 설명해주세요: 인공지능의 정의, 역사, 응용분야'
"""
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
partial_prompt = prompt.partial(topic="인공지능")

# 최종 프롬프트
final_prompt1 = partial_prompt.format(content="정의", length="100")
final_prompt2 = partial_prompt.format(content="역사", length="300")
final_prompt3 = partial_prompt.format(content="응용분야", length="100")
print(final_prompt1, final_prompt2, final_prompt3, sep="\n")
"""
다음 주제에 대해 100자 이내로 설명해주세요: 인공지능의 정의
다음 주제에 대해 300자 이내로 설명해주세요: 인공지능의 역사
다음 주제에 대해 100자 이내로 설명해주세요: 인공지능의 응용분야
"""
```

### 2\) ChatPromptTemplate
- 대화형 LLM과의 상호작용을 위해 설계된 템플릿 클래스임
- System, User, AI 등 역할 기반 메시지를 구조화하여 대화 흐름을 정의함
- 멀티턴 대화에서 문맥을 유지하며 자연스럽고 일관된 응답을 유도할 수 있음

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import (
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from pprint import pprint

# 메시지 템플릿 생성
system_message = SystemMessagePromptTemplate.from_template(
    "당신은 {role} 전문가입니다."
)
human_message = HumanMessagePromptTemplate.from_template(
    "{question}"
)

# 템플릿 생성
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message, human_message]
)

# 최종 프롬프트
formatted_prompt = chat_prompt.format(
    role="인공지능",
    question="인공지능의 정의를 설명해주세요."
)
pprint(formatted_prompt)
"""
'System: 당신은 인공지능 전문가입니다.\nHuman: 인공지능의 정의를 설명해주세요.'
"""
```

### 3\) FewShotChatMessagePromptTemplate

- 예시(입력-출력 쌍)를 메시지 형식으로 포함하여 few-shot 프롬프트를 구성할 수 있는 LangChain의 클래스임
- `ChatPromptTemplate` 클래스와 함께 사용되어, Chat 모델(ChatOpenAI, Claude 등)에 최적화된 few-shot 메시지 시퀀스를 자동으로 생성함
- 반복적인 예시 패턴을 템플릿 형태로 관리할 수 있어, 예시의 변경이나 확장이 용이하고 재사용성도 높음

```python
from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate
)
from pprint import pprint

# 예시
examples = [
    {
        "input": "노트북 화면이 선명하고 키보드 타건감이 좋아서 업무용으로 만족스럽습니다.",
        "output": "디스플레이, 키보드"
    },
    {
        "input": "무선 이어폰은 배터리도 오래가고 블루투스 연결이 끊기지 않아 편리해요.",
        "output": "배터리 수명, 블루투스 연결"
    },
    {
        "input": "이 공기청정기는 소음이 거의 없고 센서 반응 속도도 빨라요.",
        "output": "소음, 센서 반응 속도"
    }
]

# 예시 템플릿 생성
example_prompt = ChatPromptTemplate.from_messages(
    [('human', '리뷰: {input}'), ('ai', '기능 키워드:{output}')]
)

# 퓨샷 템플릿 생성
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)
final_prompt = ChatPromptTemplate.from_messages([
    ('system', '당신은 제품 리뷰에서 **기능 관련 핵심 키워드**를 추출하는 전문가입니다.'),
    few_shot_prompt,
    ('human', '리뷰: {input}'),
])

# 최종 프롬프트
formatted_prompt = final_prompt.format(
    input="로봇청소기 흡입력이 좋고 장애물 회피도 잘해서 만족합니다."
)
pprint(formatted_prompt)
"""
('System: 당신은 제품 리뷰에서 **기능 관련 핵심 키워드**를 추출하는 전문가입니다.\n'
 'Human: 리뷰: 노트북 화면이 선명하고 키보드 타건감이 좋아서 업무용으로 만족스럽습니다.\n'
 'AI: 기능 키워드:디스플레이, 키보드\n'
 'Human: 리뷰: 무선 이어폰은 배터리도 오래가고 블루투스 연결이 끊기지 않아 편리해요.\n'
 'AI: 기능 키워드:배터리 수명, 블루투스 연결\n'
 'Human: 리뷰: 이 공기청정기는 소음이 거의 없고 센서 반응 속도도 빨라요.\n'
 'AI: 기능 키워드:소음, 센서 반응 속도\n'
 'Human: 리뷰: 로봇청소기 흡입력이 좋고 장애물 회피도 잘해서 만족합니다.')
"""
```

---

## Reference

- 프롬프트 유형 설명 :🔗 [링크](https://www.promptingguide.ai/kr/introduction/basics)
- LangChain 문서 : 🔗 [API 문서](https://python.langchain.com/api_reference/core/prompts.html)