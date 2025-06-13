# LangChain - 체인 (Chains) 개요 

- 체인은 여러 구성요소(프롬프트, 모델, 파서 등)를 순차적으로 연결하여 하나의 실행 흐름을 만드는 구조임
- 단일 질의에 대한 단순 응답을 넘어, 여러 단계의 처리 과정을 묶어 자동화된 작업 흐름을 구성할 수 있음
- 체인을 사용하면 프롬프트 반복 구성, 모델 응답 후처리, 결과 포매팅 등의 작업을 모듈화하여 재사용성과 유지보수성을 높일 수 있음

---

## 1. LCEL (LangChain Expression Language)

- LCEL은 LangChain 컴포넌트를 `|` 연산자로 연결하여 선언적으로 체인을 구성하는 방식임
- 컴포넌트는 왼쪽에서 오른쪽으로 순차적으로 실행되며, 이전 출력이 다음 입력으로 전달됨  
- 정의된 체인은 하나의 `Runnable`로 간주되어, 다른 체인 구성 시 재사용 가능함  
- 배치 실행 시 내부 최적화를 통해 리소스를 절약하고 처리 속도를 향상시킬 수 있음  
- LCEL은 테스트, 실험, 복잡한 흐름 제어 등 다양한 시나리오에서 구조화된 체인을 빠르게 구성할 수 있는 효율적인 표현 방식임

---

## 2. Runnable 클래스

- LangChain의 모든 컴포넌트는 `Runnable` 인터페이스를 구현하여 일관된 방식으로 실행됨  
- 실행 메서드로는 `invoke` (단일 입력), `batch` (여러 입력), `stream` (스트리밍 처리) 등을 지원하며, 동기/비동기 처리 방식에 따라 다양하게 활용 가능  
- 모든 `Runnable` 컴포넌트는 `|` 연산자를 사용해 연결할 수 있으며, 이를 통해 재사용성과 조합성이 높은 체인을 구성할 수 있음

**RunnableSequence**
- 여러 `Runnable`을 순차적으로 연결하여 실행함
- LCEL로 연결한 체인은 내부적으로 `RunnableSequence`로 컴파일됨
- 일반적으로는 LCEL 문법을 활용하여 선언적으로 구현하는 방식을 선호함

**RunnableParallel**
- 여러 `Runnable` 객체를 딕셔너리 형태로 구성하여 병렬처리 가능함
- 동일한 입력값이 각 `Runnable`에 전달되며, 결과는 키-값 형태로 반환됨
- 주로 데이터 전처리, 변환, 포맷 조정 등에 활용되며, 다음 파이프라인 단계에서 요구하는 출력 형식으로 조정 가능함

**RunnableLambda**
- 사용자 정의 파이썬 함수를 `Runnable` 객체로 감싸는 래퍼 컴포넌트임
- 체인 내에 사용자 정의 로직을 손쉽게 통합할 수 있어 데이터 전처리·후처리 및 조건부 분기 처리 등에 유용함
- 다른 `Runnable` 객체들과 결합하여 유연하고 복잡한 처리 파이프라인 구성이 가능함

**RunnablePassthrough**
- 입력값을 변형 없이 그대로 다음 단계로 전달함
- `RunnablePassthrough`은 입력 데이터를 새로운 키로 매핑할 수 있어 복수 입력이 필요한 체인 구성 시 유용하게 활용 가능함
- 중간에 가공이 없는 투명한 데이터 흐름을 제공하므로 파이프라인 디버깅이 용이함

> ⚠️ 자세한 구현 방법은 [예제] 섹션을 통해 확인할 수 있음

---

## 3. OutputParser 클래스

- `OutputParser`는 LLM 응답을 원하는 데이터 형식으로 변환하는 데 사용하는 구성 요소임
- 문자열, JSON, XML 등 여러 구조화된 형태로 파싱할 수 있음
- 파싱된 출력은 다른 시스템이나 프로세스와 연동하는 데 유용함

### 1\) StrOutputParser

### 2\) JSONOutputParser

- `JSONOutputParser`는 LLM의 응답을 엄격한 JSON 형식으로 파싱하는 데 사용됨
- 출력값의 데이터 유효성 검증 및 일관된 스키마 보장에 유리함
- 일반적으로 LLM이 JSON 형식으로 응답하도록 `PromptTemplate`에 명시한 뒤, 해당 파서를 통해 결과를 구조화함
- 출력값을 바로 딕셔너리 형태로 변환해 사용할 수 있어, API 응답이나 다른 시스템과 연동 시 유용함

### 3\)  XMLOutputParser

- `XMLOutputParser`는 LLM의 응답을 계층적 구조를 갖는 XML 형식으로 파싱함
- XML은 노드 간의 관계 표현이 가능하여, 복잡한 데이터 구조나 문서형 응답을 표현할 때 효과적임
- 일반적인 JSON 보다 문서 중심의 구조나 메타데이터가 많은 응답을 다룰 때 유리함
- 내부적으로 XML 파싱을 위해 `defusedxml` 패키지를 사용하므로, 사전 설치가 필요함
- JSON에 비해 사용 빈도는 낮지만, RDF, 문서 포맷, 일부 산업용 스키마와 연계 시 유용하게 사용됨


---

## [예제] 여행 일정 도우미 만들기

> 💡 **Tip**
> - 하나의 Runnable, 프롬프트, 함수는 되도록 하나의 명확한 기능만 수행하도록 구성하는 것이 좋음
> - 복잡한 로직을 여러 단계로 나누어 구성하면 가독성과 유지보수성이 크게 향상됨

- 사용자의 자유로운 여행 텍스트를 요약해 구조화된 정보로 변환함
- 요약된 정보를 기반으로 실제 여행 일정 테이블을 생성함

**1. Prompt**

```python
from langchain_core.prompts import ChatPromptTemplate

summarize_templete = """
{text}

위에 입력된 텍스트를 다음 항목으로 요약해주세요: 
- 여행 일정 :
- 교통편 일정 :
- 여행 장소 :
- 여행 스타일 :
- 예산 :
- 추천 숙소 :"""

summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 여행 일정 작성을 도와주는 AI 어시스턴트입니다."),
    ("human", summarize_templete)
])

planner_prompt = ChatPromptTemplate.from_template("""
다음 텍스트의 여행 일정을 기반으로 세부 여행 일정을 짜주세요.
텍스트: {summary}
규칙:
1. 날짜 및 시간과 장소, 세부 계획 항목으로 표 형태로 작성하세요.
2. 여행 스타일과 추천 숙소, 예산에 맞추어 동선을 고려하여 장소를 추천하세요.
답변:""")
```

**2. Chain**

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from IPython.display import display, Markdown

model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.4,
    top_p=0.7
)

# 체인 구성
summarize_chain = summarize_prompt | model | StrOutputParser()
planner_chain = planner_prompt | model | StrOutputParser()

# 최종 체인
chain = (
    summarize_chain |
    RunnableParallel(
        summary=RunnablePassthrough(),
        plan=lambda x: planner_chain.invoke({"summary": x})
    ) |
    RunnableLambda(lambda x: f"<요약>\n{x['summary']}\n\n<일정>\n{x['plan']}")
)

# 체인 실행
text = """내일 오전 8시에 서울역에서 출발해서 오전 11시에 부산역에 도착해.
2박 3일동안 부산 기장군 부근에서 여행하고 싶어.
맛있는 거 먹으면서 돌아다니고 싶고, 명소도 가고 싶어.
그런데 자동차가 없어서 걸어다니거나 대중교통을 이용해야해.
그리고 여행 마지막 날은 오후 5시에 부산역에서 출발해.
여동생이랑 둘이서 가려고 하고, 예산은 50만원 내외로 부탁해."""
result = chain.invoke({"text": text})
display(Markdown(result))
```

---

## 4. 사용자 정의 Output Parser

- 기본 `OutputParser`로 처리하기 어려운 복잡한 출력 형식이나 도메인 특화된 요구사항에 대응하기 위해, 사용자 정의 파서를 직접 구현할 수 있음


```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4.1-mini", model_provider="openai")
```


```python
from langchain_core.prompts import PromptTemplate

# 프롬프트 템플릿
step_prompt = PromptTemplate(
    template="""다음 텍스트에 대해서 작업을 순서대로 수행하세요:

    [텍스트]
    {text}

    [작업 순서]
    1. 텍스트를 1문장으로 요약
    2. 핵심 키워드 3개 추출
    3. 감정 분석 수행(긍정/부정/중립)

    [작업 결과]
    """,
    input_variables=["text"]
)

# 입력 텍스트
text = """
양자 컴퓨팅은 양자역학의 원리를 바탕으로 데이터를 처리하는 새로운 형태의 계산 방식이다.
기존의 고전적 컴퓨터는 0과 1로 이루어진 이진법(bit)을 사용하여 데이터를 처리하지만,
양자 컴퓨터는 양자 비트(큐비트, qubit)를 사용하여 훨씬 더 복잡하고 빠른 계산을 수행할 수 있다.

큐비트는 동시에 0과 1의 상태를 가질 수 있는 양자 중첩(superposition) 상태를 활용하며,
이를 통해 병렬 계산과 같은 고급 기능이 가능하다.
"""
```

### 1\) RunnableLambda 기반 방식

- LLM 응답에서 특정 키워드를 추출, 조건 분기 처리, 외부 함수 호출 등 고유한 후처리 로직을 삽입할 수 있음
- 고정된 데이터 구조 이외의 유연한 형식을 다루거나, 모델 출력의 후처리를 코드 기반으로 상세히 조절할 때 유용함


```python
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from typing import Dict

# 사용자 정의 파서
def custom_parser(ai_message: AIMessage) -> Dict:
    """모델 출력을 리스트 형태로 변환"""
    return ai_message.content.split('\n')

# 실행
chain = step_prompt | llm | RunnableLambda(custom_parser)
result = chain.invoke({"text": text})

# 결과 출력
pprint(result)
```

    ['1. 요약: 양자 컴퓨팅은 양자역학의 원리를 이용해 큐비트를 통한 중첩 상태로 기존 컴퓨터보다 훨씬 빠르고 복잡한 계산을 수행하는 새로운 '
     '계산 방식이다.  ',
     '2. 핵심 키워드: 양자 컴퓨팅, 큐비트, 중첩(superposition)  ',
     '3. 감정 분석: 중립']
    

### 2\) typing 기반 방식

- 출력 구조를 가볍게 명시하면서, 간단 JSON 응답을 기대할 때 사용함


```python
from typing import TypedDict, Annotated 

# 구조화된 출력 스키마
class AnalysisResult(TypedDict):
    """분석 결과 스키마"""
    summary: Annotated[str, ..., "핵심 요약"]   # ...은 필수 입력을 의미
    keywords: Annotated[list[str], ..., "주요 키워드"]
    sentiment: Annotated[str, ..., "긍정/부정/중립"]

structured_llm = llm.with_structured_output(AnalysisResult)

# 실행
chain = step_prompt | structured_llm
output = chain.invoke({"text": text})
pprint(output)
```

    {'keywords': ['양자 컴퓨팅', '큐비트', '양자 중첩'],
     'sentiment': '중립',
     'summary': '양자 컴퓨팅은 양자역학 원리를 이용해 큐비트를 통해 고전 컴퓨터보다 훨씬 빠르고 복잡한 계산을 가능하게 하는 새로운 '
                '계산 방식이다.'}
    

### 3\) pydantic 기반 방식

- 데이터 타입 검증, 필수 항목 검사, 상세 오류 처리까지 가능한 견고한 방식임
- 구조가 명확한 응답을 기대할 수 있어, API 응답 처리, DB 저장, UI 렌더링 등과의 연동에 효과적임


```python
from typing import List, Literal
from pydantic import BaseModel, Field

# 구조화된 출력 스키마
class AnalysisResult(BaseModel):
    """분석 결과 스키마"""
    summary: str = Field(...,  description="텍스트의 핵심 내용 요약")
    keywords: List[str] = Field(..., description="텍스트에서 추출한 주요 키워드")
    sentiment: Literal["긍정", "부정", "중립"] = Field(
        ..., 
        description="텍스트의 전반적인 감정 분석 결과"
    )

structured_llm = llm.with_structured_output(AnalysisResult)

# 실행
chain = step_prompt | structured_llm
output = chain.invoke({"text": text})
print(output.summary)
print(output.keywords)
print(output.sentiment)
```

    양자 컴퓨팅은 양자 중첩 상태를 활용하는 큐비트를 통해 기존 고전 컴퓨터보다 더 복잡하고 빠른 계산을 가능하게 하는 새로운 계산 방식이다.
    ['양자 컴퓨팅', '큐비트', '양자 중첩']
    긍정
    
---

## [예제] 학습 도우미 만들기

***환경 설정***


```python
from dotenv import load_dotenv
load_dotenv()
```

### 1\) 퀴즈 생성 챗봇


```python
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 스키마 정의의
class QuizQuestion(BaseModel):
    """퀴즈 스키마"""
    question: str = Field(..., description="퀴즈 문제")
    options: List[str] = Field(..., description="보기 (4개)")
    correct_answer: int = Field(..., description="정답 번호 (1-4)")
    explanation: str = Field(..., description="정답 설명")


# 프롬프트 탬플릿
quiz_prompt = PromptTemplate(
    template="""다음 주제에 대한 퀴즈 문제를 만들어주세요:
    
주제: {topic}
난이도(상/중/하): {difficulty}

다음 조건을 만족하는 퀴즈를 생성해주세요:
1. 문제는 명확하고 이해하기 쉽게
2. 4개의 보기 제공
3. 정답과 오답은 비슷한 수준으로
4. 상세한 정답 설명 포함""",
    input_variables=["topic", "difficulty"]
)

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)

# 구조화된 Outupt Parser 설정
structured_llm = llm.with_structured_output(QuizQuestion)

# 실행
chain = quiz_prompt | structured_llm
output = chain.invoke({"topic": "LangChain", "difficulty": "상"})

# 결과 출력
pprint(f"퀴즈 문제: {output.question}")
pprint(f"보기: {output.options}")
pprint(f"정답: {output.correct_answer}")
pprint(f"정답 설명: {output.explanation}")
```

    "퀴즈 문제: LangChain에서 '체인(chain)'의 주요 역할은 무엇인가?"
    ("보기: ['여러 개의 LLM 호출을 순차적으로 연결하여 복잡한 작업을 수행한다.', '데이터베이스와의 연결을 관리한다.', '사용자 "
     "인터페이스를 구성하는 모듈이다.', '모델 학습을 위한 데이터 전처리를 담당한다.']")
    '정답: 1'
    ("정답 설명: LangChain에서 '체인'은 여러 개의 언어 모델 호출을 순차적으로 연결하여 복잡한 작업을 수행하는 역할을 합니다. 이를 "
     '통해 단일 모델 호출로는 어려운 복합적인 작업을 단계별로 처리할 수 있습니다. 데이터베이스 연결이나 UI 구성, 데이터 전처리는 각각 '
     '다른 컴포넌트가 담당합니다.')
    

### 2\) 개념 설명 챗봇


```python
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 스키마 정의의
class ConceptExplanation(BaseModel):
    """개념 설명 스키마"""
    topic: str = Field(..., description="주제")
    explanation: str = Field(..., description="개념 설명")
    examples: str = Field(..., description="사용 예시")
    related_concepts: List[str] = Field(..., description="관련된 개념 (4개)")

# 프롬프트 탬플릿
concept_prompt = PromptTemplate(
    template="""다음 주제에 대해 차근차근 설명해 주세요:
    
주제: {topic}
난이도(상/중/하): {difficulty}

다음을 차례대로 작성하세요:
1. 주제에 대한 개념 설명
2. 주제에 대한 사용 예시
3. 관련 개념 목록 (4개)
""",
    input_variables=["topic", "difficulty"]
)

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)

# 구조화된 Outupt Parser 설정
structured_llm = llm.with_structured_output(ConceptExplanation)

# 실행
chain = quiz_prompt | structured_llm
output = chain.invoke({"topic": "LangChain", "difficulty": "하"})

# 결과 출력
pprint(f"주제: {output.topic}")
pprint(f"설명: {output.explanation}")
pprint(f"예시: {output.examples}")
pprint(f"관련 개념: {output.related_concepts}")
```

    '주제: LangChain'
    ('설명: LangChain은 언어 모델을 활용하여 다양한 애플리케이션을 개발할 수 있도록 돕는 프레임워크입니다. 주로 자연어 처리 작업을 '
     '쉽게 연결하고 확장할 수 있게 설계되었습니다.')
    '예시: 예를 들어, LangChain을 사용하면 텍스트 요약, 질문 응답, 대화형 에이전트 등을 쉽게 구현할 수 있습니다.'
    "관련 개념: ['자연어 처리', '언어 모델', '프레임워크', 'AI 애플리케이션']"

---

## Reference

- LangChain 문서 : 🔗 [Runnable](https://python.langchain.com/api_reference/core/runnables.html) / [OutputParser](https://python.langchain.com/api_reference/core/output_parsers.html)