# LangChain 개념

- LLM 기반 애플리케이션 개발을 위한 프레임워크임
- LLM 애플리케이션 라이프사이클의 모든 단계를 다루는 오픈 소스 라이브러리와 도구 모음을 제공함
- Chain과 Agent라는 두 가지 핵심 기능을 통해 LLM 애플리케이션을 효율적으로 개발하도록 지원함
    - Chain : 작업을 순차적으로 실행하는 파이프라인 구조 제공
    - Agent : 자율적 의사결정이 가능한 실행 단위
        - 최근에는 복잡한 흐름 제어를 위해 **LangGraph** 기반으로 확장하여 구현하기도 함

<div style="text-align: center;">
    <img src="https://python.langchain.com/svg/langchain_stack_112024_dark.svg" 
        alt="langchain_stack" 
        width="600" 
        style="border: 0;">
</div>

---

## 1. 등장 배경

- LLM을 사용하여 요구 사항에 따라 사용자 질의를 해석할 필요성이 생김
- 이전부터 여러 시스템이나 백엔드 툴을 이용해야 하는 요구 사항이 흔하게 있음
    - 애플리케이션이 두 가지 이상의 DB가 필요할 때 매번 DB 접속을 위한 커넥터를 새로 만들면, 비용이 비쌈 (전문 개발자 필요, 유지/보수/관리/배포 오버헤드 발생)
    - DBMS에 엑세스 하기 위한 표준 API 레이어인 **ODBC**(Open Database Connectivity)와 **JDBC**(Java Database Connectivity)가 등장함
    - 이런 표준 API 레이어는 DB 시스템과 운영체제가 독립적으로 동작함
    - 따라서 애플리케이션에서 데이터 액세스 코드를 거의 변경하지 않고도 클라이언트와 서버의 다른 플랫폼으로 인식할 수 있다는 장점이 있음
- LangChain은 LLM 기반 애플리케이션 개발을 위한 개발자 친화적인 오픈 소스 프레임워크임
    - Python과 JavaScript 라이브러리를 제공함
    - LLM 애플리케이션을 구축한 다음 통합할 수 있는 중앙 집중식 개발 환경을 갖추고 있음

---

## 2. 구성요소 (Components)

### 1\) 모델 (Models)

- 텍스트 생성 작업을 중심으로, 일반 텍스트 입력(LLM) 또는 메시지 기반 입력(Chat Model)을 처리함
- 다양한 LLM(OpenAI, HuggingFace, Cohere 등)을 추상화하여 공통 인터페이스로 제공함
- 사용자는 각 모델의 세부 구현을 몰라도 동일한 방식으로 호출 가능함
- 복잡한 NLP 작업을 단순한 코드로 실행할 수 있어 개발 부담이 줄어듦
- API 키만으로 대부분의 언어 모델을 쉽게 연동할 수 있도록 설계되어 있음

### 2\) 프롬프트 (Prompts)

- 프롬프트는 사용자가 LLM에 태스크를 수행하라고 전달하는 명령어임
- LLM이 정확하고 유용한 정보를 제공하도록 프롬프트를 정교하게 작성해야함
    - 원하는 정보나 결과의 유형을 명확하게 정의하여 질문해야 함
    - 핵심 정보에 집중하고 불필요한 내용은 제거하여 중요하지 않은 정보에 LLM이 주의를 빼앗기지 않도록 해야 함
    - 열린 질문을 사용하면 LLM이 더 풍부하고 다양한 답변을 제공함
    - LLM이 문맥을 이해할 수 있도록 필요한 배경 정보를 제공하면, Hallucination 발생 가능성을 줄이고 관련성 높은 응답을 유도할 수 있음
    - 대화 흐름과 상황에 맞는 언어와 표현을 선택하여 LLM이 더 적절한 통과 내용으로 응답할 수 있도록 유도해야 함
- LangChain에는 명령을 명확히 전달할 수 있도록 도와주는 `PromptTemplate` 클래스가 있음
    - 컨텍스트와 질의를 수동으로 일일이 작성할 필요 없이, 프롬프트 구성을 구조화할 수 있음
    - 템플릿 내부에 지침을 포함하거나, Few-Shot 예제를 함께 넣어 사용할 수 있음
    - 변수 치환을 통해 동적 프롬프트를 구성할 수 있어, 다양한 입력에 대해 유연하게 대응 가능함

### 3\) 체인 (Chains)

- 체인은 LLM을 다른 구성 요소와 결합하여 일련의 태스크를 실행하여 애플리케이션을 생성함
- 체인의 각 태스크는 서로 다른 프롬프트나 LLM을 사용할 수 있음
- ex) 사용자 입력을 먼저 요약한 후, 요약된 내용을 기반으로 답변을 생성하거나 외부 API를 호출하는 등 단계적 처리 로직을 정의할 수 있음

### 4\) 인덱스 (Indices)

- 최적의 답변을 위해서는 LLM 뿐만 아니라, 출처가 확실한 고급 전문 정보와 최신 정보가 추가로 필요함 (비용 효율화를 위해 RAG와 같은 기법을 사용함)
- **Index**는 외부 문서를 LLM이 효율적으로 검색하고 참조할 수 있도록 구성한 구조를 의미함
- 일반적으로 Document Loader → Text Splitter → Embedding → Vectorstore 과정을 거쳐 인덱스를 구성함
- 대표 구현체로는 `VectorstoreIndexWrapper`, `RetrieverIndex`, `SelfQueryRetriever` 등이 있음

### 5\) 메모리 (Memory)

- 사용자가 LLM과 프롬프트 기반을 사용하는 동안 사용자의 정보를 포함하여 주요 사실을 기억하고 상호 작용에 대한 정보를 적용할 수 있음
- 메모리를 통해 LLM이 이전 대화 내용을 반영하여 더 자연스럽고 일관성 있는 응답을 생성할 수 있음
- 대화 전체를 기억하는 옵션과 지금까지의 대화 요약만 기억하는 요약 옵션이 있음
- 체인과 에이전트는 메모리 기능이 없다면 입력값을 독립적으로 처리해 앞선 대화를 고려하지 못함

### 6\) 에이전트 (Agents)

- 에이전트는 툴 사용, 질의 결정, 반복 실행 등 자율적인 의사결정이 가능한 LLM 실행 단위임
- 외부 API 호출, 계산기, 데이터베이스 질의 등을 툴로 구성하여 복잡한 문제 해결 시 활용됨
- LangChain에서는 `Tool`, `AgentExecutor`, `LLMChain` 등을 조합하여 Agent를 구성함
- 복잡한 상태 기반 흐름 제어가 필요한 경우, **LangGraph**를 활용해 에이전트의 실행 흐름을 DAG 형태로 시각화 및 제어할 수 있음

---

## 3. 설치 및 설정

- `langchain` 프레임워크를 설치하면 `langchain-core`, `langchain-community`, `langsmith` 등 프로젝트 수행에 필수적인 라이브러리들이 함께 설치됨
    - `langchain-core` : 다른 패키지에 최소한으로 의존하면서 핵심 기능을 제공함
    - `langchain-community` : 외부 서비스 및 플랫폼과의 통합을 담당함
    - `langchain-experimental` : 아직 안정화되지 않은 새로운 기능들을 포함함 (애플리케이션 개발에는 권장하지 않음)
- 다양한 외부 모델 제공자와 데이터 저장소 등과 통합을 위해서는 의존성 설치가 따로 필요함
- 만약 OpenAI에서 제공하는 LLM을 사용하려면 `langchain-openai`, `tiktoken` 라이브러리를 설치해야 함
    - `langchain-openai` : GPT-3.5, GPT-4 등 LLM 모델과 기타 보조 도구를 제공함
    - `tiktoken` : OpenAI 모델이 사용하는 Tokenizer를 제공함
- 인증 정보(API Key 등)는 `.env` 파일에 저장하고, 실행 시 환경 변수로 로드해서 사용함

```python
# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()
```

---

## 4. LangChain 구성 요소 맛보기

**Model** : `langchain_openai` 라이브러리의 `ChatOpenAI` 클래스를 통해 OpenAI 모델을 호출하여 사용할 수 있음

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4.1-mini")

response = model.invoke("Glory를 한국어로 번역해주세요.")
print("답변: ", response.content)
print("메타데이터: ", response.response_metadata)
```

**Message** : 메시지 타입을 나누어 사용함으로써 역할을 명확히 구분하고, 대화 맥락을 구조화하며, 프롬프트를 유연하게 설계할 수 있음

- `SystemMessage`: 대화의 전반적인 맥락이나 규칙을 설정하는 메시지로, LLM의 응답 스타일, 역할, 목적 등을 정의할 때 사용함 (role: system)
- `HumanMessage`: 사용자가 입력한 메시지를 나타냄 (role: user)
- `AIMessage`: AI 모델이 생성한 응답 메시지를 나타내며, 이전 응답을 기록하거나 프롬프트에 포함시킬 때 사용함 (role: assistant)

```python
from langchain_core.messages import SystemMessage, HumanMessage

system_msg = SystemMessage(content="당신은 영어를 한국어로 번역하는 AI 어시스턴트입니다.")
human_message = HumanMessage(content="Glory")

response = model.invoke([system_msg, human_message])
print("답변: ", response.content)
```

**Prompt** : 프롬프트를 일관된 형식으로 작성하고 재사용 가능하게 관리할 수 있음

- `PromptTemplate` : 단일 텍스트 입력을 변수로 구성해 포맷팅할 수 있는 기본 프롬프트 템플릿
- `ChatPromptTemplate` : 여러 메시지(Human, AI, System 등)를 포함하는 대화형 프롬프트 템플릿
- `MessagesPlaceholder` : 기존 메시지 목록을 템플릿 내 특정 위치에 삽입할 수 있도록 도와주는 클래스

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {subject}에 능숙한 비서입니다"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])
```

**Memory** : 대화 기록을 저장 및 관리하며, 컨텍스트 유지를 위해 다양한 메모리 타입을 지원함 (대화 요약, 버퍼 저장 등 포함)

- `BaseChatMessageHistory` : 메시지 히스토리를 저장하고 불러오는 기본 클래스
- `RunnableWithMessageHistory` : 체인이나 에이전트 실행 시, 자동으로 메시지 기록을 연동해 사용하는 래퍼 클래스
- `ConversationBufferMemory` : 최근 대화 내용을 버퍼 형태로 유지하여 간단한 맥락을 지속적으로 제공하는 클래스
- `SummaryMemory` : 이전 대화 내용을 요약해 저장함으로써 긴 대화 히스토리도 압축된 형태로 관리할 수 있는 클래스

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing import List

# 메모리 기반 히스토리 구현
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    
    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)
    
    def clear(self) -> None:
        self.messages = []

# 세션 저장소
store = {}

# 세션 ID로 히스토리 가져오기
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]
```

**Chain** : 여러 구성 요소(LLM, 프롬프트, 툴 등)를 순차적으로 연결하여 복잡한 작업 흐름을 구성할 수 있음

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

# 체인 구성
chain = prompt | model | StrOutputParser()

# 히스토리 관리 추가
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

# 체인 실행, 히스토리 이용해서 대화 진행
response = chain_with_history.invoke(
    {"subject": "수학", "question": "1+2는 얼마인가요?"},
    config={"configurable": {"session_id": "user1"}}
)
print("답변: ", response)
```

---

## Reference

- [링크1](https://brunch.co.kr/@ywkim36/147)
- [링크2](https://wikidocs.net/231153)
- GPT API를 활용한 인공지능 앱 개발 [2판] | 올리비에 케일린, 마리-알리스 블레트 지음 | 이일섭, 박태환 옮김