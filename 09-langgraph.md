# LangGraph

- 🔗 [Graph API 개념](https://langchain-ai.github.io/langgraph/concepts/low_level/) / [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/) / [Functional API](https://langchain-ai.github.io/langgraph/concepts/functional_api/) / [Workflows and Agents](https://langchain-ai.github.io/langgraph/tutorials/workflows/)
- 복잡한 워크플로우(에이전트, 다단계 문서 처리 등)를 그래프 형태로 모델링할 수 있음
- 그래프의 노드와 엣지 단위로 데이터 흐름을 시각적이고 구조적으로 표현 가능함
- LangChain과 함께 사용해 체인 기반 처리와 그래프 기반 로직을 병행할 수 있음
- LangChain의 직렬 구조보다 분기, 병합, 루프 등 복잡한 로직을 더 유연하게 구현 가능함
- 노드/엣지 단위 모듈화로 재사용성과 확장성이 높음
- 그래프 시각화 도구와 연동해 전체 흐름을 직관적으로 디버깅할 수 있음

---

## 1. 핵심 구성 요소

**State**
- 앱 전체에서 공유되는 데이터 스냅샷을 나타내는 구조임
- 일반적으로 `TypedDict` 또는 Pydantic의 `BaseModel` 형태로 정의함
- 각 노드 실행 결과로 상태가 덮어쓰기(override) 되어 업데이트됨
- 상태 기반으로 데이터 흐름을 체계적으로 제어할 수 있음

**Nodes**
- 에이전트의 개별 행동 단위로, 주어진 상태를 입력받아 처리함
- 내부적으로 함수 또는 연산 로직을 실행하고, 새로운 상태값을 반환함
- 하나의 노드는 하나의 작업(task)를 수행함
- 각 노드는 다른 노드와 연결되어 데이터 흐름을 형성함

**Edges**
- 현재 상태를 따라 다음에 실행할 노드를 결정함
- 조건 분기 로직이 포함될 수 있음
- 노드 간의 실행 순서를 제어하며, LangGraph의 흐름을 결정함

---

## 2. Graph 작성

### 1\) Graph 생성

**StateGraph**
- 상태 기반 그래프 구조를 정의하는 핵심 클래스임
- 대화나 처리 흐름을 `START` → `END` 구조로 체계적으로 구성함
- 노드 간 전환은 엣지를 통해 정의하며, 복잡한 조건 분기도 처리 가능함

**add_node**
- 그래프에 새로운 노드(작업 단위)를 추가하는 메소드임
- 각 노드는 독립적인 함수를 실행하며, 상태를 입력받아 결과를 반환함

**add_edge**
- 두 노드 간의 직접적인 실행 순서(연결 관계)를 정의하는 메소드임
- 한 노드의 실행이 끝난 후 다음 노드로 흐름을 이동시킴

**add_conditional_edges**
- 특정 조건에 따라 다음 노드를 동적으로 선택하는 분기 로직을 정의하는 메소드임
- 상태 값 또는 함수 변환값에 따라 흐름이 달라질 수 있음

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from IPython.display import Image, display

# 1. 상태 정의
class MyState(TypedDict):
    name: str
    is_morning: bool

# 2. 노드 함수 정의
def greet_user(state: MyState) -> MyState:
    print(f"Hi, {state['name']}!")
    return state

def say_good_morning(state: MyState) -> MyState:
    print("Good morning!")
    return state

def say_hello(state: MyState) -> MyState:
    print("Hello!")
    return state

# 3. 조건 함수 정의
def is_morning(state: MyState) -> Literal["morning", "not_morning"]:
    return "morning" if state["is_morning"] else "not_morning"

# 4. 그래프 구성
builder = StateGraph(MyState)

builder.add_node("greet_user", greet_user)
builder.add_node("say_good_morning", say_good_morning)
builder.add_node("say_hello", say_hello)

builder.add_edge(START, "greet_user")
builder.add_conditional_edges(
    "greet_user",
    is_morning,
    {
        "morning": "say_good_morning",
        "not_morning": "say_hello",
    },
)
builder.add_edge("say_good_morning", END)
builder.add_edge("say_hello", END)

# 5. 그래프 컴파일
graph = builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
```

![alt text](_png/09-langgraph00.png)

### 2\) Graph 실행

**invoke**
- 그래프를 한 번 실행하여 최종 결과 상태를 반환함

```python
graph.invoke({"name": "Bob", "is_morning": True})

"""
Hi, Bob!
Good morning!
{'name': 'Bob', 'is_morning': True}
"""
```

**stream**
- 그래프 실행 과정을 스트리밍 형태로 순차 출력함

- `stream_mode="values"`
    - 상태 값의 변경 내역만 출력함
    - 각 노드 실행 이후의 상태를 확인할 수 있음

```python
for step in graph.stream({"name": "Bob", "is_morning": False}, stream_mode="values"):
    print(step)
    print("---"*10)

"""
{'name': 'Bob', 'is_morning': False}
------------------------------
Hi, Bob!
{'name': 'Bob', 'is_morning': False}
------------------------------
Hello!
{'name': 'Bob', 'is_morning': False}
------------------------------
"""
```

- `stream_mode="updates"`
    - 어떤 노드가 어떤 값을 업데이트했는지까지 출력됨
    - 상태 변환뿐 아니아 노드별 실해우 결과 추적이 가능하여 디버깅용으로 사용할 수 있음

```python
for step in graph.stream({"name": "Bob", "is_morning": False}, stream_mode="updates"):
    print(step)
    print("---"*10)

"""
Hi, Bob!
{'greet_user': {'name': 'Bob', 'is_morning': False}}
------------------------------
Hello!
{'say_hello': {'name': 'Bob', 'is_morning': False}}
------------------------------
"""
```

---

## 3. Graph 고급 기능

### 1\) Command

- LangGraph 핵심 제어 도구로, 노드 함수의 반환값으로 사용됨
- 상태 업데이트와 다음 노드 지정이라는 두 가지 역할을 동시에 수행할 수 있음
- 복잡한 흐름 제어나 정보 전달이 필요한 상황에서 유용함
- 그래프 실행 중 동적으로 상태를 수정하거나 분기를 제어할 수 있음

**Command** vs **add_conditional_edges**
- `Command`는 상태 업데이트와 노드 이동을 동시에 처리할 때 사용되며, 특히 정보 전달이 필요한 복잡한 전환에 적합함
- `add_conditional_edged`는 단순한 분기 처리에 사용되며, 상태 변경 없이 조건에 따른 이동만 수행함
- 상태 업데이트 필요 여부에 따라 두 방식의 선택 기준이 결정될 수 있음


### 




---

## Command



## Graph API

## Reducer

[](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers)

## Messages

- LangGraph는 메시지 목록 기반의 채팅 모델 인터페이스를 활용함
- 그래프 상태에서 대화 기록은 메시지 객체 리스트로 저장되며, 이를 통해 효율적인 대화 관리가 가능함
- reducer 함수를 통해 상태 업데이트 시 메시지 목록이 어떻게 갱신될지 정의할 수 있음

`add_messages`
- 메시지 ID를 기반으로 기존 메시지를 업데이트하거나 새 메시지를 추가하는 고급 관리 기능을 제공함
- 기존 메시지의 중복 추가를 방지함

`MessagesState`

## Map-Reduce 패턴

- 동적으로 엣지를 생성하고, 개별 상태를 전달하는 방식임 (분산처리)


```python
from typing import Annotated, List, TypedDict, Optional
from pydantic import BaseModel
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import operator
from IPython.display import Image, display

# 팩트 체크 결과를 위한 Pydantic 모델
class FactCheckResult(BaseModel):
    sentence: str
    score: float

# 전체 상태 정의 (글로벌 상태)
class OverallState(TypedDict):
    query: str  # 검색 쿼리
    search_results: Optional[str]  # 검색 결과
    summary: Optional[str]  # 요약문
    fact_check: Annotated[List[FactCheckResult], operator.add]  # 팩트체크 결과 (누적)

# 로컬 상태 (단일 문장 팩트체크용)
class SentenceState(TypedDict):
    sentence: str  # 팩트체크할 문장


def search_info(state: OverallState) -> OverallState:
    search_tool = DuckDuckGoSearchResults(output_format="list")
    query = state["query"]

    # 검색 실행
    results = search_tool.invoke({"query": query})

    # 상위 3개 결과만 사용 (snippet 필드)
    filtered_results = [item['snippet'] for item in results][:3]

    return {
        "search_results": filtered_results
    }

def generate_summary(state: OverallState) -> OverallState:
    if not state["search_results"]:
        return {"summary": "검색 결과가 없습니다."}

    summary_prompt = """
    다음 검색 결과들을 요약해주세요:
    {search_results}

    핵심 포인트 3-4개로 간단히 요약:
    """

    summary = llm.invoke(summary_prompt.format(
        search_results="\n\n".join(state["search_results"])
    ))

    return {"summary": summary.content}

def fact_check_sentences(state: OverallState):
    if not state["summary"]:
        return {"fact_check": []}

    # 요약된 문장들을 분리 (간단하게 개항문자로 분리)
    sentences = state["summary"].split("\n\n")
    sentences = [s.strip() for s in sentences if s.strip()]  # 빈 문자열 제거

    print(f"Fact-checking {len(sentences)} sentences...")

    # 각 문장에 대해 팩트 체크 작업을 생성 (Send 사용)
    return [
        Send("fact_check_sentence", {"sentence": s}) for s in sentences
    ]

def fact_check_single_sentence(state: SentenceState) -> OverallState:
    """개별 문장에 대한 팩트체크 수행"""
    sentence = state["sentence"]
    print(f"Fact-checking sentence: {sentence}")

    prompt = f"""
    다음 문장의 사실 여부를 평가하고 신뢰도 점수를 0과 1 사이로 제공해주세요:
    문장: {sentence}
    신뢰도 점수:
    """
    response = llm.invoke(prompt)

    # 팩트체크 결과 생성
    print(f"Fact-check result: {response.content}")
    
    try:
        score = float(response.content)
        score = max(0.0, min(1.0, score))  # 0과 1 사이로 제한
    except ValueError:
        score = 0.5  # 기본값
    
    return {
        "fact_check": [FactCheckResult(sentence=sentence, score=score)]
    }

# 그래프 구성
builder = StateGraph(OverallState)

# 노드 추가
builder.add_node("search", search_info)
builder.add_node("generate_summary", generate_summary)
builder.add_node("fact_check_sentence", fact_check_single_sentence)

# 엣지 추가
builder.add_edge(START, "search")
builder.add_edge("search", "generate_summary")
builder.add_conditional_edges(
    "generate_summary",
    fact_check_sentences,
    ["fact_check_sentence"]
)

builder.add_edge("fact_check_sentence", END)

# 그래프 컴파일
graph = builder.compile()

# 그래프 시각화
display(Image(graph.get_graph().draw_mermaid_png()))

# 사용자 질문
inputs = {"query": "기후 변화의 주요 원인은 무엇인가요?"}

# 그래프 실행
result = graph.invoke(inputs)
pprint(result)
```

## Tool Node

- 모델이 사전에 정의된 도구 호출을 실행하는 역할하는 LangGraph 구성요소임

```python
from langgraph.prebuilt import ToolNode

# 도구 노드 정의
db_tool_node = ToolNode(tools=tools)

# LLM 모델 생성
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# 도구를 바인딩하여 모델 생성
llm_with_tools = llm.bind_tools(tools=tools)

# 도구 호출 - 한국어
tool_call = llm_with_tools.invoke([HumanMessage(content=f"테슬라는 누가 창립했나요?")])

# 도구 호출 내용 출력
pprint(tool_call.tool_calls)
print("-" * 100)

# 도구 호출 결과를 메시지로 추가하여 실행
results = db_tool_node.invoke({"messages": [tool_call]})

# 실행 결과 출력하여 확인
for result in results['messages']:
    print(f"메시지 타입: {type(result)}")
    print(f"메시지 내용: {result.content}")
    print()
```

## ReAct Agent (Reasoning and Acting)

- 가장 일반적인 에이전트로, 모델이 특정 도구를 호출하고 출력을 다시 모델에 전달하는 과정을 거침
- 모델이 도구 출력을 바탕으로 다음 행동을 결정함
- 반복되는 과정에서 또 다른 도구를 호출하거나 직접 응답을 생성함

## InMemorySaver

- [LangGraph Memory](https://langchain-ai.github.io/langgraph/concepts/memory/)
- LangGraph에서 제공하는 스레드 기반의 단기 메모리(short-term memory)로, 하나의 대화 세션 동안 대화 내용을 저장하고 추적함
- 에이전트의 상태로 단기 메모리를 관리하며, 체크포인터를 통해 데이터베이스에 저장됨
- 메모리는 그래프 실행 또는 단계 완료 시 업데이트 되며, 각 단계 시작 시 상태를 읽어들임
- 메시지 기록을 통해 대화 맥락을 유지하고, 체크포인트 기능으로 통해 언제든지 대화 재개가 가능함
- 그래프 실행 또는 단계 완료 시 자동 업데이트됨
- `SqliteSaver` 또는 `PostgresSaver`를 사용할 수도 있음

### Checkpoints
- 그래프를 컴파일할 때 체크포인터를 지정할 수 있음
- 체크포인터는 그래프의 각 단계에서 상태를 기록함
- 그래프 각 단계의 모든 상태를 컬렉션으로 저장함
- `thread_id`를 사용하여 접근 가능함
- `graph.get_state` 메소드로 스레드의 최신 상태를 조회할 수 있음
- `checkpoint_id`를 지정하여 특정 체크포인트 시점의 상태를 가져올 수 있음
- 반환값은 `StateSnapshot` 객체 리스트 형태임
- 리스트의 첫 번째 요소가 가장 최근 체크포인트임

### Replay

- `thread_id`와 `checkpoint_id`를 지정하여 특정 체크포인트 이후부터 실행 가능함
- 체크포인트 이전 단계는 재생만 하고 실제로 실행하지 않음
- 따라서 불필요한 단계의 재실행을 방지하여 효율적인 처리가 가능함
- `graph.update_state` 메소드를 통해 그래프 상태를 직접 수정할 수 있음


## InMememoryStore

## Human-in-the-Loop (사용자 개입)
- HITL은 AI 시스템에 인간의 판단과 전문성을 통합하는 시스템임
- `Breakpoints`로 특정 단계에서 실행 중지가 가능함
    - `Breakpoints`는 체크포인트 기능 기반으로 작동하는 시스템임
    - 각 노드 실행 후 그래프의 상태를 스레드에 저장하여 나중에도 접근이 가능함
    - 그래프 실행을 특정 지점에서 일시 중지하고 사용자 승인 후 재개 가능하도록 구현 가능함
- 사용자의 입력이나 승인을 기다리는 패턴으로 작동함
- 시스템 결정에 대한 인간의 통제와 검증을 보장함

