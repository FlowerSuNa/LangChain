# LangGraph

- 🔗 [Graph API 개념](https://langchain-ai.github.io/langgraph/concepts/low_level/#multiple-schemas) / [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/) / [Functional API](https://langchain-ai.github.io/langgraph/concepts/functional_api/) / [Workflows and Agents](https://langchain-ai.github.io/langgraph/tutorials/workflows/)
- LangGraph는 에이전트(Agent) 워크플로우를 그래프로 모델링함

**핵심 구성 요소**
-`State`: 애플리케이션의 현재 스냅샷을 나타내는 공유 데이터 구조로, 일반적으로 `TypedDict` 또는 Pydantic의 `BaseModel` 형태를 사용함
    - 각 상태는 다른 상태에 의해 override(덮어쓰기) 될 수 있어 데이터를 유연하게 관리할 수 있음
    - 상태 관리를 통해 체계적인 데이터 처리와 흐름 제어 가능함
- `Nodes` : 에이전트의 로직을 인코딩하며, 현재 값을 `State` 입력으로 받고, 계산이나 부수 효과를 수행한 후 업데이트된 값을 반환함
    - 노드는 독립적인 작업 단위로, 특정 함수를 실행함
    - 각 노드는 다른 노드와 연결되어 데이터 흐름을 형성함
    - 상태를 입력으로 받아 처리하고 업데이트된 상태를 반환함
- `Edges` : 

---

## StateGraph
- 상태 기반의 그래프 구조를 사용하여 대화 흐름을 체계적으로 관리하는 그래프 클래스임

```python
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display


# LLM 인스턴스 생성
llm = ChatOpenAI(model="gpt-4.1-mini")

# 상태 정의
class State(TypedDict):
    original_text: str   # 원본 텍스트
    summary: str         # 요약본

# 요약 생성 노드
def generate_summary(state: State):
    """원본 텍스트를 요약하는 노드"""
    # 로직
    return {"summary": "Summary Content"}

# StateGraph 객체 생성 (Workflow)
workflow = StateGraph(State)

# 노드 추가
workflow.add_node("generate_summary", generate_summary)

# 엣지 추가
workflow.add_edge(START, "generate_summary")
workflow.add_edge("generate_summary", END)

# 그래프 컴파일
graph = workflow.compile()

# 그래프 시각화
display(Image(graph.get_graph().draw_mermaid_png()))
```


---

## Command
- LangGraph 핵심 제어 도구로, 노드 함숭의 반환값으로 사용함
- 상태 관리와 흐름 제어를 동시에 수행할 수 있어 효율적인 그래프 운영이 가능함
- 그래프의 상태를 동적으로 업데이트하면서 다음 실행할 노드를 지정할 수 있음


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

## MemorySaver

- [LangGraph Memory](https://langchain-ai.github.io/langgraph/concepts/memory/)
- LangGraph에서 제공하는 스레드 기반의 단기 메모리(short-term memory)로, 하나의 대화 세션 동안만 정보를 유지함
- 에이전트의 상태로 단기 메모리를 관리하며, 체크포인터를 통해 데이터베이스에 저장됨
- 메모리는 그래프 실행 또는 단계 완료 시 업데이트 되며, 각 단계 시작 시 상태를 읽어들임


### Checkpoints