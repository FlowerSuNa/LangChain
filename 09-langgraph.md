# LangGraph

[](https://langchain-ai.github.io/langgraph/concepts/low_level/#multiple-schemas)

[LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/)

[Functional API](https://langchain-ai.github.io/langgraph/concepts/functional_api/)

## StateGraph
- 상태 기반의 그래프 구조를 사용하여 대화 흐름을 제계적으로 관리함


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