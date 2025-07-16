
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