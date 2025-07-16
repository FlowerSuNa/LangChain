# ReAct Agent (Reasoning and Acting)

- 가장 일반적인 에이전트로, 모델이 특정 도구를 호출하고 출력을 다시 모델에 전달하는 과정을 거침
- 모델이 도구 출력을 바탕으로 다음 행동을 결정함
- 반복되는 과정에서 또 다른 도구를 호출하거나 직접 응답을 생성함

---

## Tool Node

- 사전에 정의된 도구를 모델이 호출하도록 하는 LangGraph 구성 요소임

## create_react_agent


## Human-in-the-Loop (사용자 개입)
- HITL은 AI 시스템에 인간의 판단과 전문성을 통합하는 시스템임
- `Breakpoints`로 특정 단계에서 실행 중지가 가능함
    - `Breakpoints`는 체크포인트 기능 기반으로 작동하는 시스템임
    - 각 노드 실행 후 그래프의 상태를 스레드에 저장하여 나중에도 접근이 가능함
    - 그래프 실행을 특정 지점에서 일시 중지하고 사용자 승인 후 재개 가능하도록 구현 가능함
- 사용자의 입력이나 승인을 기다리는 패턴으로 작동함
- 시스템 결정에 대한 인간의 통제와 검증을 보장함

---

## Reference

- 🔗 [Workflows and Agents](https://langchain-ai.github.io/langgraph/tutorials/workflows/)