# ReAct (Reasoning and Acting)

- 가장 기본 에이전트 유형임
- 모델이 상황을 추론(Reasoning)한 후, 필요한 경우 적절한 도구를 호출(Acting)함
- 도구 실행 결과는 다시 모델에 입력되어, 다음 행동을 결정하는 데 사용됨
- 이 과정을 반복하며, 또 다른 도구를 호출하거나 충분한 정보가 모이면 직접 최종 응답을 생성함
- 복잡한 판단 흐름과 다단계 툴 체인이 필요한 상황에 적합함

---

## Tool Execution

### ToolNode

- 사전에 정의된 도구를 모델이 호출할 수 있도록 구성한 LangGraph 노드 유형임
- 다수의 도구 요청이 있을 경우 병렬 실행함
- 각 도구 실행 결과는 `ToolMessage`로 래핑되어 상태에 추가됨

### create_react_agent

- `ToolNode`와 함께 사용하는 에이전트 생성 함수임
- 모델이 도구를 스스로 선택하고 호출할 수 있도록 ReAct 기반 프롬프트를 구성함
- 내부적으로 도구의 설명, 사용 조건, 출력 포맷 등을 기반으로 호출 순서를 판단함

---

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