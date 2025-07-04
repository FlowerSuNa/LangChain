# Agent

- LLM을 의사결정 엔진으로 사용하여 작업을 수행하는 시스템임
- 모델은 입력된 데이터를 분석하여 맥락에 맞는 의사결정을 수행함
- 시스템은 사용자의 요청을 이해하고 적절한 해결책을 제시함
- 복잡한 작업을 자동화하여 업무 효율성을 높일 수 있음



[Git 커뮤니티?](https://github.com/modelcontextprotocol/servers?tab=readme-ov-file#%EF%B8%8F-official-integrations)

[Smithery](https://smithery.ai/)

[MCP](https://modelcontextprotocol.io/introduction)

---

## Tool Creation

- `@tool` 데코레이터를 사용하여 함수에 스키마 정보를 추가할 수 있음

## Tool Binding

- 모델-도구 연결로 입력 스키마를 자동 인식함
- LLM이 도구 호출을 할지 말지 선택함 (도구 호출 전임)


---

## Tool Calling

- LLM이 외부 시스템과 상호작용하기 위한 함수 호출 매커니즘임
- LLM은 정의된 도구나 함수를 통해 **외부 시스템**과 통신하고 작업을 수행함
- 즉, 모델이 시스템과 직접 상호작용할 수 있게 하는 기능임
- 구조화된 출력을 통해 API나 데이터베이스와 같은 시스템 요구사항을 충족함
- 스키마 기반 응답으로 시스템간 효율적 통신 가능함

```python
result.tool_calls
```

## Tool Execution

---

## AgentExecutor

- LangChain의 기본 에이전트 실행 시스템임
- 에이전트의 계획-실행-관찰 사이클을 자동으로 관리함
- 에이전트의 행동을 모니터링하고 결과를 반환함

1. 에이전트의 기본 행동과 응답 방식을 정의하는 프롬프트 템플릿을 작성함
2. LLM과 도구를 통합하여 복잡한 작업을 수행하는 에이전트를 생성함 (도구 실행 결과를 분석하여 최종 응답을 생성하는 워크플로우를 구현함)
3. 에이전트의 작업 흐름을 관리하고 결과를 처리하는 `AgentExecutor` 컴포넌트를 활용함
    - 사용자 입력부터 최종 출력까지의 전체 프로세스를 조율하고 제어함
    - 에러 처리, 로깅, 결과 포매팅 등 시스템 운영에 필요한 기능을 제공함


---

## QA 체인

- QA 페인을 구성하여 질의응답 시스템을 체계화할 수 있음
- 도구 사용을 통해 사용자가 원하는 정보를 컨텍스트로 활용 가능함


---

## MCP

(https://modelcontextprotocol.io/introduction)
(https://github.com/modelcontextprotocol/python-sdk)
