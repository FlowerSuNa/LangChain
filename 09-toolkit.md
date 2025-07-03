# Toolkit

- 도구들의 모임

[LangChain 문서](https://python.langchain.com/docs/integrations/tools/)


## StructuredTool

- 도구의 실행 방식과 응답 처리를 상세하게 커스터마이징 가능함
- 개발자가 원하는 특정 기능을 도구에 쉽게 추가할 수 있음

## Runnable 도구 변환

- `as_tool` 메소드를 이용하여 `Runnable` 도구로 변환하여 복잡한 체인을 하나의 단위로 관리 가능함
- 도구화를 통해 체인에 대한 명확한 인터페이스를 제공함
- 변환된 도구는 다른 프로젝트나 컴포넌트에서 쉽게 재사용이 가능함
- 체인의 실행 방식을 표준화하여 일관된 사용 경험을 제공함
