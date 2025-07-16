# LangGraph를 활용한 RAG 고도화

## Adaptive RAG

- 사용자 질의를 분석하고, 능정적/자기 수정적 RAG를 결합한 전략임


### No Retrieval

### Single-shot RAG

### Iterative RAG

- 단계적 검색과 응답 생성을 반복하는 전략임
- 각 반복마다 이전 결과를 분석하여 추가 검색 방향을 결정함
- 복잡하거나 다단계 질문에 적합함

## SelfRAG

[논문](https://arxiv.org/pdf/2310.11511)

- 검색 결정
- 검색된 문서 관련성 평가
- 생성된 답변의 환각 평가
- 생성된 답변의 유용성 평가

### Retrieval Grader

- 검색 평가자는 키워드 관련성과 의미적 관련성을 기준으로 결과를 평가함


### Answer Generator

- 답변 생성 시 문맥 내 정보만 사용했는지 평가함
- 답변은 직접 관련 정보만 포함하여 간결하게 작성하도록 유도함

### Hallucination Grader

- 사실 기반 답변을 평가하는 전문가로 역할을 정의함

### Answer Grader

- 답변 평가는 정보 포함 여부를 기준으로 이진 분류함

### Question Re-writer

- 질문을 명확성과 간결성 중심으로 개선하도록 유도함



## CRAG