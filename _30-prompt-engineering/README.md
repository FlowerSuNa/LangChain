# 프롬프트 엔지니어링 (Prompt Engineering)

## 개념
- 프롬프트 엔지니어링은 LLM에게 명확하고 목적에 부합하는 지시를 제공함으로써, 원하는 결과를 안정적으로 얻어내는 기술임
- 단순 질의에서 고도화된 작업 지시까지, 프롬프트의 설계 방식에 따라 출력 품질이 크게 달라짐
- 잘 설계된 프롬프트 템플릿은 재사용 가능하며, 다양한 입력에 대해 일관된 응답 품질과 정확도를 확보할 수 있음

## 설계 원칙

**1) 명확성 (Clarity)**
- 프롬프트는 불필요한 수식어나 중의적 표현을 배제하고, 핵심 요구사항에 집중하여 작성해야 함
- LLM이 혼동 없이 작업을 수행할 수 있도록, 의도한 결과물에 대해 구체적이고 정확한 지시를 제공해야 함
- 예측 가능한 결과를 얻기 위해선, 문맥이 간결하면서도 목적이 분명한 문장으로 구성되어야 함

**2\) 맥락성 (Context)**
- 배경 정보, 목적, 대상 환경 등을 명확히 포함한 문맥 정보는 LLM이 사용자의 의도를 이해하고, 보다 정밀하게 목적에 부합하는 응답을 생성하는 데 도움을 줌
- 멀티턴 대화, 문서 요약, 코드 생성 등 복잡한 작업일수록 적절한 맥락 제공은 출력 품질과 정확도를 크게 향상 시킴

**3\) 구조화 (Structure)**
- 프롬프트는 일관된 입력-출력 형식을 유도할 수 있도록 구조화되어야 함
- 명확한 포맷 예시, 출력 템플릿, JSON 형식 등의 구조화를 적용하면 LLM의 응답 안정성을 높일 수 있음
- 구조화된 출력은 후처리 자동화, 오류 감지, 시스템 간 연동 등에 유리하며, 프롬프트의 재사용성과 유지보수 효율성도 함께 향상됨