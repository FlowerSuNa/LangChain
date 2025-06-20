# RAG 답변 성능 평가


- LangSmith, LangFuse 사용
- [Guardrails](https://github.com/guardrails-ai/guardrails?tab=readme-ov-file) 사용
- [LangSmith 문서](https://docs.smith.langchain.com/evaluation/concepts#evaluators)


### 휴리스틱 평가

- 특정 규칙에 기반한 결정론적 함수로 작동하며, 명확한 기준에 따라 판단을 수행함
- 주로 단순 검증에 활용되며, 챗봇 응답의 공백 여부, 생성된 코드의 컴파일 가능함 등을 확인함
- 평가 기준이 명확하고 객관적이어서 정확한 분류나 검증이 필요한 경우에 효과적임
- 복잡한 상황보다는 명확한 규칙이 존재하는 간단한 검증 작업에 적합함

### 정량 평가 지표

- ROUGE와 BLEU는 텍스트 생성 품질을 평가하는 대표적인 정량 평가지표임
- 생성된 텍스트와 참조 텍스트 간의 단어 중첩도를 계산하여 품질을 수치화함
- 대규모 자동화 평가가 필요한 경우 효율적이며, 객관적인 비교가 가능한 장점이 있음
- 하지만, 문맥이나 의미의 유사성은 완벽하게 포착하지 못하는 한계점이 존재함
- 일반적인 챗봇에 쓰기는 모호하지만, RAG 기반이라면 쓸만하다고 생각됨

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**

- 생성된 요약문의 품질을 평가함

**BLEU (Billngual Evaluation Understudy)**

- 기계 번역의 품질을 평가하는 대표적인 지표로, 생성된 번역문과 참조 번역문 간의 n-gram 정확도를 계산함
- 0에서 1사이의 값을 가지며, 1에 가까울수록 번역 품질이 좋음을 의미함

- BP(Brevity Penalty) : 생성문이 참조문보다 짧을 경우 패널티를 부함함
    - min(1, exp(1-참조문길이/생성문길이))
- 일반적으로 BLEU-1부터 BLEU-4까지의 기하평균을 사용함

### 문자열 및 임베딩 거리 평가

**String Distance**
- 레벤슈타인 거리를 사용해 두 문자열이 얼마나 다른지 측정함
- 한 문자열을 다른 문자열로 변환하는 데 필요한 최소 편집 횟수를 계산함
- 점수는 0에서 1사이로 정규화되며, 0애 가까울수록 문자열이 유사함을 의미함
- `rapidfuzz` 라이브러리를 통해 효율적인 계산이 가능하며, 대규모 평가에 적합함

**Embedding Distance**
- 텍스트를 고차원 벡터로 변환하여 의미적 유사도를 계산함
- 코사인 거리, 유클리디안 거리, 맨헤튼 거리 등 다양한 거리 메트릭을 선택적으로 활용할 수 있음
- OpenAI나 HuggingFace 등 다양한 임베딩 제공자를 설정할 수 있음
- 단순 문자열 비교와 달리 문맥적 의미를 고려한 평가가 가능함

### LLM-as-Judge

[LangChain](https://python.langchain.com/api_reference/_modules/langchain/evaluation/criteria/eval_chain.html#Criteria)

- LLM을 평가자로 활용하여 텍스트 출력의 품질을 전문적으로 판단함
- 평가 기준을 프롬프트 형태로 명확히 정의하여 일관됭 평가를 수행함
- 다양한 품질 측면(정확성, 관련성, 일관성 등)을 종합적으로 평가함

**Reference-free 평가**
- 참조 답변 없이 독립적으로 출력 품질을 평가하는 방식임
- 객관적인 품질 기준을 바탕으로 평가가 진행되어 참조 데이터 구축에 대한 부담이 없음
- 불필요한 반복이나 장황함없이 핵심 내용을 전달하는지 (Conciseness), 논리적 흐름과 구조가 명확한지 (Coherence), 실질적인 도움이 되는 정도가 어느정도 인지 (Helpfulness), 해로운 내용을 포함하는 지 (Harmfulness/Maliciousness/Misogyny, Criminaliy) 등을 평가할 수 있음

**Reference-based 평가**
- 참조 답변(Ground Truth)과 출력을 직접 비교하는 방식임
- 참조 답변을 기준으로 출력의 정확성과 일관성을 객관적으로 평가함

```python
from langchain.evaluation import load_evaluator

# labeled criteria 평가자 정의
labeled_crieria_evaluator = load_evaluator(
    evaluator="labeled_criteria", 
    criteria="correctness",
    llm=ChatOpenAI(model="gpt-4.1-mini", temperature=0.0),
)

# 평가 수행
labeled_crieria_eval_result = labeled_crieria_evaluator.evaluate_strings(
    input=question,               # 평가에 고려할 내용: 질문
    prediction=answer,            # 평가 대상: LLM 모델의 예측
    reference=ground_truth,       # 평가 기준: 정답
)
```

```python
from langchain.evaluation import load_evaluator

# labeled score string 평가자 정의
labeled_score_string_evaluator = load_evaluator(
    evaluator="labeled_score_string", 
    criteria="relevance",
    normalize_by=10,
    llm=ChatOpenAI(model="gpt-4.1", temperature=0.0),
)

# 평가 수행
labeled_score_string_eval_result = labeled_score_string_evaluator.evaluate_strings(
    input=question,               # 평가에 고려할 내용: 질문
    prediction=answer,            # 평가 대상: LLM 모델의 예측
    reference=ground_truth,       # 평가 기준: 정답
)
```

```python
from langchain.evaluation import load_evaluator

# custom labeled criteria 평가자 정의
labeled_crieria_evaluator = load_evaluator(
    evaluator="labeled_criteria", 
    criteria={
        "correctness": "Given the provided reference, is the answer correct?",
        "relevance": "Does the answer appropriately address the question?",
    },
    llm=ChatOpenAI(model="gpt-4.1-mini", temperature=0.0),
)

# 평가 수행
labeled_crieria_eval_result = labeled_crieria_evaluator.evaluate_strings(
    input=question,               # 평가에 고려할 내용: 질문
    prediction=answer,            # 평가 대상: LLM 모델의 예측
    reference=ground_truth,       # 평가 기준: 정답
)
```

```python
from langchain_core.prompts import PromptTemplate
from langchain.evaluation import load_evaluator

# 사용자 정의 프롬프트 템플릿 생성
template = """Respond Y or N based on how well the following response follows the specified rubric. Grade only based on the rubric and expected response:

Grading Rubric: {criteria}
Expected Response: {reference}

DATA:
---------
Question: {input}
Response: {output}
---------
Write out your explanation for each criterion in 한국어, then respond with Y or N on a new line."""

prompt = PromptTemplate.from_template(template)

# custom labeled criteria 평가자 정의
labeled_crieria_evaluator = load_evaluator(
    evaluator="labeled_criteria", 
    criteria={
        "correctness": "Give the provided reference, is the answer correct?",
        "relevance": "Does the answer appropriately address the question?",    
    },
    llm=ChatOpenAI(model="gpt-4.1-mini", temperature=0.0),
    prompt=prompt,                # 사용자 정의 프롬프트 사용
)

# 평가 수행
labeled_crieria_eval_result = labeled_crieria_evaluator.evaluate_strings(
    input=question,               # 평가에 고려할 내용: 질문
    prediction=answer,            # 평가 대상: LLM 모델의 예측
    reference=ground_truth,       # 평가 기준: 정답
)
```

```python
from langchain.evaluation import load_evaluator
from langchain_google_genai import ChatGoogleGenerativeAI

# QA 평가기 로드
qa_evaluator = load_evaluator(
    "qa",        # 평가방법 지정: qa, context_qa, cot_qa 
    llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0),  # LLM 모델 지정
)

# 평가 수행
result = qa_evaluator.evaluate_strings(
    prediction="서울은 대한민국의 수도이며 인구는 약 960만 명이다",
    input="서울의 인구는 얼마인가?",
    reference="서울의 인구는 약 960만 명이다"
)
```