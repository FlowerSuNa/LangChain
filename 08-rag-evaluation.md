# RAG 성능 평가 개요

- RAG는 외부 지식 검색과 LLM 결합으로 응답 품질을 향상 시키는 기법임
- LLM-as-judge 방식으로 사실성, 관련성, 충실도, 유용성을 평가할 수 있음
- 체계적인 A/B 테스트로 각 컴포넌트별 성능을 비교하고 영향도 분석으로 최적의 구성을 도출해야 함
- 오프라인(참조답변 기반), 온라인(실시간), 페어와이즈(비교) 평가 방법론이 있음

## 평가 대상

**검색 (Retrieval)**
- 관련 문서와 쿼리 간의 연관성을 통해 검색된 문서가 쿼리의 정보 요구를 얼마나 잘 충족하는지 평가해야 함
- 관련 문서와 후보 문서 간의 정확성을 통해 시스템이 적절한 문서를 식별하는 능력을 정량적으로 측정해야 함

**생성 (Generation)**
- 응답과 쿼리의 연관성(Relacance), 응답과 관련 문서 간의 충실도(Faithfulness), 응답과 샘플 응답 간의 정확성(Correctness)를 평가해야 함

**추가 고려 사항**
- 핵심 성능 지표의 응답속도 (Latency), 검색 다양성 (Diversity), 잡음 내구성 (Noise Robustness)를 고려해야 함
- 불충분 정보 거부(negative rejection), 오정보 식별(counterfactual robustness) 등 안전성을 평가해야 함
- 가독성(readability), 유해성(toxicity), 복잡성(perplexity) 등 사용자 경험을 고려해야 함

[Reference](https://huggingface.co/learn/cookbook/en/rag_evaluation)

## Ragas

- 평가 테스트셋 구축

### Persona

[문서](https://docs.ragas.io/en/stable/howtos/customizations/testgenerator/_persona_generator/?h=persona)

### Test Dataset


## 평가 지표 (Evaluation Metric)

[Ragas 문서](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)

- RAG 성능평가를 위해 정보를 검색하는 Retrieval의 성능을 평가할 수 있음
- 지표로는 HitRate, MRR, NDCG 등이 있음

### 1\) 검색 평가 (Retrieval Evaluation)

- Non-Rank Based Metrics : Accuracy, Precision, Recall@k 등을 통해 관련성의 이진적 평가를 수행함
- Rank Based Metrics : MMR(Mean Reciprocal Rank), MAP(Mean Average Precision)을 통해 검색 결과의 순위를 고려한 평가를 수행함
- RAG 특화 지표 : 기존 검색 평가 방식의 한계를 보완하는 LLM-as-judge 방식을 도입할 수 있음
- 포괄적 평가 : 정확도, 관련성, 다양성, 강건성을 통합적으로 측정함

### 2\) 생성 평가 (Generation Evaluation)

- 배치작업을 수행하여 효율적으로 할수 있음 
- 평가는 고성능 모델을 사용해야 함

- 전통적 평가 : ROUGE(요약), BLEU(번역), BertScore(의미 유사도) 등의 지표를 활용함
- LLM 기반 평가 : 응집성, 관련성, 유창성을 종합적으로 판단하는 새로운 접근법을 도입함 (전통적인 참조 비교가 어려운 상황에서 유용함)
- 다차원 평가 : 품질, 일관성, 사실성, 가독성, 사용자 만족도를 포괄적으로 측정함
- 상세 프롬프트와 사용자 선호도 기준으로 생성 텍스트의 품질을 평가함


---

# Retrieval Metrics

## Hit Rate

- 정보 검색 시스템의 성능을 측정하는 기본적인 평가 지표로, 검색 결과에 정답 문서가 모두 포함되어 있다면 1, 아니면 0으로 평가함 (이진법 평가, 정의하기 나름..)
- 계산 방식이 단순하여 직관적으로 이해하기 쉽고, 전체 검색 시스템의 기본적인 성능을 빠르게 파악할 수 있다는 장점이 있음
- 최종 값은 모든 검색 쿼리에 대한 평균값으로 산출되며, 0과 1 사이의 값을 가짐
- 1에 가까울수록 검색 시스템의 성능이 우수함을 의미함
- 정보 검색 시스템의 기본적인 성능을 평가하는 데 널리 사용되는 지표임
- 순서가 중요하지 않음 -> Non-Rank 
- RAG 시스템에서는 중요한 지표라고 생각됨

- 검색기의 k 값 (검색할 문서 수)가 지표 계산에 중요함
- 만약 검색기의 성능이 더이상 나아지지 않는다면 굳이 k를 증가시킬 필요가 없음
- 평가 지표를 계산하는 모듈은 요즘 랭스미스로 넘김
- 예전에 쓰던 평가 지표가 있기는 함
- rag를 평가할만큼의 지표는 추가적으로 만들어서 써야함
- autoRAG - RAG 평가를 자동화해주는 도구, 비용 관리가 잘 안됨.. 편하긴 하나... 오토엠엘 느낌..
[링크](https://docs.auto-rag.com/evaluate_metrics/retrieval.html)
[링크](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/autorag/autorag/evaluation/metric/retrieval.py)



## MRR (Mean Reciprocal Rank)

- MMR은 검색 결과에서 첫 번째로 등장하는 관련 문서의 순위를 기반으로 성능을 평가하는 지표임
- 특히 사용자가 원하는 정보를 얼마나 빨리 찾을 수 있는지 측정함
- 사용자 경험(UX) 관점에서 중요한 의미를 가지며, 검색 시스템의 실질적인 유용성을 평가하는 데 효과적임
- 각 검색 쿼리별로 첫 번째 관련 문서의 순위의 역수를 구한 후, 전체 쿼리에 대한 평균을 계산함 (정답이 3번째에 있다면 1/3)

## mAP@k (Mean Average Precision at k)

- RAG 시스템에서 중요한 지표라고 생각됨
- 검색 결과의 품질을 평가하는 고급 지표로, 상위k개의 검색 결과 내에서 관련 문서들의 순위와 정확도를 종합적으로 평가함
- 단순히 관련 문서의 존재 여부나 첫 등장 순위 뿐만 아니라, 상위k개 걀과 내에서 관련 문서들의 전반적인 분포와 순서까지 고려하여 더욱 세밀한 성능 평가가 가능함
- 검색 서비스에서 사용자에게 가장 중요한 상위 검색 결과의 품질에 초점을 맞추고 있어, 실용적인 관점에서 시스템의 성능을 평가하는 데 매우 효과적임
- 정보 검색 시스템의 실제 사용성과 가장 밀접하게 연관되어 있어, 현대 검색 엔진 개발에서 널리 사용되고 있음
[링크](https://lsjsj92.tistory.com/663)

## NDCG (Normalized Discounted Cumulative Gain)

[링크](https://www.pinecone.io/learn/offline-evaluation/)
- 검색 및 추천 시스템의 순위 품질을 평가하는 고급 지표로, 단순히 관련성 여부 뿐만 아니라 결과의 순서까지 고려하여 시스템의 성능을 평가함
- 검색 결과의 위치에 따라 가중치를 다르게 부여함
- 상위에 있는 관련 문서에는 더 높은 가중치를, 하위에 있는 문서에는 더 낮은 가중치를 부여함
- NDCG는 이상적인 순위와 비교하여 정규화된 점수를 제공하므로, 다른 검색 결과나 시스템 간의 비교가 용이함
- 특히 사용자의 실제 만족도와 밀접한 관련이 있어, 현대의 검색 및 추천 시스템 평가에서 매우 중요하게 사용됨

---

## 쿼리 확장 (Query Expansion)

- 검색 성능 향상을 위한 기법 중 하나임
- 어떻게 쿼리를 작성할 것인지
- 쿼리의 의도와 문서의 표현이 매칭이 잘 될지, 쿼리는 글자수가 문서보다 적음, 쿼리의 특정 키워드에 매몰될 수도 있음 
- 검색이 잘 되도록 쿼리를 재작성해 보는 기법들

### Query Reformulation

- [논문](https://arxiv.org/pdf/2305.14283)
- LLM을 사용하여 원래 질문을 다른 형태로 재작성하는 방식임
- 동의어 확장, 질문 명확화, 관련 키워드 추가 등 다양한 방식으로 쿼리를 변형함
- 검색의 다양성과 정확도를 향상시키는 특징이 있음
- 모호한 질문을 **명확하게 구체화**하여 검색 정확도 향상함 (temperature 낮게 설정?)
- 하나의 질문에 대해 다양한 변형 쿼리를 생성하여 검색 커버리지를 확대할 수 있음

### Multi Query

- Retriever에 지정된 LLM을 활용하여 원본 쿼리를 확장하는 방법임
- 하나의 질문에 대해 **다양한 관점과 표현**으로 여러 개의 쿼리를 자동 생성함
- LLM의 생성 능력을 활용하여 검색의 다양성과 포괄성을 향상시키는 특징이 있음, 검색 범위를 자연스럽게 확장함
- 검색의 다양성과 포괄성이 향상되어 관련 문서 검색 확률이 증가함

### Decomposition

- [논문](https://arxiv.org/pdf/2205.10625)
- 복잡한 질문을 여러 개의 단순한 하위 질문으로 분해하는 LEAST-TO-MOST PROMPTING 전략을 사용함
- 단계별 분해 전략을 통해 복잡한 질문을 작은 단위로 나누어 처리함
- 각 하위 질문마다 독립적인 검색 프로세스를 진행하여 정확도를 향상시킴
- 복잡한 질문을 단순화하여 검색의 정확도를 높이는 특징이 있음

### Step-back Prompting

- [논문](https://arxiv.org/pdf/2310.06117)
- 큰 그림을 보자? Decomposition 기법과 살짝 반대의 방향성?
- 주어진 구체적인 질문에서 한 걸음 물러나 더 일반적인 개념이나 배경을 먼저 검색함
- 더 넓은 맥락에서 점차 구체적인 답변으로 좁혀가는 방식을 사용함
- 포괄적 접근법을 활용하여 복잡한 질문에 대한 이해도를 높임
- 일반적 맥락에서 시작하여 구체적 해답을 찾아가는 체계적 접근 방식임
- 복잡한 질문에 대해 더 포괄적이고 정확한 답변을 제공하는 특징이 있음

### HyDE (Hypothetical Document Embedding)

- [논문](https://arxiv.org/pdf/2212.10496)
- 주어진 질문에 대해 가상의 이상적인 답변 문서를 LLM으로 생성함
- 생성된 가상 문서를 임베딩하여 이를 기반으로 실제 문서를 검색하는 방식임
- 주어진 질문과 문서간에는 차이가 있다는 가정에서 시작하는 아이디어임
- 분량을 문서 규모에 맞춤
- 사실이 아니거나 디테일 함은 다르겠지만, 비슷한 맥락과 스토리 라인을 만들어 낼 수 있다고 가정함
- 질문의 맥락을 더 잘 반영한 검색이 가능한 특징이 있음
- 맥락 기반 검색 방식으로 질문의 의도를 더 정확하게 반영함

---

# 검색 후처리

## Re-rank (재순위화)

- 검색 순서를 바꿈
- 관련성 높은 문서를 최대한 앞으로 옮기는 것이 목적
- 프롬프트가 길어졌을때 앞쪽에 위치하는 것이 좋음
- 웬만하면 써주는 것이 좋음
- 관련성 없는 문서를 제거해 주자.. 불필요한 정보는 최소화하고 의미 있는 정보만 프롬프트에 넣어주자..
- 문서들을 어떻게 프롬프트에 넣을 것인지 중요함

[논문](https://arxiv.org/pdf/2407.21059)

- 검색 결과를 재분석하여 최적의 순서로 정렬하는 고도화된 기술임
- 먼저 기본 검색 알고리즘으로 관련 문서들을 찾은 후, 더 정교한 기준으로 이들을 재평가하여 최종 순위를 결정함
- 사용자의 검색 의도에 맞는 정확도 향상을 통해 검색 품질을 개선함
- 검색 결과의 품질을 높이기 위한 체계적인 최적화 방법론임

### Cross Encoder Reranker

- [velog](https://velog.io/@xuio/Cross-Encoder%EC%99%80-Bi-Encoder-feat.-SentenceBERT)
- 두 개의 문장 또는 문서 간 관계를 분석하는 기법임
- 통합 인코딩 방식으로 검색 쿼리와 검색된 문서 간 유사도를 더 정확하게 계산함
- 검색은 엄청 정교하지만, 문서가 많으면 비효율적임

- [LangChain 문서](https://python.langchain.com/docs/integrations/retrievers/cohere-reranker/#related)


### LLM Reranker

- LLM를 활용하여 검색 결과의 재순위화를 수행함
- 쿼리와 문서 간의 관련성 분석을 통해 최적의 순서를 도출함
- 비용이 많이 비쌈

## Contextual Compression (맥락적 압축)

- 검색된 문서를 쿼리 관련 정보만을 선별적으로 추출함
- 효율적인 처리를 통해 LLM 비용 절감과 응답 품질 향상을 달성함

### LLMChainFilter

- LLM 기반 필터링으로 검색된 문서의 포함 여부를 결정함
- 선택적 필터링을 통해 관련성 높은 문서만을 최종 반환함
- 순서는 고려되지 않음
- 문서 원본을 보존하면서 관련성 기반의 스마트한 선별을 수행하는 방식

### LLMChainExtractor

- LLM 기반 추출로 문서에서 쿼리 관련 핵심 내용만 선별함
- 문서에서 관련성 있는 문장(정보)만 추출함
- 순차적 처리 방식으로 각 문서를 검토하여 관련 정보를 추출함
- 맞춤형 요약을 통해 쿼리에 최적화된 압축 결과를 생성함
- 쿼리 맥락에 따른 선별적 정보 추출로 효율적인 문서 압축을 실현함

### EmbeddingsFilter

- 임베딩 기반 필터링으로 문서와 쿼리 간 유사도를 계산함
- LLM 미사용 방식으로 빠른 처리 속도와 비용 효율성을 확보함
- LLM 호출보다 저렴하고 빠른 옵션임
- 유사도 기준 선별을 통해 높은 문서만을 효과적으로 추출함
- 경제적이고 신속한 임베딩 기반의 문서 필터링 기법임
- 진짜 관련성 낮은 문서를 제거하려고 쓸때 괜찮을 듯.. 정교함은 위에 방법보다 떨어질수 있음

### DocumentCompressorPiepline

- 파이프라인 구조로 여러 압축기를 순차적으로 연결하여 처리함
- 복합 변환 기능으로 문서 분할 및 중복 제거 등 다양한 처리가 가능함
- 유연한 확장성을 통해 `BaseDocumentTransformers` 추가로 기능을 확장함
- 가중 압축기를 연계하여 포괄적이고 효과적인 문서 처리를 구현하는 방식임

---

- human feedback데이터가 매우 귀함
- 요즘은 챗봇은 rlhf 기법을 도입함

---

[링크](https://medium.com/@richardhightower/is-rag-dead-anthropic-says-no-290acc7bd808)




Contextual Retrieval : [Anrheopic](https://www.anthropic.com/news/contextual-retrieval)


- 고객의 피드백 데이터 확보가 LLM 시스템 성능의 키임
- 후발 주자가 못따라가는 이유 중 하나이기도 함 (OpenAI, Claud, Gemini 따라가기 힘듬...)
- 다만 특수 분야에 대한 것을 부족하므로 그 부분은 시장성이 있어보임..


[평가에 대한..](https://github.com/huggingface/evaluation-guidebook)

[](https://docs.smith.langchain.com/evaluation/tutorials/agents?_gl=1*1dsm7vj*_gcl_au*NjY4MDM2NzUwLjE3NDc5MDIyNjM.*_ga*NjE1MTEyNDE5LjE3NTE5MzQ0MjU.*_ga_47WX3HKKY2*czE3NTI4OTQwMzYkbzMzJGcxJHQxNzUyODk0MTM5JGo2MCRsMCRoMA..#trajectory)
[](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag_local/?h=trajectory#response)
[](https://arxiv.org/abs/2401.16745)
[](https://huggingface.co/papers/2401.16745)