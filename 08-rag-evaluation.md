# RAG 성능 평가 개요

- 

[Reference](https://huggingface.co/learn/cookbook/en/rag_evaluation)

## Ragas

- 평가 테스트셋 생성

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