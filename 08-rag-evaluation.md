# RAG 성능 평가 개요

- 

[Reference](https://huggingface.co/learn/cookbook/en/rag_evaluation)

## Ragas

### Persona

[문서](https://docs.ragas.io/en/stable/howtos/customizations/testgenerator/_persona_generator/?h=persona)

### Test Dataset


## 평가 지표 (Evaluation Metric)

[Ragas 문서](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)

### 1\) 검색 평가 (Retrieval Evaluation)

- Non-Rank Based Metrics : Accuracy, Precision, Recall@k 등을 통해 관련성의 이진적 평가를 수행함
- Rank Based Metrics : MMR(Mean Reciprocal Rank), MAP(Mean Average Precision)을 통해 검색 결과의 순위를 고려한 평가를 수행함

### 2\) 생성 평가 (Generation Evaluation)

