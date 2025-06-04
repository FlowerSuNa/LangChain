# Embedding

- 텍스트를 벡터 형태로 변환하는 작업임
- 자연어를 수치화하여 모델이 이해할 수 있도록 함

## 1. Word Embedding

- 유사한 의미를 갖는 단어들이 벡터 공간 상에서 가까운 위치에 매핑되도록 학습함
- 단어 간 의미 관계를 수치적으로 반영할 수 있음

*사전 단계) Tokenization*

```python
# 허깅페이스 트랜스포머 라이브러리에서 토크나이저 로드
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('beomi/kcbert-base')

# 토큰화
texts = ["자연어처리를 공부합니다", "자연어를 배웁니다", "자연어 처리방법을 사용합니다"]
tokenized_texts = []
for text in texts:
    tokenized_texts.append(tokenizer.tokenize(text))

combined_texts = [" ".join(tokens) for tokens in tokenized_texts]
```

```markdown
# combined_texts
['자연 ##어 ##처리 ##를 공부 ##합니다',
 '자연 ##어를 배 ##웁 ##니다',
 '자연 ##어 처리 ##방법 ##을 사용 ##합니다']
```

**1) Bag of Words (BoW)**

- 단어의 출현 빈도를 벡터로 표현하는 기법임
- 구현이 간단하고 직관적이지만, 단어의 순서와 문맥 정보를 반영하지 못하며 희소 행렬이 생성됨

```python
# BoW 벡터화
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()  
bow_matrix = vectorizer.fit_transform(combined_texts)

bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
bow_df
```

|   | 공부 | 니다 | 방법 | 사용 | 어를 | 자연 | 처리 | 합니다 |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| 1 | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 |
| 2 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 1 |

**2) TF-IDF (Term Frequency-Inverse Document Frequency)**

- 단어의 중요도를 반영하여 벡터를 구성하는 기법임
- Term Frequency (TF): 특정 문서 내 단어의 등장 빈도
- Inverse Document Frequency (IDF): 전체 문서에서 해당 단어가 얼마나 희귀한지를 반영
- 자주 등장하지만 여러 문서에 공통적으로 나타나는 단어에는 낮은 가중치를 부여하고, 드물게 등장하지만 특정 문서에서 중요한 단어에는 높은 가중치를 부여함

```python
# Tfidf 벡터화
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(combined_texts)

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
tfidf_df
```

|   | 공부 | 니다 | 방법 | 사용 | 어를 | 자연 | 처리 | 합니다 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.631745 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.373119 | 0.480458 | 0.480458 |
| 1 | 0.000000 | 0.652491 | 0.000000 | 0.000000 | 0.652491 | 0.385372 | 0.000000 | 0.000000 |
| 2 | 0.000000 | 0.000000 | 0.534093 | 0.534093 | 0.000000 | 0.315444 | 0.406192 | 0.406192 |

**3) Word2Vec**

- 단어 간 의미적 유사성을 벡터 공간에 반영하도록 학습하는 딥러닝 기반 임베딩 기법임
- 주어진 문맥에서 단어를 예측하거나, 특정 단어가 주어졌을 때 주변 단어를 예측하는 방식(CBOW, Skip-gram)을 통해 단어 의미를 학습함
- 의미적으로 유사한 단어들이 벡터 공간상에서 가깝게 위치함

```python
# Word2Vec 벡터화
from gensim.models import Word2Vec #type: ignore

word2vec_model = Word2Vec(
    sentences=tokenized_texts,  # 학습용 문장 데이터
    vector_size=5,  # 임베딩 된 단어벡터의 차원
    window=2,       # 주변 단어 개수 (context window: 좌우 2개)
    min_count=1,    # 최소 등장 횟수 (빈도가 낮은 단어는 제거)
    workers=4,
    sg=1            # 0: CBOW, 1: Skip-gram
)

vocab = word2vec_model.wv.index_to_key
word_vectors = [word2vec_model.wv[word] for word in vocab]
word2vec_df = pd.DataFrame(word_vectors, index=vocab)
word2vec_df
```

| 단어     | 0        | 1        | 2        | 3        | 4        |
|---|---:|---:|---:|---:|---:|
| 자연     | -0.010725 | 0.004729  | 0.102067  | 0.180185  | -0.186059 |
| ##합니다 | -0.142336 | 0.129177  | 0.179460  | -0.100309 | -0.075267 |
| ##어     | 0.147610  | -0.030669 | -0.090732 | 0.131081  | -0.097203 |
| 사용     | -0.036320 | 0.057532  | 0.019837  | -0.165704 | -0.188976 |
| ##을     | 0.146235  | 0.101405  | 0.135154  | 0.015257  | 0.127018  |
| ##방법   | -0.068107 | -0.018928 | 0.115371  | -0.150433 | -0.078722 |
| 처리     | -0.150232 | -0.018601 | 0.190762  | -0.146383 | -0.046675 |
| ##니다   | -0.038755 | 0.161549  | -0.118618 | 0.000903  | -0.095075 |
| ##웁     | -0.192071 | 0.100146  | -0.175192 | -0.087837 | -0.000702 |
| 배       | -0.005962 | -0.153220 | 0.192279  | 0.099634  | 0.184681  |
| ##어를   | -0.163158 | 0.089916  | -0.082742 | 0.016491  | 0.169972  |
| 공부     | -0.089244 | 0.090350  | -0.135739 | -0.070970 | 0.187970  |
| ##를     | -0.031553 | 0.006427  | -0.082813 | -0.153654 | -0.030160 |
| ##처리   | 0.049396  | -0.017761 | 0.110673  | -0.054860 | 0.045201  |

## 2. Sentence Embedding

- 단어 임베딩 개념을 확장하여, 문장 전체의 의미를 하나의 고정 길이 벡터로 표현하는 기법임
- 의미적으로 유사한 문장일수록 벡터 공간상에서 가까운 위치에 매핑되도록 학습함
- 검색, 유사도 분석, 의미 기반 클러스터링 등 다양한 자연어 처리 작업에 활용됨

```python
### SBERT를 이용한 문장 벡터화
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-m3')
sentence_vector = model.encode("자연어처리를 공부합니다")

print(f"문장 벡터 크기: {len(sentence_vector)}")
print(sentence_vector)
```
