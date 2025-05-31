# Tokenization

## 1. Korean Word Tokenization

- 형태소 분석 기반으로 의미 있는 최소 단위로 분리함
- 조사, 어미 등 문법 요소도 개별 토큰으로 분리함

```python
# Kiwi 형태소 분석기 로드
from kiwipiepy import Kiwi 
kiwi = Kiwi()

# 토큰화
text = "자연어처리를 공부합니다"
tokens = kiwi.tokenize(text)
```

```markdown
# tokens
[Token(form='자연어 처리', tag='NNP', start=0, len=5),
 Token(form='를', tag='JKO', start=5, len=1),
 Token(form='공부', tag='NNG', start=7, len=2),
 Token(form='하', tag='XSV', start=9, len=1),
 Token(form='ᆸ니다', tag='EF', start=9, len=3)]
```

## 2. Korean Subword Tokenization

- 자주 등장하는 문자열 패턴을 통계적으로 학습하여, 형태소보다 더 작은 단위로 분리함
- 신조어, 미등록어(OOV) 문제에 유리함
- 주로 BPE(Byte Pair Encoding) 또는 SentencePiece 알고리즘 기반 모델 사용함
- `KcBERT` 토크나이저 ([문서](https://huggingface.co/beomi/kcbert-base))
    - SentencePiece 기반 서브워드 토크나이저를 사용함
    - 한국어에 특화된 어휘 사전을 보유함
    - 특수 토큰은 [CLS]-문장 시작, [SEP]-문장 구분, [MASK], [PAD], [UNK]를 지원함
    - 서브워드는 '##' 접두어 표시로 구분함

```python
# 허깅페이스 트랜스포머 라이브러리에서 토크나이저 로드
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('beomi/kcbert-base')

# 토큰화
text = "자연어처리를 공부합니다"
tokens = tokenizer.tokenize(text)
```

```markdown
# tokens
['자연', '##어', '##처리', '##를', '공부', '##합니다']
```