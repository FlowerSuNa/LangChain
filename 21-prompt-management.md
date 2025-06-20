# 프롬프트 관리

[Langfuse 기반](https://langfuse.com/docs)
[LangChain 연동](https://langfuse.com/docs/prompts/example-langchain)
- 최근 버전3 나옴

## 프롬프트 생성

- 변수는 꼭 이중 중괄호 사용할 것 `{{변수명}}`

```python
# 텍스트 프롬프트 생성
langfuse.create_prompt(
    name="test",  # 프롬프트 이름
    type="text",          
    prompt="당신은 {{subject}} 전문가 입니다. {{title}}에 대해 설명하세요.",
    labels=["production"],       # 프로덕션 레이블
    tags=["test", "qa", "text"],    # 태그
    config={
        "model": "gpt-4.1-mini",
        "temperature": 0.7
    }
)
```


```python
# 챗 프롬프트 생성
langfuse.create_prompt(
    name="test-chat",  # 프롬프트 이름
    type="chat",          
    prompt=[
        {
            "role": "system",
            "content": "당신은 {{subject}} 전문가 입니다."
        },
        {
            "role": "user",
            "content": "{{title}}에 대해 설명하세요."
        }
    ],
    labels=["production"],       # 프로덕션 레이블
    tags=["test", "qa", "text"],    # 태그
    config={
        "model": "gpt-4.1-mini",
        "temperature": 0.7
    }
)  
```

```python
# 프로덕션 버전 가져오기
prompt = langfuse.get_prompt("test-chat")

# 프롬프트 출력
print(f"모델: {prompt.config['model']}")
print(f"온도: {prompt.config['temperature']}")
print(f"라벨: {prompt.labels}")
print(f"태그: {prompt.tags}")
print(f"프롬프트: {prompt.prompt}")
print("-" * 100)

# 랭체인 프롬프트 출력
print(f"프롬프트: {prompt.get_langchain_prompt()}")
```

```python
# 새로운 버전 생성
langfuse.create_prompt(
    name="test",  # 같은 이름 사용
    type="text",          
    prompt="당신은 {{subject}} 전문가 입니다. {{title}}에 대해 설명하세요.",
    labels=["staging"],       # 프로덕션 레이블
    tags=["test", "qa"],    # 태그
    config={
        "model": "gpt-4.1-mini",
        "temperature": 0.7
    }
)

# staging 버전 가져오기
prompt_staging = langfuse.get_prompt("test", label="staging")

# 프롬프트 출력
print(f"모델: {prompt_staging.config['model']}")
print(f"온도: {prompt_staging.config['temperature']}")
print(f"라벨: {prompt_staging.labels}")
print(f"프롬프트: {prompt_staging.prompt}")
```