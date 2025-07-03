# Agent

- LLM을 의사결정 엔진으로 사용하여 작업을 수행하는 시스템임
- 모델은 입력된 데이터를 분석하여 맥락에 맞는 의사결정을 수행함
- 시스템은 사용자의 요청을 이해하고 적절한 해결책을 제시함
- 복잡한 작업을 자동화하여 업무 효율성을 높일 수 있음



[Git 커뮤니티?](https://github.com/modelcontextprotocol/servers?tab=readme-ov-file#%EF%B8%8F-official-integrations)

[Smithery](https://smithery.ai/)

[MCP](https://modelcontextprotocol.io/introduction)


## Tool Calling

- LLM이 외부 시스템과 상호작용하기 위한 함수 호출 매커니즘임
- LLM은 정의된 도구나 함수를 통해 **외부 시스템**과 통신하고 작업을 수행함
- 즉, 모델이 시스템과 직접 상호작용할 수 있게 하는 기능임
- 구조화된 출력을 통해 API나 데이터베이스와 같은 시스템 요구사항을 충족함
- 스키마 기반 응답으로 시스템간 효율적 통신 가능함

### 1\) Tool Creation

- `@tool` 데코레이터를 사용하여 함수에 스키마 정보를 추가할 수 있음

### 2\) Tool Binding

- 모델-도구 연결로 입력 스키마를 자동 인식함
- LLM이 도구 호출을 할지 말지 선택함 (도구 호출 전임)


```python
from langchain_openai import ChatOpenAI

# 모델 정의
model = ChatOpenAI(model="gpt-4.1-mini",temperature=0)

# 도구 목록
tools = [get_weather, search_db]

# 도구를 모델에 바인딩
model_with_tools = model.bind_tools(tools)

# 사용자 쿼리를 모델에 전달
result = model_with_tools.invoke("서울 날씨 어때?") # content 비어있음, 어떤 함수를 호출할지만 정함

```

content='' additional_kwargs={'tool_calls': [{'id': 'call_l8a8p3CBa4kLdPJGQC8X0WO0', 'function': {'arguments': '{"city":"서울"}', 'name': 'get_weather'}, 'type': 'function'}], 'refusal': None} response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 99, 'total_tokens': 113, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4.1-mini-2025-04-14', 'system_fingerprint': 'fp_6f2eabb9a5', 'id': 'chatcmpl-BoTKS9Ti1OUGzpm9FtGux1R6WZxQn', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None} id='run--c5493795-5ee3-480b-a8ba-3b8ee9ad427e-0' tool_calls=[{'name': 'get_weather', 'args': {'city': '서울'}, 'id': 'call_l8a8p3CBa4kLdPJGQC8X0WO0', 'type': 'tool_call'}] usage_metadata={'input_tokens': 99, 'output_tokens': 14, 'total_tokens': 113, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}

### 3\) Tool Calling

```python
result.tool_calls
```

### 4\) Tool Execution

---

## AgentExecutor


---


