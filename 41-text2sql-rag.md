# Text2SQL RAG

## Text2SQL

- 자연어로 된 질문을 SQL 쿼리로 반환하도록 하는 기술임
- 데이터베이스 스키마 기반으로 정확한 쿼리를 생성하는 것이 목표임
- 개발가 아닌 사용자도 데이터베이스 검색이 가능함
- 즉, 데이터베이스 접근성을 높이는 자연어 인터베이스 기술임



[Link](https://python.langchain.com/docs/integrations/tools/sql_database/)

[LangSmith Hub](https://smith.langchain.com/hub/)


(https://python.langchain.com/docs/integrations/tools/sql_database/#setup)

(https://python.langchain.com/docs/how_to/sql_large_db/#many-tables)

(https://python.langchain.com/docs/integrations/tools/)

## Chain 구현

### create_sql_query_chain

```python
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatOpenAI
from langchain.chains import create_sql_query_chain

# 데이터베이스 연결
db = SQLDatabase.from_uri("db...")

# SQL Chain 설정
llm = ChatOpenAI(model="gpt-4.1-mini")
sql = create_sql_query_chain(llm=llm, db=db)

# SQL 생성
generated_sql = sql.invoke({'question':"..."})
```

### 구현

```python
from langchain_openai import ChatOpenAI
from langchain import hub

# 모델 생성
llm = ChatOpenAI(model="gpt-4.1-mini")

# SQL 생성 프롬프트 다운로드 (Langchain Hub)
query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

# 프롬프트 메시지 출력
for message in query_prompt_template.messages:
    message.pretty_print()
```

```
================================ System Message ================================

Given an input question, create a syntactically correct {dialect} query to run to help find the answer. Unless the user specifies in his question a specific number of examples they wish to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Only use the following tables:
{table_info}
================================ Human Message =================================

Question: {input}
```

```python
# 입력 필드 확인
query_prompt_template.input_schema.model_json_schema()
```

```
{'properties': {'dialect': {'title': 'Dialect', 'type': 'string'},
  'input': {'title': 'Input', 'type': 'string'},
  'table_info': {'title': 'Table Info', 'type': 'string'},
  'top_k': {'title': 'Top K', 'type': 'string'}},
 'required': ['dialect', 'input', 'table_info', 'top_k'],
 'title': 'PromptInput',
 'type': 'object'}
```

```python
from typing import Annotated, TypedDict
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool

class State(TypedDict):
    question: str  # 입력 질문
    query: str     # 생성된 쿼리
    result: str    # 쿼리 결과
    answer: str    # 생성된 답변

class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

# 데이터베이스 연결
db = SQLDatabase.from_uri("db...")

# 쿼리 생성
response = write_query({"question": "..."})

# 쿼리 실행
execute_query({"query": response["query"]})
```


## Agent - React Pattern

```python
from langgraph.prebuilt import create_react_agent
```


## Cardinality

- 고유명사 처리를 위한 벡터 스토어를 구축하고, 사용자 질문 내 고유명사 맞춤법 자동 검증 기능을 추가할 수 있음
- 이를 통해. 정확한 엔티티 매칭을 통해 쿼리 및 검색 정확도를 향상시킬 수 있음


---

