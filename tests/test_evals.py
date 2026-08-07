import pytest
from langchain_core.documents import Document
import app
from evals.golden_datasets import (TOOL_ROUTING_GOLDEN, ANSWER_GOLDEN,)

#load real local Llama model once for evals
@pytest.fixture(scope="session")
def real_llm_config():
    return app.build_llm(
        llm_provider="local",
        model_path=app.model_path,
        n_ctx=2048,
        temperature=0,
        openai_model="gpt-3.5-turbo",
        )

#behaves like Chroma for tools but no embeddings or OpenAI
class EvalVectorStore:
    def __init__(self, document_text, file_name):
        self.document = Document(
            page_content=document_text,
            metadata={
                "file_name": file_name,
                "source_system": "eval",
                "relative_path": file_name,
                "source": file_name,
                "file_type": ".txt",
                },
            )
        self.search_calls = []

    def similarity_search(self, query, k=4,):
        self.search_calls.append((query, k))
        return [self.document]

    def get(self, include=None, limit=500, offset=0,):
        metadatas = [self.document.metadata]
        return {"metadatas": (metadatas[offset: offset + limit])}

#Tool routing
@pytest.mark.eval
@pytest.mark.parametrize(
    "case",
    TOOL_ROUTING_GOLDEN,
    ids=[case["name"] for case in TOOL_ROUTING_GOLDEN],
    )

def test_eval_tool_routing(real_llm_config, case,):
    messages = [{
        "role": "user",
        "content": case["input"]
        }]
    step = app.agent_step(
        llm_config=real_llm_config,
        messages=messages,
        system_prompt=app.SYSTEM_PROMPT,
        max_tokens=256,
        temperature=0,
        )

    assert step is not None, ("Agent failed to return a valid structured "
                              "action.")
    assert step.get("action") == "tool_use"
    assert step.get("tool") == (case["expected_tool"])
    assert isinstance(step.get("arguments"), dict,)

    if (case["expected_tool"]=="search_docs"):
        query = step["arguments"].get("query","",)
        assert query.strip()!=""

@pytest.mark.eval
@pytest.mark.parametrize(
    "case",
    ANSWER_GOLDEN,
    ids=[case["name"] for case in ANSWER_GOLDEN],
    )

def test_eval_end_to_end(real_llm_config, case,):
    vectorstore = EvalVectorStore(
        document_text=(case["document_text"]),
        file_name=case["file_name"],
        )
    result = app.run_loop(
        llm_config=real_llm_config,
        system_prompt=app.SYSTEM_PROMPT,
        user_input=case["input"],
        vectorstore=vectorstore,
        max_tokens=512,
        temperature=0,
        max_steps=5,
        trace=False,
        )
    answer = result.get("answer", "",)
    normalized_answer = (answer.strip().lower().rstrip("."))

    if case["expect_unknown"]:
        assert (normalized_answer.startswith("i don't know")
                or
                normalized_answer.startswith("i do not know"))
        return

    assert result.get("error") is None

    for expected_text in (case["must_contain"]):
        assert (expected_text.lower() in answer.lower())

    documents_used = result.get("documents_used", [],)
    used_files = [document.get("file_name") for document in documents_used]

    assert (case["file_name"] in used_files)
