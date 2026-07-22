import sys
import pytest
from langchain_core.documents import Document
import app


#Fake Chroma store (no real database or OpenAI used)
class FakeVectorStore:
    def __init__(self):
        self.search_calls=[]

    def similarity_search(self, query: str, k:int=4):
        self.search_calls.append((query,k))
        return [Document(
            page_content="Egnyte is referenced in the company document.",
            metadata={
                "file_name": "platform_sop.txt",
                "source": "documents/platform_sop.txt",
                "file_type": ".txt",
                },
            )]

    def get(self, include=None):
        return {
            "metadatas": [
                {"file_name": "platform_sop.txt"},
                {"file_name": "investment_update.pdf"},
                {"file_name": "platform_sop.txt"}, #to test duplicate handling
                ]
            }

#Fake store to record documents added during reindexing
class FakeAddVectorStore:
    def __init__(self):
        self.added_documents=[]

    def add_documents(self, documents):
        self.added_documents.extend(documents)

#########
# TESTS #
#########
def test_extract_json_from_plain_json():
    result=app.extract_json_from_text(
        '{"action": "final_answer", "answer": "Hello"}'
        )
    assert result=={
        "action": "final_answer",
        "answer": "Hello",
        }

def test_extract_json_from_surrounding_text():
    result=app.extract_json_from_text(
        'Model response: {"decision": "search_docs"} extra text'
        )
    assert result=={
        "decision": "search_docs",
        }

def text_extract_json_invalid_input():
    result = app.extract_json_from_text(
        "No valid JSON"
        )
    assert result is None

def test_indexing(tmp_path):
    supported_file = tmp_path / "company_notes.txt"
    supported_file.write_text(
        "UFund uses this doc for testing.",
        encoding="utf-8",
        )
    unsupported_file = tmp_path / "image.xyz"
    unsupported_file.write_text(
        "Unsupported content",
        encoding="utf-8",
        )
    docs, skipped_files=app.indexing(tmp_path)
    assert len(docs)==1
    assert docs[0].metadata["file_name"]=="company_notes.txt"
    assert docs[0].metadata["file_type"]==".txt"
    assert "UFund uses this document" in docs[0].page_content
    assert len(skipped_files)==1
    assert "unsupported file type" in skipped_files[0]

def test_search_docs():
    vectorstore=FakeVectorStore()
    results=app.search_docs(
        vectorstore=vectorstore,
        query="Egnyte",
        k=3,
        )
    assert vectorstore.search_calls==[("Egnyte",3)]
    assert len(results)==1
    assert results[0]["file_name"]=="platform_sop.txt"
    assert "Egnyte" in results[0]["content"]

def test_summarize_docs():
    vectorstore=FakeVectorStore()
    result=app.summarize_docs(
        vectorstore=vectorstore,
        limit=10,
        )
    assert result["file_count"]==2
    assert result["available_files"]==[
        "investment_update.pdf",
        "platform_sop.txt",
        ]

def test_execute_tool_unknown_tool():
    with pytest.raises(ValueError, match="Unknown tool"):
        app.execute_tool(
            tool_name="fake_tool",
            arguments={},
            vectorstore=FakeVectorStore(),
            )

def test_execute_tool_call_error():
    result=app.execute_tool_call(
        tool_call={
            "tool": "fake_tool",
            "arguments": {},
            },
        vectorstore=FakeVectorStore(),
        )
    assert result["success"] is False
    assert result["tool"]=="fake_tool"
    assert "Unknown tool" in result["error"]

def test_run_loop(monkeypatch):
    planned_steps=iter([
        {
            "action": "tool_use",
            "tool": "search_docs",
            "arguments": {
                "query": "Egnyte",
                "k":2,
                },
            "reason": "Search company documents.",
            },
        {
            "action": "final_answer",
            "answer": "Egnyte is mentioned in the platform SOP.",
            "reason": "Supported by retrieved document.",
            },
        ])
    def fake_agent_step(**kwargs):
        return next(planned_steps)
    monkeypatch.setattr(app,"agent_step",fake_agent_step,)
    vectorstore=FakeVectorStore()
    result = app.run_loop(
        llm_config={"provider": "fake"},
        system_prompt="Test system prompt",
        user_input="What does UFund say about Egnyte?",
        vectorstore=vectorstore,
        max_steps=3,
        )
    assert result["answer"]==(
        "Egnyte is mentioned in the platform SOP."
        )
    assert vectorstore.search_calls==[("Egnyte",2)]

def test_normal_question(monkeypatch,tmp_path,):
    chroma_directory=tmp_path / "chroma_db"
    chroma_directory.mkdir()
    fake_vectorstore=FakeVectorStore()
    monkeypatch.setattr(app,"CHROMA_DIR", chroma_directory,)
    monkeypatch.setenv("OPENAI_API_KEY","fake-test-key",)
    monkeypatch.setattr(app,"OpenAIEmbeddings",lambda **kwargs: object(),)
    monkeypatch.setattr(app,"Chroma",lambda **kwargs: fake_vectorstore,)
    monkeypatch.setattr(app,"build_llm",lambda **kwargs: {"provider": "fake"},)
    monkeypatch.setattr(app,"run_loop",lambda **kwargs: {
        "answer": "Test answer",
        "steps": [],
        },)
    def fail_if_indexing_is_called(*args, **kwargs):
        pytest.fail(
            "indexing() must not run during a normal question."
        )
    monkeypatch.setattr(app,"indexing",fail_if_indexing_is_called,)
    monkeypatch.setattr(sys,"argv",[
        "app.py",
        "--prompt",
        "What does the document say?",
        ],)
    app.main()

def test_reindex_loads_documents_once(monkeypatch,tmp_path,):
    fake_vectorstore = FakeAddVectorStore()
    indexing_calls = []
    def fake_indexing(path, printing=False):
        indexing_calls.append(path)
        return [Document(
            page_content=(
                "This is a document used for reindex testing."
            ),
            metadata={
                "file_name": "test.txt",
                "source": "test.txt",
                "file_type": ".txt",
            },
        )], []
    monkeypatch.setattr(app,"CHROMA_DIR",tmp_path / "chroma_db",)
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path / "documents",)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key",)
    monkeypatch.setattr(app, "OpenAIEmbeddings",lambda **kwargs: object(),)
    monkeypatch.setattr(app, "Chroma",lambda **kwargs: fake_vectorstore,)
    monkeypatch.setattr(app, "indexing", fake_indexing,)
    monkeypatch.setattr(sys, "argv",[
        "app.py",
        "--reindex",
        ],)
    app.main()
    assert len(indexing_calls) == 1
    assert len(fake_vectorstore.added_documents) >= 1
