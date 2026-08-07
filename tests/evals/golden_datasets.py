TOOL_ROUTING_GOLDEN = [
    {
        "name": "project question uses search",
        "input": (
            "According to the company documents, what is the Nashville project "
            "occupancy?"
        ),
        "expected_tool": "search_docs",
    },
    {
        "name": "specific document question uses search",
        "input": (
            "What do the company documents say about Egnyte?"
        ),
        "expected_tool": "search_docs",
    },
    {
        "name": "document listing uses summarize",
        "input": (
            "What documents are available in the knowledge base?"
        ),
        "expected_tool": "summarize_docs",
    },
    {
        "name": "file list uses summarize",
        "input": (
            "List the files currently available."
        ),
        "expected_tool": "summarize_docs",
    },
]


ANSWER_GOLDEN = [
    {
        "name": "grounded occupancy answer",
        "input": (
            "According to the documents, what is the Nashville project "
            "occupancy?"
        ),
        "document_text": (
            "The Nashville project currently has an occupancy rate of 91%."
        ),
        "file_name": "nashville_test.txt",
        "must_contain": ["91"],
        "expect_unknown": False,
    },
    {
        "name": "unsupported answer becomes unknown",
        "input": (
            "According to the documents, who is the CEO's favorite musician?"
        ),
        "document_text": (
            "The Nashville project currently has an occupancy rate of 91%."
        ),
        "file_name": "nashville_test.txt",
        "must_contain": [],
        "expect_unknown": True,
    },
]
