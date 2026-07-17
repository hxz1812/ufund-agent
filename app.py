from dotenv import load_dotenv
import sys
import argparse
from llama_cpp import Llama
from pathlib import Path

import json
import time
import io
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import tiktoken
import numpy as np
from langchain_core.load import dumps, loads

from typing import Any, List, Optional


model_path = str(Path.home() / "Desktop" / "agents-from-scratch" / "models" / "llama-3-8b-instruct.gguf")

stop = ["<|eot_id|>","User:"]


SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

FOLDER_ID = "1OqyC8z5ECyFALzjpQLE-vaNyNI5T2ey0"

OUTPUT_DIR = Path("exported_docs_for_rag")


GOOGLE_EXPORT_TYPES = {
    "application/vnd.google-apps.document": {
        "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "extension": ".docx",
    },
    "application/vnd.google-apps.spreadsheet": {
        "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "extension": ".xlsx",
    },
    "application/vnd.google-apps.presentation": {
        "mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "extension": ".pptx",
    },
    "application/vnd.google-apps.drawing": {
        "mime_type": "image/png",
        "extension": ".png",
    },
}

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

RAG_ANSWER_SCHEMA = """
{
  "answer": "string",
  "evidence_files": ["string"],
  "confidence": "high | medium | low",
  "missing_information": ["string"]
}
"""

TOOL_CHOICES = [
    "calculator",
    "search_docs",
    "summarize_docs",
    "answer_from_docs",
    "none",
]

class AgentState:
    def __init__(self):
        self.steps=0
        self.done=False
    def reset(self):
        self.steps=0
        self.done=False
    def increment_step(self):
        self.steps+=1
    def mark_done(self):
        self.done=True
    def to_dict(self):
        return {"steps": self.steps, "done": self.done}

def main() -> None:
    args = parse_args()
    
    user_input=args.prompt
    max_tokens=args.max_tokens
    temperature=args.temperature
    n_ctx=args.n_ctx
    max_steps=args.max_steps
    export_drive=args.export_drive
    llm_provider=args.llm_provider
    openai_model=args.openai_model

    system_prompt=("You are an assistant who truthfully and thoughtfully answers questions the members of UFund Investment LLC have.")

    prompt = f"""{system_prompt}

CRITICAL INSTRUCTIONS:
1. Everything in your response will have evidence to support it from given documents and text.
2. If you cannot find evidence or facts, you will respond with "I don't know."

User request: {user_input}

Response:"""

    kwargs = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stop": stop if stop is not None else ["<|eot_id|>","User:"],
    }
    
    #load gdrive files
    if export_drive:
        service=get_drive_service()
        exported,downloaded,skipped=export_folder(
            service=service,
            folder_id=FOLDER_ID,
            output_dir=OUTPUT_DIR,
            )

    #indexing files
    DOCS_PATH="exported_docs_for_rag"
    docs,skipped_files=indexing(DOCS_PATH)

    
    #getting llm
    llm_config=build_llm(llm_provider=llm_provider,model_path=model_path,
                         n_ctx=n_ctx,temperature=temperature,openai_model=openai_model)

    #embedder
    #embd = OpenAIEmbeddings()

    #split documents into chunks
    text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=50,
        )
    splits = text_splitter.split_documents(docs)
    #print("Splits:", len(splits))

    #embed and store chunks in chroma
    vectorstore=Chroma.from_documents(
        documents=splits,
        #embedding=OpenAIEmbeddings(),
        )

    #retrieve docs
    retriever=vectorstore.as_retriever(
        search_kwargs={"k":4}
        )
    retrieved_docs=retriever.invoke(user_input) 

    #answer FOR OPENAI
    '''answer_chain = prompt | llm | StrOutputParser()
    
    TRACE_DIR = Path("local_traces")
    TRACE_DIR.mkdir(exist_ok=True)

    answer, retrieved_docs = run_rag_with_local_trace(user_input)
    print("ANSWER:")
    print(answer)
    print()
    print("RETRIEVED DOCS:")
    for i, doc in enumerate(retrieved_docs):
        print("RESULT", i)
        print(doc.metadata)
        print(doc.page_content[:500])
        print()'''

    #LOCAL LLM ANSWER
    '''answer=answer_with_local_llm(
        llm=llm,
        user_input=user_input,
        retrieved_docs=retrieved_docs,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature
        )
    print(answer)'''

    #structured output
    structured_answer = answer_rag_structured(
        llm_config=llm_config,
        user_input=user_input,
        retrieved_docs=retrieved_docs,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if structured_answer is None:
        print("Model failed to return valid JSON.")
    else:
        print(json.dumps(structured_answer, indent=2, ensure_ascii=False))

    #decide() test
    choices=["answer_from_context","not_enough_context"]
    decide_test=decide(llm_config=llm_config,system_prompt=system_prompt,
                       user_input=user_input,choices=choices,)
    print("Decide test:", decide_test)

    #tool test
    tool_test=request_tool(llm_config=llm_config,system_prompt=system_prompt,
                           user_input=user_input)

    #loops
    results=run_loop(llm_config,system_prompt,user_input)
    for i, result in enumerate(results,1):
        print(f"Iteration {i}:")
        action = result.get("action","unknown")
        reason = result.get("reason","No reason provided")
        print(f"  Action: {action}")
        print(f"  Reason: {reason}")
        if i < len(results):
            print()

    #multi query
    template = """You are an AI language modle assistant. Your task is to generate
five different versions of the given user question to retrieve relevant documents
from a vector database. By generating multiple perspectives on the user question,
your goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}"""
    prompt_perspectives=ChatPromptTemplate.from_template(template)
    generate_queries=(
        prompt_perspectives
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
        )
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    docs = retrieval_chain.invoke({"question":user_input})
    #print(len(docs))

##############
# PARSE ARGS #
##############
def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--n-ctx",type=int,default=2048)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--export-drive", action="store_true")
    parser.add_argument("--llm-provider", choices=["local","openai"],default="local",)
    parser.add_argument("--openai-model", default="gpt-3.5-turbo",)
    return parser.parse_args()

###############
# GETTING LLM #
###############
def build_llm(llm_provider,model_path,n_ctx,temperature,openai_model):
    if llm_provider=="local":
        return {
            "provider": "local",
            "client": Llama(
                model_path=str(model_path),
                temperature=temperature,
                n_ctx=n_ctx,
                verbose=False,
                ),
            }
    if llm_provider=="openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Missing OPENAI_API_KEY")
        return {
            "provider": "openai",
            "client": ChatOpenAI(
                model_name=openai_model,
                temperature=temperature,
                ),
            }
    raise ValueError(f"Unsupported LLM provider: {llm_provider}")

def call_llm_text(llm_config, prompt, max_tokens=512, temperature=0, stop=None,):
    provider=llm_config["provider"]
    client = llm_config["client"]
    if provider == "local":
        response= client(prompt=prompt,max_tokens=max_tokens,temperature=temperature,stop=stop or ["<|eot_id|>", "User:"])
        return response["choices"][0]["text"].strip()
    if provider=="openai":
        response=client.invoke(prompt)
        return response.content.strip()
    raise ValueError(f"Unsupported LLM provider: {provider}")
        
    
##########################
#    GDRIVE FUNCTIONS    #
##########################
def get_drive_service():
    creds=None
    if os.path.exists("token.json"):
        creds=Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow=InstalledAppFlow.from_client_secrets_file(
                "credentials.json",
                SCOPES,
                )
            creds=flow.run_local_server(port=0)
        with open("token.json","w") as token:
            token.write(creds.to_json())
    return build("drive","v3",credentials=creds)

def safe_filename(name):
    bad_chars=['/','\\',':','*','?','"','<','>','|']
    for char in bad_chars:
        name=name.replace(char,"_")
    return name.strip()

def list_files_in_folder(service,folder_id):
    files=[]
    page_token=None
    while True:
        response=(service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            spaces="drive",
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token,).execute()
                  )
        files.extend(response.get("files",[]))
        page_token=response.get("nextPageToken")
        if not page_token:
            break
    return files

def export_google_file(service,file_id,file_name,export_info,output_path):
    request=service.files().export_media(
        fileId=file_id,
        mimeType=export_info["mime_type"]
        )
    with io.FileIO(output_path,"wb") as file_handle:
        downloader=MediaIoBaseDownload(file_handle,request)
        done=False
        while not done:
            status,done=downloader.next_chunk()

def download_regular_file(service,file_id,output_path):
    request=service.files().get_media(fileId=file_id)
    with io.FileIO(output_path,"wb") as file_handle:
        downloader=MediaIoBaseDownload(file_handle,request)
        done=False
        while not done:
            status,done=downloader.next_chunk()

def export_folder(service,folder_id,output_dir,relative_path=""):
    output_dir=Path(output_dir)
    current_output_dir=output_dir/relative_path
    current_output_dir.mkdir(parents=True,exist_ok=True)
    files=list_files_in_folder(service,folder_id)
    exported=[]
    downloaded=[]
    skipped=[]
    for file in files:
        file_id=file["id"]
        file_name=file["name"]
        mime_type=file["mimeType"]
        clean_name=safe_filename(file_name)
        if mime_type=="application/vnd.google-apps.folder":
            print(f"Entering folder: {file_name}")
            child_exported,child_downloaded,child_skipped=export_folder(
                service=service,
                folder_id=file_id,
                output_dir=output_dir,
                relative_path=str(Path(relative_path)/clean_name),
                )
            exported.extend(child_exported)
            downloaded.extend(child_downloaded)
            skipped.extend(child_skipped)
            continue
        try:
            if mime_type in GOOGLE_EXPORT_TYPES:
                export_info= GOOGLE_EXPORT_TYPES[mime_type]
                output_path=current_output_dir/f"{clean_name}{export_info['extension']}"
                export_google_file(
                    service=service,
                    file_id=file_id,
                    file_name=file_name,
                    export_info=export_info,
                    output_path=output_path,
                    )
                exported.append(str(output_path))
                print(f"Exported: {output_path}")
            elif mime_type.startswith("application/vnd.google-apps."):
                skipped.append(f"{file_name} | unsupported Google Workspace type: {mime_type}")
                print(f"Skipped: {file_name}")
            else:
                output_path=current_output_dir/clean_name
                download_regular_file(
                    service=service,
                    file_id=file_id,
                    output_path=output_path,
                    )
                downloaded.append(str(output_path))
                print(f"Downloaded: {output_path}")
        except Exception as e:
            skipped.append(f"{file_name} | ERROR: {e}")
            print(f"Error: {file_name} | {e}")
    return exported,downloaded,skipped

############
# INDEXING #
############
def indexing(DOCS_PATH,printing=False):
    docs=[]
    skipped_files=[]
    for path in Path(DOCS_PATH).rglob('*'):
        if path.is_dir() or path.name.startswith('.'):
            continue
        suffix = path.suffix.lower()
        try:
            if suffix in ['.txt', '.md']:
                loader=TextLoader(str(path),encoding='utf-8')
            elif suffix == '.pdf':
                loader=pyPDFLoader(str(path))
            elif suffix=='.docx':
                loader=Docx2txtLoader(str(path))
            elif suffix=='.csv':
                loader=CSVLoader(str(path))
            else:
                skipped_files.append(f"{path} | unsupported file type")
            loaded_docs=loader.load()
            for doc in loaded_docs:
                doc.metadata["source"]=str(path)
                doc.metadata["file_name"]=path.name
                doc.metadata["file_type"]=suffix
                doc.page_content=f"File name: {path.name}\nSource: {path}\n\n{doc.page_content}"
            docs.extend(loaded_docs)
        except Exception as e:
            skipped_files.append(f"{path} | ERROR: {e}")
    if printing:
        print("Loaded docs:", len(docs))
        print("Skipped files:", len(skipped_files))
        if docs:
            print("First doc metadata:")
            print(docs[0].metadata)
            print("Loaded file names:")
            for name in sorted(set(doc.metadata.get("file_name") for doc in docs)):
                print(name)
        if skipped_files:
            print("Skipped:")
            for item in skipped_files[:20]:
                print(item)
    return docs, skipped_files

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding=tiktoken.get_encoding(encoding_name)
    num_tokens=len(encoding.encode(string))
    return num_tokens

def cosine_similarity(vec1,vec2):
    dot_product=np.dot(vec1,vec2)
    norm_vec1=np.linalg.norm(vec1)
    norm_vec2=np.linalg.norm(vec2)
    return dot_product/(norm_vec1*norm_vec2)

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

##################################
# ACTUALLY RUNNING RAG + TRACING #
##################################
def run_rag_with_local_trace(question):
    retrieved_docs=retriever.invoke(question)
    context= "\n\n".join(doc.page_content for doc in retrieved_docs)
    answer = answer_chain.invoke({
        "context": context,
        "question": question,
        })
    trace = {
        "question": question,
        "answer": answer,
        "retrieved_docs": [
            {
                "file_name": doc.metadata.get("file_name"),
                "source": doc.metadata.get("source"),
                "file_type": doc.metadata.get("file_type"),
                "preview": doc.page_content[:1000],
                }
            for doc in retrieved_docs
            ],
        "context_preview": context[:3000],
        }
    trace_path = TRACE_DIR / f"rag_trace{int(time.time())}.json"
    with open(trace_path, "w", encoding = "utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)
    print("Saved local trace to:", trace_path)
    return answer, retrieved_docs

def answer_with_local_llm(llm, user_input, retrieved_docs, system_prompt, max_tokens=512, temperature=0):
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = f"""{system_prompt}

CRITICAL INSTRUCTIONS:
1. Everything in your response will have evidence to support it from given documents and text.
2. If you cannot find evidence or facts, you will respond with "I don't know."
3. Keep the answer as brief as possible. Do not make things up, do not provide extra explanations.

Question: {user_input}

Context:
{context}

Answer:"""

    response = llm(prompt,max_tokens=max_tokens,temperature=temperature,)
    return response["choices"][0]["text"].strip()

#union of retrieved docs
def get_unique_union(documents: list[list]):
    flattened_docs=[dumps(doc) for sublist in documents for doc in sublist]
    unique_docs=list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

#####################
# STRUCTURED OUTPUT #
#####################
def extract_json_from_text(text: str) -> dict | None:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start:end+1])
    except json.JSONDecodeError:
        return None


def generate_structured(
    llm_config,
    user_input: str,
    schema: str,
    system_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    stop= None,
) -> dict | None:

    prompt = f"""{system_prompt}

CRITICAL INSTRUCTIONS:
1. Respond with ONLY valid JSON.
2. No explanations, no markdown, no extra text before or after the JSON.
3. Start your response with {{ and end with }}.
4. Use only the retrieved context provided by the user request.
5. If the answer is not supported by the context, set "answer" to "I don't know."

Schema you must follow:
{schema}

User request:
{user_input}

Response (JSON only):"""

    for attempt in range(3):
        response = call_llm_text(llm_config=llm_config,prompt=prompt,max_tokens=max_tokens,temperature=temperature,stop=stop)
        parsed = extract_json_from_text(response)
        if parsed is not None:
            return parsed

    return None


def answer_rag_structured(
    llm_config,
    user_input: str,
    retrieved_docs,
    system_prompt: str,
    schema: str = RAG_ANSWER_SCHEMA,
    max_tokens: int = 512,
    temperature: float = 0.0,):
    context = "\n\n".join(
        f"File name: {doc.metadata.get('file_name')}\n"
        f"Source: {doc.metadata.get('source')}\n"
        f"Content:\n{doc.page_content}"
        for doc in retrieved_docs
    )

    structured_user_input = f"""Question:
{user_input}

Retrieved context:
{context}

Instructions:
- Answer using only the retrieved context.
- Do not use outside knowledge.
- If the retrieved context does not explicitly answer the question, set "answer" to "I don't know."
- Put source file names in "evidence_files".
- Use "confidence" as high, medium, or low.
- If context is insufficient, explain what is missing in "missing_information".
"""

    return generate_structured(
        llm_config=llm_config,
        user_input=structured_user_input,
        schema=schema,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

###################
# DECISION-MAKING #
###################
def decide(llm_config, system_prompt:str, user_input:str, choices: list[str],
           max_tokens:int=512,temperature:float=0.0, stop:List[str]=["<|eot_id|>","User:"])->str | None:
    options = "\n".join(f"- {choice}" for choice in choices)
    prompt = f"""{system_prompt}
You must choose ONE of the following options. Respond with ONLY valid JSON.

CRITICAL INSTRUCTIONS:
1. Response with ONLY valid JSON
2. No explanations, no markdown, no other text
3. Start your response with {{ and end with }}

Available choices:
{options}

Required JSON format:
{{"decision": "one_of_the_choices_above"}}

User request: {user_input}

Response (JSON only):"""

    for attempt in range(3):
        response = call_llm_text(llm_config=llm_config,prompt=prompt,
                                 max_tokens=max_tokens,temperature=temperature,
                                 stop=stop,)
        parsed = extract_json_from_text(response)
        if parsed and "decision" in parsed:
            decision = parsed["decision"]
            if decision in choices:
                return decision
    return None

#########
# TOOLS #
#########
def request_tool(llm_config, system_prompt:str, user_input:str, choices:list[str]=TOOL_CHOICES,max_tokens:int=512,temperature:float=0.0,
                 stop:list[str]=["<|eot_id|>", "<|end_of_text|>", "\nUser request:", "\nInput:"])->dict|None:
    options = "\n".join(f"- {choice}" for choice in choices)
    prompt=f"""{system_prompt}

You are a tool-calling assistant. When asked a math question, you must respond with ONLY valid JSON.

Available tools: {options}

Tool guidance:
- calculator: use for arithmetic/math calculations
- search_docs: use when the user asks to find information in documents
- summarize_docs: use when the user asks to summarize documents
- answer_from_docs: use when the user asks a question that should be answered from retrieved documents
- none: use when no tool is needed

CRITICAL INSTRUCTIONS:
1. Respond with ONLY valid JSON
2. No explanations, no markdown, no other text
3. Start your response with {{ and end with }}

Example output format:
{{"tool": "calculator", "arguments": {{"a": 42, "b": 7, "operation": "divide"}}}}

User request: {user_input}

Response (JSON only):"""
    for attempt in range(3):
        response=call_llm_text(llm_config=llm_config,prompt=prompt,
                               max_tokens=max_tokens,temperature=temperature,
                               stop=stop,)
        parsed=extract_json_from_text(response)
        if parsed and "tool" in parsed and "arguments" in parsed:
            return parsed
    return None

def execute_tool_call(llm_config,tool_call:dict)->Any:
    return execute_tool(tool_call["tool"],tool_call["arguments"])

#########
# LOOPS #
#########
def agent_step(llm_config, state: AgentState, system_prompt:str, user_input:str,
               max_tokens:int=512,temperature:float=0.0)-> dict | None:
    state_dict=state.to_dict()
    prompt=f"""{system_prompt}

You are an agent. You must decide the next action and respond with ONLY valid JSON.

Current state: steps = {state_dict.get('steps', 0)}, done = {state_dict.get('done', False)}

Available actions: analyze, research, summarize, answer, done

CRITICAL INSTRUCTIONS:
1. Respond with ONLY valid JSON
2. No explanations, no markdown, no other text
3. Start your response with {{ and end with }}

Required JSON format:
{{"action": "action_name", "reason": "explanation"}}

User input: {user_input}

Response (JSON only):"""

    for attempt in range(3):
        response = call_llm_text(llm_config=llm_config, prompt=prompt,
                                 max_tokens=max_tokens, temperature=temperature,
                                 stop=stop)
        parsed = extract_json_from_text(response)
        if parsed and "action" in parsed:
            if "reason" not in parsed:
                parsed["reason"] = f"Taking action: {parsed['action']}"
            state.increment_step()
            return parsed
    return None

def run_loop(llm_config, system_prompt:str, user_input:str, temperature:float=0.0,
             max_steps:int=5):
    state=AgentState()
    state.reset()
    results=[]
    while not state.done and state.steps<max_steps:
        action = agent_step(llm_config, state,system_prompt, user_input, temperature)
        if action:
            results.append(action)
            if action.get("action")=="done":
                state.mark_done()
        else:
            break
    return results

if __name__ == "__main__":
    main()
