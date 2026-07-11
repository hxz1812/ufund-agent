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
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("Missing OPENAI_API_KEY. Add it to your .env file.")


def main() -> None:
    #user inputs
    parser=argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--n-ctx",type=int,default=2048)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--export-drive", type=bool,default=False)

    args = parser.parse_args()

    user_input=args.prompt
    max_tokens=args.max_tokens
    temperature=args.temperature
    n_ctx=args.n_ctx
    max_steps=args.max_steps
    export_drive=args.export_drive

    system_prompt=("You are an assistant who truthfully and thoughtfully answers questions the members of UFund Investment LLC have.")

    prompt = f"""{system_prompt}

CRITICAL INSTRUCTIONS:
1. Everything in your response will have evidence to support it from given documents and text.
2. If you cannot find evidence or facts, you will respond with "I don't know."

User request: {user_input}

Response:"""

    #local model
    llm = Llama(
        model_path=model_path,
        temperature=temperature,
        n_ctx=n_ctx,
        verbose=False,
    )

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

    #retriever
    retriever=vectorstore.as_retriever(
        search_kwargs={"k":4}
        )

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
    retrieved_docs=retriever.invoke(user_input)
    answer=answer_with_local_llm(
        llm=llm,
        user_input=user_input,
        retrieved_docs=retrieved_docs,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature
        )
    print(answer)

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

if __name__ == "__main__":
    main()
