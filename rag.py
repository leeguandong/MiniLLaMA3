import os
import torch
from transformers import pipeline, AutoTokenizer, GenerationConfig, PhiForCausalLM
from langchain.document_loaders import TextLoader

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from config import RAGConfig

config = RAGConfig()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embedding = HuggingFaceBgeEmbeddings(
    model_name=config.general_embedding_model,
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True},
    query_instruction="为文本生成向量表示用于文本检索")

tokenizer = AutoTokenizer.from_pretrained(config.model_file)
model = PhiForCausalLM.from_pretrained(config.model_file)
phi_pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.float32,
                    device=device)


def get_token_len(text: str) -> int:
    '''
    统计token长度
    '''
    tokens = tokenizer.encode(text)
    return len(tokens)


def get_doc_db(doc_db, doc):
    if not os.path.exists(doc_db):
        # 1. 从文件读取本地数据集
        loader = TextLoader(doc)
        documents = loader.load()

        # 2. 拆分文档
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=96,
                                                       chunk_overlap=8,
                                                       length_function=get_token_len,
                                                       separators=["\n\n", "\n", "。", " ", ""])
        splited_documents = text_splitter.split_documents(documents)
        # print(splited_documents[0:2])

        # 3. 向量化并保存到本地目录
        db = Chroma.from_documents(splited_documents, embedding, persist_directory=doc_db)
        db.persist()
    else:
        db = Chroma(persist_directory=doc_db, embedding_function=embedding)
    return db


def predict(db, template, max_new_token):
    question = "快舟一号甲的近地轨道运载能力是多少？"

    # 构造prompt
    similar_docs = db.similarity_search(question, k=1)
    for i, doc in enumerate(similar_docs):
        template += f"{i}. {doc.page_content}"

    template += f'\n以下为问题：\n{question}'
    print(template)

    prompt = f"##提问:\n{template}\n##回答:\n"
    outputs = phi_pipe(prompt,
                       num_return_sequences=1,
                       max_new_tokens=max_new_token,
                       pad_token_id=tokenizer.eos_token_id)

    print(outputs[0]['generated_text'][len(prompt):])


if __name__ == "__main__":
    doc_db = config.doc_db
    doc = config.doc
    template = config.template
    max_new_token = config.max_new_token

    db = get_doc_db(doc_db, doc)

    predict(db, template, max_new_token)
