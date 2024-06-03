#!/usr/bin/env python
# coding: utf-8

from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
import json
import string
import re

class Bm25Retriever(object):

    def __init__(self):
        self.stopwords = []
        with open("models/stopwords.txt", "r") as f:
            lines = f.readlines()
            for word in lines:
                word = word.strip().strip("\n")
                self.stopwords.append(word)
        self.documents = []
        self.full_documents = {}
        self.retriever = None

    # 初始化BM25的知识库
    def init_bm25(self, documents):
        self.documents = []
        self.full_documents = {}
        for idx, line in enumerate(documents):
           line = line.strip("\n").strip()
           if(len(line)<5):
               continue
           new_line = re.sub("[^\w\s]", "", line)
           new_line = new_line.replace("\n","")
           tokens = ' '.join([word for word in new_line.split(" ") if word not in self.stopwords and word not in string.punctuation])
           self.documents.append(Document(page_content=tokens, metadata={"id": idx}))
           self.full_documents[idx] = Document(page_content=line, metadata={"id": idx})
        self.retriever = BM25Retriever.from_documents(self.documents)

    # 获得得分在topk的文档和分数
    def GetBM25TopK(self, query, topk):
        self.retriever.k = topk
        query = re.sub("[^\w\s]", "", query)
        query = query.replace("\n","")
        query_tokens = ' '.join([word for word in query.split(" ") if word not in self.stopwords and word not in string.punctuation])
        ans_docs = self.retriever.invoke(query_tokens)
        ans = []
        for line in ans_docs:
            idx = line.metadata["id"]
            ans.append(self.full_documents[idx])
        return ans
if __name__ == '__main__':
    
    import bz2
    import json
    from main_content_extractor import MainContentExtractor
    
    dataset_path = "/root/autodl-tmp/meta-comprehensive-rag-benchmark-kdd-cup-2024/data/dev_data.jsonl.bz2"
    bm25 = Bm25Retriever() 
    with bz2.open(dataset_path, "rt") as file: 
        for idx, line in enumerate(file): 
            jdata = json.loads(line)
            pages = jdata["search_results"]
            query = jdata["query"]
            print("########################  " + str(idx))
            print("query:   ")
            print(query)
            pages_sentences = []
            for page in pages:
                html_str = page["page_result"]
                url = page["page_url"]
                extracted_markdown = MainContentExtractor.extract(html_str, include_links = False, output_format="markdown")
                if(extracted_markdown):
                    sentences = re.split(r'\#+',extracted_markdown)
                    pages_sentences.extend(sentences)
            bm25.init_bm25(pages_sentences)
            res = bm25.GetBM25TopK(query, 1)
            print(res)
