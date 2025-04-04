import networkx as nx
import json
import random
from tqdm import tqdm
import pickle
import os
import json
import itertools
from tqdm import tqdm
import random
import argparse
import logging
from collections import Counter
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
import torch
import pickle
import networkx as nx
import csv
import transformers
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
import httpx
import pyarrow.parquet as pq
import pandas as pd
from igraph import Graph
import igraph as ig

import re
# 使用2张GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

cot_system_instruction = ('As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. If the information is not enough, you can use your own knowledge to answer the question.'
                          'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                          'Conclude with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations.')
cot_system_instruction_no_doc = ('As an advanced reading comprehension assistant, your task is to analyze the questions and then answer them. '
                                 'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                                 'Conclude with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations.')

# This is the instruction for the KG-based QA task
cot_system_instruction_kg = ('As an advanced reading comprehension assistant, your task is to analyze extracted information and corresponding questions meticulously. If the knowledge graph information is not enough, you can use your own knowledge to answer the question. '
                                'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                                'Conclude with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations.')

def compute_graph_embeddings(node_list, edge_list_string, sentence_encoder, file_name):
    # 判定embedding是否存在，如果存在则直接加载，否则计算
    if os.path.exists(file_name + "/entity_embedding.pkl") and os.path.exists(file_name + "/fact_embedding.pkl"):
        with open(file_name + "/entity_embedding.pkl", "rb") as f:
            entity_embededdings = pickle.load(f)
        with open(file_name + "/fact_embedding.pkl", "rb") as f:
            fact_embededdings = pickle.load(f)
        print("load embeddings from file")
        return entity_embededdings, fact_embededdings

    entity_embededdings = sentence_encoder.encode(node_list, show_progress_bar=True, description="Encode Entities", batch_size=16)
    fact_embededdings = sentence_encoder.encode(edge_list_string, show_progress_bar=True, description="Encode Facts", batch_size=16)
    with open(file_name+"/entity_embedding.pkl", "wb") as f:
        pickle.dump(entity_embededdings, f)
    with open(file_name+"/fact_embedding.pkl", "wb") as f:
        pickle.dump(fact_embededdings, f)
    return entity_embededdings, fact_embededdings

def build_faiss_index(node_embededdings, edge_embededdings):
       
    dimension = len(node_embededdings[0])
    
    node_faiss_index = faiss.IndexHNSWFlat(dimension, 8)

    X = np.array(node_embededdings).astype('float32')
    print("the shape of node embeddings: ", X.shape)

    # normalize the vectors
    faiss.normalize_L2(X)
    print("Add index for nodes")

    # batched add
    for i in tqdm(range(0,X.shape[0], 32)):
        node_faiss_index.add(X[i:i+32])
    # index.add(X)


    print("Add node index:", node_faiss_index.ntotal)

    edge_faiss_index = faiss.IndexHNSWFlat(dimension, 8)

    X = np.array(edge_embededdings).astype('float32')
    print("the shape of edge embeddings: ", X.shape)

    # normalize the vectors

    faiss.normalize_L2(X)
    print("Add index for edges")

    # batched add
    for i in tqdm(range(0,X.shape[0], 32)):
        edge_faiss_index.add(X[i:i+32])
    # index.add(X)
    print("Add edge index:", edge_faiss_index.ntotal)

    return node_faiss_index, edge_faiss_index

def evaluate(generated_text, reference_text):

        # evaluate in two different methods, EM and F1
        # The EM score is the percentage of questions that were answered exactly correctly
        # The F1 score is the average of the F1 score of each question. The F1 score is the harmonic mean of precision and recall
        # The precision is the number of correct words in the answer divided by the number of words in the answer
        # The recall is the number of correct words in the answer divided by the number of words in the reference answer
        # The F1 score is 2 * (precision * recall) / (precision + recall)

        generated_text = generated_text.split()
        reference_text = reference_text.split()

        correct = 0
        for word in generated_text:
            if word in reference_text:
                correct += 1
        try:
            precision = correct / len(generated_text)
        except:
            precision = 0
        try:
            recall = correct / len(reference_text)
        except:
            recall = 0

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        em = 1 if generated_text == reference_text else 0

        return em, f1

def llm_pipline():
    # Initialize the llm model
    limits = httpx.Limits(max_keepalive_connections=8, max_connections=8)
    client = httpx.Client(limits=limits, timeout=httpx.Timeout(5*60, read=5*60))
    model = AzureOpenAI(azure_endpoint ="https://hkust.azure-api.net", api_key="YOUR-API-KEY", api_version = "2024-05-01-preview", http_client=client)
    return model

def llm_infer(llm, model_name, messages):
    try:
        response = llm.chat.completions.create(
            model = model_name,
            messages = messages,
            max_completion_tokens = 2048,
            temperature = 0.0
        )
        response_message = response.choices[0].message.content
        metadata = {
            "prompt_tokens": response.usage.prompt_tokens, 
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }
    except:
        response_message = "Nothing"
        metadata = {
                "prompt_tokens": 0, 
                "completion_tokens": 0,
                "finish_reason": "stop",
        }
    return response_message, metadata

def normalize_answer(answer: str) -> str:

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(answer))))

def get_questions_docs(file_path):
    with open(file_path, "r") as f:
        questions = json.load(f)
    question_list = [question["question"] for question in questions]
    answer_list = [question["answer"] for question in questions]
    return question_list, answer_list

def clean_text(text):
    # remove NUL as well
    new_text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("\v", " ").replace("\f", " ").replace("\b", " ").replace("\a", " ").replace("\e", " ").replace(";", ",")
    new_text = new_text.replace("\x00", "")
    return new_text

def load_chunks_embedding(chunk_file_path):
        # the chunks are in the format of (hasi_id, content, embedding)
        chunks = pd.read_parquet(chunk_file_path)
        contents = chunks["content"].tolist()
        chunk_embeddings = np.array(chunks["embedding"].tolist()) #(#chunks, 4096)
        return contents, chunk_embeddings

def kg_data_loader_pkl(graph_dir):
    with open(graph_dir, "rb") as f:
        KG = pickle.load(f)
    edge_list = list(KG.edges) #三元组
    node_list = list(KG.nodes) #字符串
    edge_list_with_relation = [(edge[0], KG.edges[edge]["relation"], edge[1])  for edge in edge_list]
    edge_list_string = [f"{edge[0]}  {KG.edges[edge]['relation']}  {edge[1]}" for edge in edge_list]

    return KG, node_list, edge_list, edge_list_with_relation, edge_list_string

def kg_data_loader_graphML(graph_dir):
    KG = ig.Graph.Read_GraphML(graph_dir)
    node_list = KG.vs['id'] #字符串列表，node的数量
    edge_list = [(KG.vs[e.source]['id'], KG.es[e.index]['relation'], KG.vs[e.target]['id']) for e in KG.es] #三元组，124388
    edge_list_with_relation = [(KG.vs[e.source]['id'], KG.es[e.index]['relation'], KG.vs[e.target]['id']) for e in KG.es]
    edge_list_string = [f"{KG.vs[e.source]['id']}  {KG.es[e.index]['relation']}  {KG.vs[e.target]['id']}" for e in KG.es]
    print("load KG from graphML")
    print("number of nodes: ", len(node_list), "number of edges: ", len(edge_list))
    return KG, node_list, edge_list, edge_list_with_relation, edge_list_string


def min_max_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

class HippoRAGRetriver():
    def __init__(self, graph_dir, question_file_path, chunk_file_path, embedding_path , sentence_encoder):
        
        self.model= llm_pipline()
        self.sentence_encoder = sentence_encoder
        self.queries, self.answers = get_questions_docs(question_file_path)
        self.query_embeddings = sentence_encoder.encode(self.queries, show_progress_bar=True, description="Encode Queries")
        self.KG, self.node_list, self.edge_list, self.edge_list_with_relation, self.edge_list_string = kg_data_loader_graphML(graph_dir) #graph.ml
        self.entity_embeddings, self.fact_embeddings = compute_graph_embeddings(self.node_list, self.edge_list_string, self.sentence_encoder, embedding_path) # 3115个，3888个
        
        self.contents, self.chunk_embeddings = load_chunks_embedding(chunk_file_path)
        self.ent_node_to_num_chunk = {}
        self.entity_name_to_idx = {name: idx for idx, name in enumerate(self.node_list)}
        self.passage_node_idxs = [node.index for node in self.KG.vs if node["type"] == "passage"]
    
    def ner(self, text):
        messages = [
            {"role": "system", "content": "Please extract the entities from the following question and output them separated by comma, in the following format: entity1, entity2, ..."},
            {"role": "user", "content": f"Extract the named entities from: Are Portland International Airport and Gerald R. Ford International Airport both located in Oregon?"},
            {"role": "system", "content": "Portland International Airport, Gerald R. Ford International Airport, Oregon"},
            {"role": "user", "content": f"Extract the named entities from: {text}"},
        ]
        response_message, metadata = llm_infer(self.model, "gpt-4o-mini", messages)
    
        return response_message
    
    def get_fact_scores(self, query_idx): 

        # get the fact scores for the query

        query_fact_scores = np.dot(self.fact_embeddings, self.query_embeddings[query_idx].T) # shape: (#facts, )
        query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
        query_fact_scores = min_max_normalize(query_fact_scores)

        return query_fact_scores

    def rerank_facts(self, query_fact_scores, top_k): 

        # rerank the facts based on the scores, return the top k facts' indices and the top k facts
        
        top_k_facts_indices = np.argsort(query_fact_scores)[-top_k:][::-1].tolist()
        top_k_facts = [self.edge_list_with_relation[i] for i in top_k_facts_indices]

        return top_k_facts_indices, top_k_facts

    def dense_passage_retrieval(self, query_idx):

        # the DPR method ranks the documents based on their similarity scores with the query

        query_doc_scores = np.dot(self.chunk_embeddings, self.query_embeddings[query_idx].T) # shape: (#chunks, )
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores
    
    def count_entity_in_chunk(self):

        # count the number of chunks for each entity

        for node in self.KG.vs:
            self.ent_node_to_num_chunk[node["id"]] = len(set(node["file_id"].split(",")))
        print("complete counting the number of chunks for each entity.")

    def graph_search_with_fact_entities(self, query_idx, query_fact_scores, top_k_facts, top_k_facts_indices):
        # query_idx: the index of the query
        # link_top_k: the number of facts to link to the query
        # query_fact_scores: the scores of the facts for the query

        linking_score_map = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights = np.zeros(len(self.KG.vs['id'])) # 包括entity 和 event，chunk
        passage_weights = np.zeros(len(self.KG.vs['id'])) # 包括entity 和 event，chunk
    

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[top_k_facts_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores

            for phrase in [subject_phrase, object_phrase]:
                phrase_id = self.entity_name_to_idx.get(phrase, None)
                if phrase_id is not None:
                    phrase_weights[phrase_id] = fact_score
                    if self.ent_node_to_num_chunk[phrase] != 0:
                        phrase_weights[phrase_id] /= self.ent_node_to_num_chunk[phrase]
                if phrase not in phrase_scores:
                    phrase_scores[phrase] = []
                phrase_scores[phrase].append(fact_score)

        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        # 不执行get_top_k_weights

        # Get passage scores according to chosen dense retrieval model
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query_idx)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = len(self.KG.vs['id']) - len(self.contents) + dpr_sorted_doc_id
            passage_weights[passage_node_id] = passage_dpr_score * 0.05
            passage_node_text = self.contents[dpr_sorted_doc_id]
            linking_score_map[passage_node_text] = passage_dpr_score * 0.05

        node_weights = phrase_weights + passage_weights

        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'

        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.ppr(node_weights)

        assert len(ppr_sorted_doc_ids) == len(self.contents), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.contents)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores

    def ppr(self, node_weights):

        reset_prob = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)
        pagerank_scores = self.KG.personalized_pagerank(
            vertices=range(self.KG.vcount()),
            damping=0.5,
            directed=False,
            weights=None,
            reset=reset_prob,
            implementation='prpack'
        ) # reset_prob是节点的权重，是一个一维数组，长度应该是节点的数量
        
        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores

    def retrieve(self, retrieve_top_k = 100):

        retrieval_results = []
        for q_idx, query in tqdm(enumerate(self.queries), desc="Retrieving", total=len(self.queries)):
            query_fact_scores = self.get_fact_scores(q_idx)
            top_k_fact_indices, top_k_facts = self.rerank_facts(query_fact_scores, 5)
            
            if len(top_k_facts) == 0:
                logger.info('No facts found after reranking, return DPR results')
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(q_idx)
            else:
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(q_idx, query_fact_scores, top_k_facts, top_k_fact_indices)

            top_k_docs = [self.contents[doc_id] for doc_id in sorted_doc_ids[:retrieve_top_k]]
            retrieval_results.append({"question": query, "docs": top_k_docs, "doc_scores": sorted_doc_scores[:retrieve_top_k]})

        return retrieval_results

    def rag_qa(self):

        retrieval_results = self.retrieve(100)

        qa_results = []
        for q_idx, retrieval_result in tqdm(enumerate(retrieval_results), desc="Answering", total=len(retrieval_results)):
            question = retrieval_result["question"]
            docs = retrieval_result["docs"]
            # doc_scores = retrieval_result["doc_scores"]
            docs = "\n".join(docs)
            messages = [
                {"role": "system", "content": "".join(cot_system_instruction)},
                {"role": "user", "content": f"{docs}\n\n{question}"},
            ]
            response_message, metadata = llm_infer(self.model, "gpt-4o-mini", messages)
            if "Answer:" in response_message:
                response_message = response_message.split("Answer:")[-1]
            if "answer:" in response_message:
                response_message = response_message.split("answer:")[-1]
            qa_results.append({"question": question, "answer": response_message, "gold_answer": self.answers[q_idx], "metadata": metadata})

        return qa_results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", type=str, default="/home/xzhoucs/my_files/ip_code/2wikimultihopqa_kg_entity_only_from_corpus_chunks.graphml") #与chunks拼接后的知识图谱的路径
    parser.add_argument("--question_file_path", type=str, default="/home/xzhoucs/my_files/hippoRAG2/HippoRAG/src/hipporag/gold_2wikimultihopqa.json")# 问题的路径，包含问题和答案
    parser.add_argument("--chunk_file_path", type=str, default="/home/xzhoucs/my_files/hippoRAG2/HippoRAG/outputs/2wikimultihopqa/gpt-4o-mini_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet")# chunks embedding的路径
    parser.add_argument("--embedding_path", type=str, default="/home/xzhoucs/my_files/ip_code/embeddings_entity") #存储图谱中embeddings路径
    parser.add_argument("--result_file_path", type=str, default="/home/xzhoucs/my_files/ip_code/my_2wiki_result_entity.json") #存储结果的路径

    args = parser.parse_args()

    sentence_encoder = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)
    retriever = HippoRAGRetriver(args.graph_dir, args.question_file_path, args.chunk_file_path, args.embedding_path, sentence_encoder)
    retriever.count_entity_in_chunk()
    qa_results = retriever.rag_qa()
    with open(args.result_file_path, "w") as f:
        json.dump(qa_results, f)
    
    em, f1 = 0, 0
    for qa_result in qa_results:
        em_, f1_ = evaluate(normalize_answer(qa_result["answer"]), normalize_answer(qa_result["gold_answer"]))
        em += em_
        f1 += f1_
    em /= len(qa_results)
    f1 /= len(qa_results)
    print(f"EM: {em}, F1: {f1}")
    
