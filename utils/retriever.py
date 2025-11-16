import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import RETRIEVAL_K


def hybrid_retrieve(query, vectorstore, bm25_index, corpus_docs, k=RETRIEVAL_K):
    if not vectorstore:
        return []
    
    try:
        # FAISS semantic search
        faiss_docs = vectorstore.similarity_search_with_score(query, k=k)
        
        # BM25 keyword search
        if bm25_index and corpus_docs:
            query_tokens = query.lower().split()
            bm25_scores = bm25_index.get_scores(query_tokens)
            
            # Normalize scores
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
            max_faiss = max([score for _, score in faiss_docs]) if faiss_docs else 1
            
            # Create score dictionary
            doc_scores = {}
            
            # Add BM25 scores
            for doc, score in zip(corpus_docs, bm25_scores):
                doc_content = doc.page_content
                doc_scores[doc_content] = {
                    "bm25": score / max_bm25,
                    "faiss": 0,
                    "doc": doc
                }
            
            # Add FAISS scores
            for doc, score in faiss_docs:
                doc_content = doc.page_content
                faiss_similarity = 1 - (score / max_faiss) if max_faiss > 0 else 0
                
                if doc_content in doc_scores:
                    doc_scores[doc_content]["faiss"] = faiss_similarity
                else:
                    doc_scores[doc_content] = {
                        "bm25": 0,
                        "faiss": faiss_similarity,
                        "doc": doc
                    }
            
            # Calculate combined scores (60% FAISS, 40% BM25)
            for content in doc_scores:
                doc_scores[content]["combined"] = (
                    0.4 * doc_scores[content]["bm25"] +
                    0.6 * doc_scores[content]["faiss"]
                )
            
            # Sort by combined score
            ranked_docs = sorted(
                doc_scores.values(),
                key=lambda x: x["combined"],
                reverse=True
            )
            
            return [item["doc"] for item in ranked_docs[:k]]
        
        return [doc for doc, _ in faiss_docs[:k]]
    
    except Exception as e:
        print(f"Error during retrieval: {str(e)}")
        return []


def retrieve_context(query, vectorstore, bm25_index, corpus_docs, k=RETRIEVAL_K):
    try:
        # Retrieve documents
        docs = hybrid_retrieve(query, vectorstore, bm25_index, corpus_docs, k)
        
        # Format context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Get page numbers
        pages = sorted(set([doc.metadata.get('page', 0) + 1 for doc in docs[:3]]))
        
        return context, pages
    
    except Exception as e:
        print(f"Error retrieving context: {str(e)}")
        return "", []
