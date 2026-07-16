import os
import re
from typing import List, TypedDict, Dict, Any
from duckduckgo_search import DDGS
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from hybrid_retrieval import hybrid_retrieve

# Define State
class AgentState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    web_search_required: bool
    sources: List[Dict[str, Any]]
    steps: List[str]

class StudyAgent:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Add Nodes
        workflow.add_node("retrieve", self.node_retrieve)
        workflow.add_node("grade_documents", self.node_grade_documents)
        workflow.add_node("web_search", self.node_web_search)
        workflow.add_node("generate", self.node_generate)

        # Define Edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        
        # Decide next step after grading
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_next_step,
            {
                "web_search": "web_search",
                "generate": "generate"
            }
        )
        
        workflow.add_edge("web_search", "generate")

        # Decide if final generation has hallucinations
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation,
            {
                "grounded": END,
                "hallucinating_fallback": "web_search",
                "max_attempts_reached": END
            }
        )

        return workflow.compile()

    # --- Node Implementations ---

    def node_retrieve(self, state: AgentState) -> Dict[str, Any]:
        print("\n--- AGENT: RETRIEVING DOCUMENTS ---")
        question = state["question"]
        # Use our high-precision Hybrid Retrieval (Dense + BM25 + RRF)
        docs_with_scores = hybrid_retrieve(self.vectorstore, question, k=5)
        docs = [doc for doc, _ in docs_with_scores]
        
        steps = state.get("steps", [])
        steps.append("Retrieved chunks using Hybrid Retrieval (Vector + BM25)")

        return {
            "documents": docs,
            "steps": steps
        }

    def node_grade_documents(self, state: AgentState) -> Dict[str, Any]:
        print("\n--- AGENT: GRADING DOCUMENT RELEVANCE ---")
        question = state["question"]
        docs = state["documents"]
        steps = state.get("steps", [])
        steps.append("Grading source document relevance to question")

        if not docs:
            print("No documents retrieved. Flagging for web search.")
            return {
                "web_search_required": True,
                "steps": steps
            }

        # Combine document contents for a single grading prompt to save API latency
        context = "\n\n".join([f"Document {i}:\n{d.page_content}" for i, d in enumerate(docs, 1)])
        
        prompt = f"""
You are an academic grader evaluating if the provided document context contains information relevant to answer the question.

CONTEXT:
{context}

QUESTION:
{question}

Does the CONTEXT contain enough relevant information to answer the QUESTION?
Respond with exactly one word: 'yes' or 'no'.
Do not include any other text, explanation, or punctuation.
"""
        try:
            response = self.llm.invoke(prompt)
            verdict = response.content.strip().lower()
            print(f"Document Relevance Verdict: {verdict}")
            
            if "yes" in verdict:
                return {
                    "web_search_required": False,
                    "steps": steps
                }
            else:
                print("All retrieved documents graded as IRRELEVANT. Triggering Web Search.")
                return {
                    "web_search_required": True,
                    "steps": steps
                }
        except Exception as e:
            print(f"Error grading documents: {e}. Defaulting to generate.")
            return {
                "web_search_required": False,
                "steps": steps
            }

    def node_web_search(self, state: AgentState) -> Dict[str, Any]:
        print("\n--- AGENT: PERFORMING WEB SEARCH FALLBACK ---")
        question = state["question"]
        steps = state.get("steps", [])
        steps.append("Performed web search fallback for supplementary context")

        # Clean the query for better search matches
        search_query = re.sub(r"^(who (was|is)|what (is|are|was)|how to|explain|tell me about|can you)\s+", "", question, flags=re.IGNORECASE)
        search_query = search_query.strip("?. ")
        print(f"Original query: '{question}' -> Cleaned search query: '{search_query}'")

        web_docs = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=3))
                
                # If auto backend fails, try html backend
                if not results:
                    results = list(ddgs.text(search_query, backend="html", max_results=3))
                
                for r in results:
                    web_docs.append(
                        Document(
                            page_content=r.get("body", ""),
                            metadata={
                                "source": r.get("href", "web"),
                                "pages": "web",
                                "title": r.get("title", "Web Source")
                            }
                        )
                    )
            print(f"Web search completed successfully. Found {len(web_docs)} web snippets.")
        except Exception as e:
            print(f"Error calling DuckDuckGo Search: {e}")

        # Keep original docs but append the new web docs
        current_docs = state.get("documents", [])
        return {
            "documents": current_docs + web_docs,
            "steps": steps
        }

    def node_generate(self, state: AgentState) -> Dict[str, Any]:
        print("\n--- AGENT: GENERATING ANSWER ---")
        question = state["question"]
        docs = state["documents"]
        steps = state.get("steps", [])
        steps.append("Generating final response with context")

        context_parts = []
        sources = []
        seen_keys = set()

        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            pages = doc.metadata.get("pages", "unknown")
            title = doc.metadata.get("title", "")
            
            context_parts.append(
                f"[Source {i} - {source}, page/location: {pages}]\n{doc.page_content}"
            )
            
            # Format sources for UI return
            source_key = f"{source}::{pages}"
            if source_key not in seen_keys:
                seen_keys.add(source_key)
                sources.append({
                    "source": source,
                    "pages": pages,
                    "title": title
                })

        context = "\n\n".join(context_parts)

        prompt = f"""
You are an academic study assistant.

Answer the user's question clearly, thoroughly, and objectively based ONLY on the provided context below.

Rules:
1. Use ONLY the provided context. If information is not in the context, do not make things up.
2. Do NOT include inline citations or bracketed sources (like [Source 1] or [Page 1]) in your text.
3. If the answer is not present in the context, reply exactly:
Information not found in notes.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
        try:
            response = self.llm.invoke(prompt)
            generation = response.content.strip()
        except Exception as e:
            print(f"Generation failed: {e}")
            generation = "Error generating answer."

        return {
            "generation": generation,
            "sources": sources,
            "steps": steps
        }

    # --- Routing / Conditional Edge Implementations ---

    def decide_next_step(self, state: AgentState) -> str:
        if state.get("web_search_required", False):
            return "web_search"
        return "generate"

    def grade_generation(self, state: AgentState) -> str:
        print("\n--- AGENT: CHECKING FOR HALLUCINATIONS ---")
        generation = state["generation"]
        docs = state["documents"]
        steps = state.get("steps", [])

        # If answer is default "Information not found", no need to check hallucination
        if "information not found" in generation.lower() or "error generating" in generation.lower():
            return "max_attempts_reached"

        context = "\n\n".join([d.page_content for d in docs])
        
        prompt = f"""
You are an academic evaluator checking for hallucinations.

SUPPORTING CONTEXT:
{context}

GENERATED ANSWER:
{generation}

Is the GENERATED ANSWER fully grounded in and supported by the SUPPORTING CONTEXT?
Respond with exactly 'yes' or 'no'.
Do not write any explanation, introduction, or punctuation.
"""
        try:
            response = self.llm.invoke(prompt)
            verdict = response.content.strip().lower()
            print(f"Hallucination Grader Verdict: {verdict}")

            # Check if web search fallback has already run to avoid infinite loop
            has_web_search = any(d.metadata.get("pages") == "web" for d in docs)

            if "yes" in verdict:
                steps.append("Fact check passed: Answer is fully grounded in context.")
                return "grounded"
            else:
                if not has_web_search:
                    print("Generation contains potential hallucinations. Forcing Web Search fallback.")
                    steps.append("Fact check failed: Answer contained ungrounded claims. Rerouting to Web Search.")
                    return "hallucinating_fallback"
                else:
                    print("Generation contains potential hallucinations, but web search has already run. Returning best effort.")
                    steps.append("Fact check warning: Some claims might not be fully grounded, but search fallbacks exhausted.")
                    return "max_attempts_reached"
        except Exception as e:
            print(f"Error grading generation: {e}")
            return "max_attempts_reached"

    def invoke(self, question: str) -> Dict[str, Any]:
        """
        Runs the compiled graph for the given question.
        Returns a dictionary with: answer, sources, and agent steps.
        """
        initial_state = {
            "question": question,
            "documents": [],
            "generation": "",
            "web_search_required": False,
            "sources": [],
            "steps": []
        }
        final_state = self.graph.invoke(initial_state)
        return {
            "answer": final_state["generation"],
            "sources": final_state["sources"],
            "steps": final_state["steps"]
        }
