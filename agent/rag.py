from __future__ import annotations

from dataclasses import dataclass
import re
from time import perf_counter
from typing import Callable, TypeVar

from customer_agent_demo.agent.embeddings import get_embeddings
from customer_agent_demo.agent.models import (
    DocumentGrade,
    EvidenceDecision,
    GroundingGrade,
    HallucinationDecision,
    QueryRewrite,
    RagResult,
    RelevanceGrade,
    RetrievedDoc,
)
from customer_agent_demo.agent.prompts import load_prompt
from customer_agent_demo.config import DemoSettings


REFERENCE_SECTION_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:引用|引用列表|参考|参考资料|参考来源|资料来源|来源)\s*(?:如下|列表)?\s*[:：]?\s*\n?[\s\S]*$",
    re.IGNORECASE,
)
REFERENCE_LINE_PATTERN = re.compile(r"^\s*\[\d+\]\s+.+(?:https?://|chunk\s*#|#\d+).*$", re.IGNORECASE)
INSUFFICIENT_ANSWER_PATTERN = re.compile(
    r"(没有在当前知识库找到足够依据|没有足够依据|无相关数据支持|知识库中无相关|未找到.*依据|无法.*确认|无法.*回答)"
)
INSUFFICIENT_EVIDENCE_ANSWER = "我没有在当前知识库找到足够依据。为了避免误导，我先不编造答案。你可以补充设备型号、使用场景或问题细节，我再继续帮你查。"
RISKY_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*(?:位|天|小时|分钟|米|feet|ft|%|℃|°c|mg/dl|mmol/l)?", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]{2,}")
STOPWORDS = {"可以", "能不能", "是不是", "怎么", "多少", "多久", "这个", "它", "请问", "一下"}
T = TypeVar("T")


@dataclass(slots=True)
class RagService:
    settings: DemoSettings

    def answer(self, question: str, *, topic_hint: str | None = None) -> RagResult:
        pipeline_steps: list[dict] = []
        rewritten_question = _timed_step(
            pipeline_steps,
            "rewrite_question",
            lambda: self._rewrite_question(question, topic_hint=topic_hint).rewritten_question,
        )
        candidates = _timed_step(
            pipeline_steps,
            "retrieve",
            lambda: dedupe_retrieved_sources(
                self.retrieve(rewritten_question, topic_hint=topic_hint),
                limit=self.settings.agent_top_k,
            ),
        )
        grades = _timed_step(
            pipeline_steps,
            "grade_documents",
            lambda: self.grade_documents(rewritten_question, candidates, attempt=0),
        )
        docs = [doc for doc, grade in zip(candidates, grades) if grade.binary_score == "yes"]
        # CRAG-style bounded correction: rejected retrieval feeds a query rewrite,
        # then retrieval and grading run once more.  A hard cap prevents loops/cost spikes.
        for attempt in range(1, self.settings.agent_corrective_retries + 1):
            if docs or not candidates:
                break
            rewritten_question = _timed_step(
                pipeline_steps,
                "corrective_rewrite",
                lambda: self._rewrite_question(
                    question,
                    topic_hint=topic_hint,
                    rejected_docs=candidates,
                ).rewritten_question,
            )
            candidates = _timed_step(
                pipeline_steps,
                "retrieve_retry",
                lambda: dedupe_retrieved_sources(
                    self.retrieve(rewritten_question, topic_hint=topic_hint),
                    limit=self.settings.agent_top_k,
                ),
            )
            grades = _timed_step(
                pipeline_steps,
                "grade_documents_retry",
                lambda: self.grade_documents(rewritten_question, candidates, attempt=attempt),
            )
            docs = [doc for doc, grade in zip(candidates, grades) if grade.binary_score == "yes"]
        evidence_decision = self._decide_evidence(rewritten_question, candidates, docs, grades)
        _annotate_last_step(
            pipeline_steps,
            status=evidence_decision.status,
            output_summary=f"accepted={len(docs)}, rejected={len(candidates) - len(docs)}",
            blocked_reason=evidence_decision.reason if evidence_decision.status == "insufficient_evidence" else None,
        )
        debug_trace = self._build_debug_trace(
            question,
            docs,
            evidence_decision,
            candidate_docs=candidates,
            grades=grades,
            pipeline_steps=pipeline_steps,
            rewritten_question=rewritten_question,
        )
        if evidence_decision.status == "insufficient_evidence":
            return RagResult(
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                retrieved_docs=[],
                answer_status="insufficient_evidence",
                evidence_decision=evidence_decision,
                debug_trace=debug_trace,
            )

        answer = _strip_generated_references(
            _timed_step(
                pipeline_steps,
                "generate",
                lambda: self._generate_answer(rewritten_question, docs),
            ),
            docs,
        )
        _annotate_last_step(pipeline_steps, status="grounded", output_summary=_summarize_text(answer))
        if _looks_like_insufficient_answer(answer):
            debug_trace["generation_warning"] = "llm_refused_after_grounded_retrieval"
        answer = f"{answer.rstrip()}\n\n{format_references(docs)}"
        hallucination_decision = _timed_step(
            pipeline_steps,
            "hallucination_check",
            lambda: self.check_hallucination(answer, docs),
        )
        _annotate_last_step(
            pipeline_steps,
            status=hallucination_decision.status,
            output_summary=hallucination_decision.reason,
            blocked_reason=hallucination_decision.failure_type if hallucination_decision.status == "failed" else None,
        )
        debug_trace["pipeline_steps"] = pipeline_steps
        debug_trace["hallucination_decision"] = hallucination_decision.model_dump()
        if hallucination_decision.status == "failed":
            final_decision = EvidenceDecision(
                status="insufficient_evidence",
                reason=hallucination_decision.failure_type or hallucination_decision.reason,
                top_score=evidence_decision.top_score,
            )
            debug_trace["evidence_status"] = final_decision.status
            debug_trace["evidence_reason"] = final_decision.reason
            return RagResult(
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                retrieved_docs=[],
                answer_status="insufficient_evidence",
                evidence_decision=final_decision,
                debug_trace=debug_trace,
            )
        return RagResult(
            answer=answer,
            retrieved_docs=docs,
            answer_status="grounded",
            evidence_decision=evidence_decision,
            debug_trace=debug_trace,
        )

    def retrieve(self, question: str, *, topic_hint: str | None = None) -> list[RetrievedDoc]:
        if self.settings.agent_retrieval_strategy == "hybrid":
            return self._search_hybrid(question, topic_hint=topic_hint)
        return self._search_dense(question, topic_hint=topic_hint)

    def _search_dense(self, question: str, *, topic_hint: str | None = None) -> list[RetrievedDoc]:
        if not self.settings.embedding_configured:
            return []
        try:
            from langchain_qdrant import QdrantVectorStore
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "Install qdrant-client and langchain-qdrant from requirements.txt."
            ) from exc

        embeddings = get_embeddings(self.settings)
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            url=self.settings.qdrant_url,
            collection_name=self.settings.qdrant_collection,
        )
        query = f"{topic_hint}\n{question}" if topic_hint else question
        results = vector_store.similarity_search_with_score(query, k=self.settings.agent_top_k)
        docs: list[RetrievedDoc] = []
        for document, score in results:
            metadata = document.metadata
            docs.append(
                RetrievedDoc(
                    source_title=str(metadata.get("source_title") or "unknown"),
                    source_url=str(metadata.get("source_url") or ""),
                    chunk_index=int(metadata.get("chunk_index") or 0),
                    chunk_text=str(metadata.get("chunk_text") or document.page_content),
                    score=float(score),
                    vector_score=float(score),
                    final_score=float(score),
                    retrieval_source="dense",
                    product=metadata.get("product"),
                )
            )
        docs.sort(key=lambda item: item.score, reverse=True)
        return docs

    def _search_hybrid(self, question: str, *, topic_hint: str | None = None) -> list[RetrievedDoc]:
        from customer_agent_demo.agent.hybrid import HybridRetriever, LocalSparseRetriever, dense_docs_to_hits

        query = f"{topic_hint}\n{question}" if topic_hint else question
        try:
            dense_docs = self._search_dense(question, topic_hint=topic_hint)
        except Exception:
            dense_docs = []
        sparse_hits = LocalSparseRetriever().search(query, top_k=self.settings.agent_top_k * 3)
        fused = HybridRetriever(alpha=self.settings.agent_fusion_alpha).fuse(dense_docs_to_hits(dense_docs), sparse_hits)
        return [hit.doc for hit in fused[: self.settings.agent_top_k]]

    def grade_documents(self, question: str, docs: list[RetrievedDoc], *, attempt: int = 0) -> list[DocumentGrade]:
        return [self._grade_document(question, doc, attempt=attempt) for doc in docs]

    def _grade_document(self, question: str, doc: RetrievedDoc, *, attempt: int) -> DocumentGrade:
        score = doc.final_score if doc.final_score is not None else doc.score
        if score < self.settings.agent_min_relevance_score:
            return _document_grade(doc, "no", "score_below_min_relevance", "retrieval_mismatch", attempt=attempt)
        if self.settings.llm_configured and self.settings.agent_llm_graders_enabled:
            try:
                grade = self._llm_document_grade(question, doc)
                return _document_grade(
                    doc, grade.binary_score, grade.reason,
                    None if grade.binary_score == "yes" else "retrieval_mismatch",
                    grader="llm", attempt=attempt,
                )
            except Exception:
                # Availability failures must not turn into unsupported answers.
                pass
        overlap, coverage = _keyword_overlap_coverage(question, doc.chunk_text)
        if overlap == 0 or coverage < 0.25:
            return _document_grade(doc, "no", "heuristic_insufficient_query_coverage", "retrieval_mismatch", attempt=attempt)
        return _document_grade(doc, "yes", "heuristic_relevance_fallback", None, attempt=attempt)

    def check_hallucination(self, answer: str, docs: list[RetrievedDoc]) -> HallucinationDecision:
        has_reference_line = any(REFERENCE_LINE_PATTERN.match(line) for line in answer.splitlines())
        if "引用：" not in answer or not has_reference_line:
            return HallucinationDecision(status="failed", reason="answer_missing_required_references", failure_type="format_unstable")

        evidence_text = "\n".join(doc.chunk_text for doc in docs)
        answer_body = REFERENCE_SECTION_PATTERN.sub("", answer).strip()
        if self.settings.llm_configured and self.settings.agent_llm_graders_enabled:
            try:
                grade = self._llm_grounding_grade(answer_body, evidence_text)
                if not grade.grounded:
                    return HallucinationDecision(
                        status="failed", reason=grade.reason, failure_type="hallucination",
                        unsupported_claims=grade.unsupported_claims, grader="llm",
                    )
                return HallucinationDecision(status="grounded", reason=grade.reason, grader="llm")
            except Exception:
                pass
        unsupported_numbers = [
            number
            for number in RISKY_NUMBER_PATTERN.findall(answer_body)
            if number.strip() and not _is_number_supported(number, evidence_text)
        ]
        if unsupported_numbers:
            return HallucinationDecision(
                status="failed",
                reason="answer_contains_numbers_not_supported_by_evidence",
                failure_type="hallucination",
                risky_numbers=unsupported_numbers, grader="heuristic",
            )
        return HallucinationDecision(status="grounded", reason="heuristic_grounding_fallback", grader="heuristic")

    def _rewrite_question(
        self, question: str, *, topic_hint: str | None = None, rejected_docs: list[RetrievedDoc] | None = None,
    ) -> QueryRewrite:
        stripped = question.strip()
        if not self.settings.llm_configured:
            context = f"{topic_hint}\n" if topic_hint and topic_hint.lower() not in stripped.lower() else ""
            return QueryRewrite(rewritten_question=f"{context}{stripped}", reason="contextualized_query_fallback")
        rejected_context = "\n\n".join(doc.chunk_text[:600] for doc in (rejected_docs or []))
        try:
            chat = self._structured_chat(max_tokens=300)
            return chat.with_structured_output(QueryRewrite).invoke(
                load_prompt("rag_rewrite.md").format(question=stripped, topic_hint=topic_hint or "", rejected_context=rejected_context)
            )
        except Exception:
            return QueryRewrite(rewritten_question=stripped, reason="rewrite_model_unavailable")

    def _decide_evidence(
        self,
        question: str,
        candidate_docs: list[RetrievedDoc],
        accepted_docs: list[RetrievedDoc],
        grades: list[DocumentGrade],
    ) -> EvidenceDecision:
        if not candidate_docs:
            return EvidenceDecision(status="insufficient_evidence", reason="knowledge_missing", top_score=None)
        if not accepted_docs:
            top_score = _ranking_score(candidate_docs[0])
            reason = "retrieval_mismatch"
            if all(grade.failure_type == "retrieval_mismatch" for grade in grades):
                reason = "retrieval_mismatch"
            return EvidenceDecision(status="insufficient_evidence", reason=reason, top_score=top_score)
        return self._has_sufficient_evidence(question, accepted_docs)

    def _has_sufficient_evidence(self, question: str, docs: list[RetrievedDoc]) -> EvidenceDecision:
        if not docs:
            return EvidenceDecision(
                status="insufficient_evidence",
                reason="no_retrieved_docs",
                top_score=None,
            )

        top_doc = docs[0]
        top_score = top_doc.final_score if top_doc.final_score is not None else top_doc.score
        if top_score < self.settings.agent_min_relevance_score:
            return EvidenceDecision(
                status="insufficient_evidence",
                reason=f"top_score_below_min_score:{top_score:.3f}<{self.settings.agent_min_relevance_score:.3f}",
                top_score=top_score,
            )

        return EvidenceDecision(status="grounded", reason="graded_evidence_available", top_score=top_score)

    def _build_debug_trace(
        self,
        question: str,
        docs: list[RetrievedDoc],
        evidence_decision: EvidenceDecision,
        *,
        candidate_docs: list[RetrievedDoc] | None = None,
        grades: list[DocumentGrade] | None = None,
        pipeline_steps: list[dict] | None = None,
        rewritten_question: str | None = None,
    ) -> dict:
        candidate_docs = candidate_docs if candidate_docs is not None else docs
        grades = grades or []
        return {
            "question": question,
            "rewritten_question": rewritten_question or question,
            "retrieval_strategy": self.settings.agent_retrieval_strategy,
            "top_k": self.settings.agent_top_k,
            "min_score": self.settings.agent_min_relevance_score,
            "fusion_alpha": self.settings.agent_fusion_alpha if self.settings.agent_retrieval_strategy == "hybrid" else None,
            "evidence_status": evidence_decision.status,
            "evidence_reason": evidence_decision.reason,
            "document_grades": [grade.model_dump() for grade in grades],
            "pipeline_steps": pipeline_steps or [],
            "final_hits": [] if evidence_decision.status == "insufficient_evidence" else [
                {
                    "source_title": doc.source_title,
                    "source_url": doc.source_url,
                    "chunk_index": doc.chunk_index,
                    "score": doc.score,
                    "vector_score": doc.vector_score,
                    "sparse_score": doc.sparse_score,
                    "final_score": doc.final_score,
                    "retrieval_source": doc.retrieval_source,
                }
                for doc in docs
            ],
            "candidate_hits": [
                {
                    "source_title": doc.source_title,
                    "source_url": doc.source_url,
                    "chunk_index": doc.chunk_index,
                    "score": doc.score,
                    "vector_score": doc.vector_score,
                    "sparse_score": doc.sparse_score,
                    "final_score": doc.final_score,
                    "retrieval_source": doc.retrieval_source,
                }
                for doc in candidate_docs
            ],
        }

    def _generate_answer(self, question: str, docs: list[RetrievedDoc]) -> str:
        if not self.settings.llm_configured:
            return _fallback_grounded_answer(docs)

        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        context = "\n\n".join(
            f"[{index}] {doc.source_title}\nURL: {doc.source_url}\n"
            f"chunk #{doc.chunk_index}\n{doc.chunk_text}"
            for index, doc in enumerate(docs, start=1)
        )
        chat = ChatOpenAI(
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_api_base,
            model=self.settings.llm_model,
            temperature=self.settings.llm_temperature,
            max_tokens=min(self.settings.llm_max_tokens, 800),
            extra_body=self.settings.llm_extra_body,
        )
        message = chat.invoke(
            [
                SystemMessage(content=load_prompt("rag_answer.md").format(context=context, question=question)),
                HumanMessage(content=question),
            ]
        )
        content = message.content
        if isinstance(content, list):
            return "\n".join(item.get("text", "") for item in content if isinstance(item, dict)).strip()
        return str(content).strip()

    def _structured_chat(self, *, max_tokens: int) -> object:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_api_base,
            model=self.settings.llm_model,
            temperature=0,
            max_tokens=max_tokens,
            extra_body=self.settings.llm_extra_body,
        )

    def _llm_document_grade(self, question: str, doc: RetrievedDoc) -> RelevanceGrade:
        chat = self._structured_chat(max_tokens=250)
        prompt = load_prompt("rag_document_grader.md").format(question=question, document=doc.chunk_text)
        return chat.with_structured_output(RelevanceGrade).invoke(prompt)

    def _llm_grounding_grade(self, answer: str, evidence: str) -> GroundingGrade:
        chat = self._structured_chat(max_tokens=350)
        prompt = load_prompt("rag_grounding_grader.md").format(answer=answer, evidence=evidence)
        return chat.with_structured_output(GroundingGrade).invoke(prompt)


def format_references(docs: list[RetrievedDoc]) -> str:
    lines = ["引用："]
    for index, doc in enumerate(dedupe_retrieved_sources(docs), start=1):
        lines.append(f"[{index}] {doc.source_title} - {doc.source_url} - chunk #{doc.chunk_index}")
    return "\n".join(lines)


def dedupe_retrieved_sources(docs: list[RetrievedDoc], *, limit: int | None = None) -> list[RetrievedDoc]:
    best_by_source: dict[tuple[str, str], RetrievedDoc] = {}
    order: list[tuple[str, str]] = []
    for doc in docs:
        key = _source_identity(doc)
        existing = best_by_source.get(key)
        if existing is None:
            best_by_source[key] = doc
            order.append(key)
            continue
        if _ranking_score(doc) > _ranking_score(existing):
            best_by_source[key] = doc

    deduped = [best_by_source[key] for key in order]
    deduped.sort(key=_ranking_score, reverse=True)
    return deduped[:limit] if limit is not None else deduped


def _timed_step(steps: list[dict], name: str, fn: Callable[[], T]) -> T:
    started = perf_counter()
    try:
        result = fn()
    except Exception as exc:
        steps.append(
            {
                "name": name,
                "status": "error",
                "duration_ms": int((perf_counter() - started) * 1000),
                "blocked_reason": str(exc),
            }
        )
        raise
    steps.append(
        {
            "name": name,
            "status": "ok",
            "duration_ms": int((perf_counter() - started) * 1000),
            "output_summary": _summarize_value(result),
        }
    )
    return result


def _annotate_last_step(
    steps: list[dict],
    *,
    status: str,
    output_summary: str | None = None,
    blocked_reason: str | None = None,
) -> None:
    if not steps:
        return
    steps[-1]["status"] = status
    if output_summary is not None:
        steps[-1]["output_summary"] = output_summary
    if blocked_reason:
        steps[-1]["blocked_reason"] = blocked_reason


def _summarize_value(value: object) -> str:
    if isinstance(value, str):
        return _summarize_text(value)
    if isinstance(value, list):
        return f"items={len(value)}"
    return type(value).__name__


def _summarize_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:120]


def _document_grade(
    doc: RetrievedDoc,
    binary_score: str,
    reason: str,
    failure_type: str | None,
    *,
    grader: str = "heuristic",
    attempt: int = 0,
) -> DocumentGrade:
    return DocumentGrade(
        source_title=doc.source_title,
        source_url=doc.source_url,
        chunk_index=doc.chunk_index,
        binary_score=binary_score,  # type: ignore[arg-type]
        reason=reason,
        failure_type=failure_type,  # type: ignore[arg-type]
        score=_ranking_score(doc),
        grader=grader,  # type: ignore[arg-type]
        attempt=attempt,
    )


def _fallback_grounded_answer(docs: list[RetrievedDoc]) -> str:
    lines = ["基于当前知识库命中的资料，可以确认："]
    for doc in docs[:2]:
        lines.append(f"- {doc.chunk_text}")
    return "\n".join(lines)


def _normalize_title(t: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "", t).lower()


def _strip_generated_references(answer: str, docs: list[RetrievedDoc] | None = None) -> str:
    stripped = REFERENCE_SECTION_PATTERN.sub("", answer).strip()
    lines = stripped.splitlines()
    while lines:
        last_line = lines[-1].strip()
        if not last_line:
            lines.pop()
            continue
        if REFERENCE_LINE_PATTERN.match(last_line):
            lines.pop()
            continue
        if docs:
            match = re.match(r"^\s*\[\d+\]\s*(.+)$", last_line)
            if match:
                content = match.group(1).strip()
                content_norm = _normalize_title(content)
                if content_norm:
                    is_title_match = False
                    for doc in docs:
                        title_norm = _normalize_title(doc.source_title)
                        if title_norm and (content_norm in title_norm or title_norm in content_norm):
                            is_title_match = True
                            break
                    if is_title_match:
                        lines.pop()
                        continue
        break
    return "\n".join(lines).strip()


def _looks_like_insufficient_answer(answer: str) -> bool:
    return INSUFFICIENT_ANSWER_PATTERN.search(answer) is not None


def _ranking_score(doc: RetrievedDoc) -> float:
    return doc.final_score if doc.final_score is not None else doc.score


def _source_identity(doc: RetrievedDoc) -> tuple[str, str]:
    if doc.source_url:
        return ("url", doc.source_url.strip())
    return ("title", re.sub(r"\s+", "", doc.source_title).strip().lower())


def _keyword_overlap(question: str, evidence: str) -> int:
    return _keyword_overlap_coverage(question, evidence)[0]


def _keyword_overlap_coverage(question: str, evidence: str) -> tuple[int, float]:
    eng_tokens = re.findall(r"[a-zA-Z0-9]+", question)
    han_chars = re.findall(r"[\u4e00-\u9fff]", question)
    han_bigrams = [han_chars[i] + han_chars[i+1] for i in range(len(han_chars)-1)]
    # Single Han characters create many accidental matches (for example “在” or
    # “是”), so the offline fallback deliberately uses only meaningful bigrams.
    tokens = set(eng_tokens) | set(han_bigrams)
    question_tokens = {t.lower() for t in tokens if t not in STOPWORDS}
    evidence_lower = evidence.lower()
    overlap = sum(1 for token in question_tokens if token in evidence_lower)
    return overlap, overlap / len(question_tokens) if question_tokens else 0.0


def _is_number_supported(number_str: str, evidence_text: str) -> bool:
    val = number_str.strip()
    if not val:
        return True

    # Try to extract the core numeric part (digits and decimals)
    # This allows unit spacing like "2 小时" to match "2" or "2片" or "2"
    num_match = re.match(r"^\d+(?:\.\d+)?", val)
    if not num_match:
        return True

    core_num = num_match.group(0)
    if core_num in evidence_text:
        return True

    # Year mapping: if core_num is a 4-digit year (e.g. 2023), check if 2-digit representation (23) is in evidence
    if len(core_num) == 4 and core_num.isdigit() and (core_num.startswith("19") or core_num.startswith("20")):
        short_year = core_num[2:]
        if short_year in evidence_text:
            return True

    return False
