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
    """
    RAG 服务类，负责协调和执行检索增强生成（RAG）管道的所有步骤。
    包括：查询重写 -> 检索 -> 过滤/打分 -> 检索纠错（可选）-> 生成 -> 幻觉校验。
    """
    settings: DemoSettings

    def answer(self, question: str, *, topic_hint: str | None = None) -> RagResult:
        """
        根据用户提出的问题，执行完整的 RAG 流程并返回生成的答案。

        参数:
            question: 用户的原始输入问题。
            topic_hint: 可选的主题提示，用于辅助检索时的上下文信息。

        返回:
            RagResult: 包含最终答案、检索到的文档、状态评估及调试信息的对象。
        """
        pipeline_steps: list[dict] = []
        
        # 步骤 1: 查询重写（将口语化或不完整的问题转化为更适合检索的关键词/查询句）
        rewritten_question = _timed_step(
            pipeline_steps,
            "rewrite_question",
            lambda: self._rewrite_question(question, topic_hint=topic_hint).rewritten_question,
        )
        
        # 步骤 2: 文档检索（从向量数据库或混合检索中召回候选文档，并去重）
        candidates = _timed_step(
            pipeline_steps,
            "retrieve",
            lambda: dedupe_retrieved_sources(
                self.retrieve(rewritten_question, topic_hint=topic_hint),
                limit=self.settings.agent_top_k,
            ),
        )
        
        # 步骤 3: 文档打分与相关性评估
        grades = _timed_step(
            pipeline_steps,
            "grade_documents",
            lambda: self.grade_documents(rewritten_question, candidates, attempt=0),
        )
        
        # 提取评分被标记为相关的文档 ("yes")
        docs = [doc for doc, grade in zip(candidates, grades) if grade.binary_score == "yes"]
        
        # CRAG (Corrective RAG) 纠错机制:
        # 如果检索到的所有文档都被拒绝（不相关），则利用已拒绝的文档作为反面教材，重新重写查询，并再次进行检索和打分。
        # 设定重试上限次数，防止无限循环或产生过高 LLM 开销。
        for attempt in range(1, self.settings.agent_corrective_retries + 1):
            if docs or not candidates:
                break
            # 使用已拒绝的文档对查询进行纠错式重写
            rewritten_question = _timed_step(
                pipeline_steps,
                "corrective_rewrite",
                lambda: self._rewrite_question(
                    question,
                    topic_hint=topic_hint,
                    rejected_docs=candidates,
                ).rewritten_question,
            )
            # 重新检索
            candidates = _timed_step(
                pipeline_steps,
                "retrieve_retry",
                lambda: dedupe_retrieved_sources(
                    self.retrieve(rewritten_question, topic_hint=topic_hint),
                    limit=self.settings.agent_top_k,
                ),
            )
            # 重新打分
            grades = _timed_step(
                pipeline_steps,
                "grade_documents_retry",
                lambda: self.grade_documents(rewritten_question, candidates, attempt=attempt),
            )
            docs = [doc for doc, grade in zip(candidates, grades) if grade.binary_score == "yes"]
            
        # 步骤 4: 评估可用证据是否充足
        evidence_decision = self._decide_evidence(rewritten_question, candidates, docs, grades)
        _annotate_last_step(
            pipeline_steps,
            status=evidence_decision.status,
            output_summary=f"accepted={len(docs)}, rejected={len(candidates) - len(docs)}",
            blocked_reason=evidence_decision.reason if evidence_decision.status == "insufficient_evidence" else None,
        )
        
        # 构建用于调试/前端展示的追溯信息（trace）
        debug_trace = self._build_debug_trace(
            question,
            docs,
            evidence_decision,
            candidate_docs=candidates,
            grades=grades,
            pipeline_steps=pipeline_steps,
            rewritten_question=rewritten_question,
        )
        
        # 如果证据不足，直接返回拒绝回答的预设提示，不再调用生成模块
        if evidence_decision.status == "insufficient_evidence":
            return RagResult(
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                retrieved_docs=[],
                answer_status="insufficient_evidence",
                evidence_decision=evidence_decision,
                debug_trace=debug_trace,
            )

        # 步骤 5: 基于召回的关联文档生成回答
        answer = _strip_generated_references(
            _timed_step(
                pipeline_steps,
                "generate",
                lambda: self._generate_answer(rewritten_question, docs),
            ),
            docs,
        )
        _annotate_last_step(pipeline_steps, status="grounded", output_summary=_summarize_text(answer))
        
        # 检查生成内容中是否包含类似“无法回答/未找到依据”的兜底话术
        if _looks_like_insufficient_answer(answer):
            debug_trace["generation_warning"] = "llm_refused_after_grounded_retrieval"
            
        # 在生成答案末尾附加上格式化后的引用文档列表
        answer = f"{answer.rstrip()}\n\n{format_references(docs)}"
        
        # 步骤 6: 幻觉检测（确保生成的答案有确凿的检索文本支撑，无凭空捏造）
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
        
        # 如果幻觉检测失败，则将其视为“证据不足/无法支撑回答”，退回到兜底答案
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
        """
        执行检索。根据系统配置，选用稠密向量检索（dense）或稠密+稀疏混合检索（hybrid）。

        参数:
            question: 用于检索的查询问题。
            topic_hint: 可选的主题提示词。

        返回:
            list[RetrievedDoc]: 检索到的候选文档列表。
        """
        if self.settings.agent_retrieval_strategy == "hybrid":
            return self._search_hybrid(question, topic_hint=topic_hint)
        return self._search_dense(question, topic_hint=topic_hint)

    def _search_dense(self, question: str, *, topic_hint: str | None = None) -> list[RetrievedDoc]:
        """
        利用向量数据库（Qdrant）进行稠密向量相似度检索。

        参数:
            question: 检索问题。
            topic_hint: 可选的主题提示。

        返回:
            list[RetrievedDoc]: 命中相似性评分的文档列表。
        """
        if not self.settings.embedding_configured:
            return []
        try:
            from langchain_qdrant import QdrantVectorStore
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "Install qdrant-client and langchain-qdrant from requirements.txt."
            ) from exc

        # 初始化 Embedding 编码器与 Qdrant 客户端
        embeddings = get_embeddings(self.settings)
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            url=self.settings.qdrant_url,
            collection_name=self.settings.qdrant_collection,
        )
        # 将主题提示词与提问拼接作为最终的检索输入，提升检索的领域相关性
        query = f"{topic_hint}\n{question}" if topic_hint else question
        results = vector_store.similarity_search_with_score(query, k=self.settings.agent_top_k)
        docs: list[RetrievedDoc] = []
        for document, score in results:
            metadata = document.metadata
            docs.append(
                RetrievedDoc(
                    chunk_id=str(metadata.get("chunk_id") or ""),
                    source_title=str(metadata.get("source_title") or "unknown"),
                    source_url=str(metadata.get("source_url") or ""),
                    chunk_text=str(metadata.get("context_text") or document.page_content),
                    score=float(score),
                    vector_score=float(score),
                    final_score=float(score),
                    retrieval_source="dense",
                    product=metadata.get("product"),
                )
            )
        # 按相似度得分降序排列
        docs.sort(key=lambda item: item.score, reverse=True)
        return docs

    def _search_hybrid(self, question: str, *, topic_hint: str | None = None) -> list[RetrievedDoc]:
        """
        混合检索。合并稠密向量检索与基于 BM25 的本地稀疏文本检索（Sparse），提高长尾及专业词汇的召回率。

        参数:
            question: 检索问题。
            topic_hint: 可选主题提示。

        返回:
            list[RetrievedDoc]: 经混合打分与重排后的前 N 个文档。
        """
        from customer_agent_demo.agent.hybrid import HybridRetriever, LocalSparseRetriever, dense_docs_to_hits

        query = f"{topic_hint}\n{question}" if topic_hint else question
        try:
            dense_docs = self._search_dense(question, topic_hint=topic_hint)
        except Exception:
            dense_docs = []
        # 获取稀疏检索结果（本地倒排索引）
        sparse_hits = LocalSparseRetriever().search(query, top_k=self.settings.agent_top_k * 3)
        # 使用归一化加权融合对 Dense 和 Sparse 的结果重排。
        fused = HybridRetriever(alpha=self.settings.agent_fusion_alpha).fuse(dense_docs_to_hits(dense_docs), sparse_hits)
        return [hit.doc for hit in fused[: self.settings.agent_top_k]]

    def grade_documents(self, question: str, docs: list[RetrievedDoc], *, attempt: int = 0) -> list[DocumentGrade]:
        """
        对召回的文档进行批量相关性评估评分。

        参数:
            question: 待匹配的问题。
            docs: 被评价的文档列表。
            attempt: 当前是 RAG 重试的第几次。

        返回:
            list[DocumentGrade]: 各个文档的评估打分结果。
        """
        return [self._grade_document(question, doc, attempt=attempt) for doc in docs]

    def _grade_document(self, question: str, doc: RetrievedDoc, *, attempt: int) -> DocumentGrade:
        """
        评判单个文档与用户提问的相关度。
        若得分过低，直接过滤；若配置了 LLM Grading 且可用，则首选 LLM 进行判定；若 LLM 失败或未启用，则回退到启发式文本词频覆盖度校验。
        """
        score = doc.final_score if doc.final_score is not None else doc.score
        # 启发式过滤：低于设定的最小相关度阈值，则直接判定为不相关
        if score < self.settings.agent_min_relevance_score:
            return _document_grade(doc, "no", "score_below_min_relevance", "retrieval_mismatch", attempt=attempt)
        
        # 1. 尝试使用大语言模型（LLM）进行智能相关度评判
        if self.settings.llm_configured and self.settings.agent_llm_graders_enabled:
            try:
                grade = self._llm_document_grade(question, doc)
                return _document_grade(
                    doc, grade.binary_score, grade.reason,
                    None if grade.binary_score == "yes" else "retrieval_mismatch",
                    grader="llm", attempt=attempt,
                )
            except Exception:
                # 若大模型调用出错（网络或配额问题），不能直接使系统崩溃，降级到启发式校验
                pass
                
        # 2. 启发式双重保险：计算关键词覆盖度和重合度
        overlap, coverage = _keyword_overlap_coverage(question, doc.chunk_text)
        if overlap == 0 or coverage < 0.25:
            return _document_grade(doc, "no", "heuristic_insufficient_query_coverage", "retrieval_mismatch", attempt=attempt)
        return _document_grade(doc, "yes", "heuristic_relevance_fallback", None, attempt=attempt)

    def check_hallucination(self, answer: str, docs: list[RetrievedDoc]) -> HallucinationDecision:
        """
        检测生成的回答是否包含幻觉（没有可信检索证据支持的捏造信息）。
        支持格式检查、LLM 幻觉判定以及数值一致性硬过滤。

        参数:
            answer: LLM 生成的最终答案（含引用信息）。
            docs: 用于做比对支持的参考文档集合。

        返回:
            HallucinationDecision: 幻觉检测结论（通过/未通过、具体原因和不可靠内容）。
        """
        # 1. 检查生成的回答中是否包含强制要求的“引用：”标签及格式规范
        has_reference_line = any(REFERENCE_LINE_PATTERN.match(line) for line in answer.splitlines())
        if "引用：" not in answer or not has_reference_line:
            return HallucinationDecision(status="failed", reason="answer_missing_required_references", failure_type="format_unstable")

        evidence_text = "\n".join(doc.chunk_text for doc in docs)
        answer_body = REFERENCE_SECTION_PATTERN.sub("", answer).strip()
        
        # 2. 尝试使用 LLM 进行深层次的语义蕴含与事实一致性校验（Grounding Check）
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
                # 降级到启发式规则校验
                pass
                
        # 3. 启发式数值硬过滤：数字在事实陈述中极其关键，若回答中含有检索文本中从未出现过的数字，则判定为幻觉风险
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
        """
        大语言模型辅助重写问题，以获取更好的检索召回效果。
        如果是 corrective_rewrite（纠错重写），会加入已拒绝的文档内容进行反向提示。
        """
        stripped = question.strip()
        if not self.settings.llm_configured:
            context = f"{topic_hint}\n" if topic_hint and topic_hint.lower() not in stripped.lower() else ""
            return QueryRewrite(rewritten_question=f"{context}{stripped}", reason="contextualized_query_fallback")
        # 限制传入的反面教材字符数，防止上下文溢出
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
        """
        判定当前的知识库检索结果是否足以支撑问题的生成。

        参数:
            question: 待回答问题。
            candidate_docs: 所有召回的候选文档。
            accepted_docs: 经过相关度评估被认可的相关文档。
            grades: 所有的评分记录。

        返回:
            EvidenceDecision: 最终判断是否可信并具备足够证据。
        """
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
        """
        检测已被采纳的文档库中最优相似度分值，判定其是否高于预设的最低相关度界限。
        """
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
        """
        组装管道中所有的中间态信息，构造成易于读取的 trace 日志，便于后续链路分析与评估。
        """
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
                    "chunk_id": doc.chunk_id,
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
                    "chunk_id": doc.chunk_id,
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
        """
        利用大语言模型（LLM）基于上下文（检索到的文档）生成事实性的回答。
        若模型未配置，则回退到直接把相关文本拼装成简版回答。
        """
        if not self.settings.llm_configured:
            return _fallback_grounded_answer(docs)

        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        # 整理上下文文档列表，包含标题、源链接、分块编号和具体内容，输入至 LLM System Prompt 中
        context = "\n\n".join(
            f"[{index}] {doc.source_title}\nURL: {doc.source_url}\n"
            f"id: {doc.chunk_id or 'unknown'}\n{doc.chunk_text}"
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
        """
        初始化一个面向结构化输出的 Chat 实例，硬编码温控度（temperature）为 0 保证输出的高一致性与低随机性。
        """
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
        """
        调用 LLM 对文档块与提问的实际相关性进行结构化输出判定。
        """
        chat = self._structured_chat(max_tokens=250)
        prompt = load_prompt("rag_document_grader.md").format(question=question, document=doc.chunk_text)
        return chat.with_structured_output(RelevanceGrade).invoke(prompt)

    def _llm_grounding_grade(self, answer: str, evidence: str) -> GroundingGrade:
        """
        调用 LLM 判断生成的回答是否可以被给定的参考证据（文档）充分蕴含支撑（避免语义级别的幻觉问题）。
        """
        chat = self._structured_chat(max_tokens=350)
        prompt = load_prompt("rag_grounding_grader.md").format(answer=answer, evidence=evidence)
        return chat.with_structured_output(GroundingGrade).invoke(prompt)


def format_references(docs: list[RetrievedDoc]) -> str:
    """
    格式化引用文献列表为标准文本输出。
    格式: [序号] 标题 - 网址 - chunk #索引号
    """
    lines = ["引用："]
    for index, doc in enumerate(dedupe_retrieved_sources(docs), start=1):
        lines.append(f"[{index}] {doc.source_title} - {doc.source_url}")
    return "\n".join(lines)


def dedupe_retrieved_sources(docs: list[RetrievedDoc], *, limit: int | None = None) -> list[RetrievedDoc]:
    """
    对召回的文档集合进行来源去重。
    如果同一个来源（URL/标题）返回了多个分块（chunk），只保留分值最高的那一个分块，最后根据相关度分数从高到低重新排序。

    参数:
        docs: 输入的待去重文档列表。
        limit: 限制最终返回的最大文档数。
    """
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
    """
    计算并在 steps 列表中记录单个管道步骤的执行时长、状态以及阶段结果简述。
    """
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
    """
    更新步骤列表中最后一项的运行状态、结论摘要及阻塞原因。
    """
    if not steps:
        return
    steps[-1]["status"] = status
    if output_summary is not None:
        steps[-1]["output_summary"] = output_summary
    if blocked_reason:
        steps[-1]["blocked_reason"] = blocked_reason


def _summarize_value(value: object) -> str:
    """
    为了在 Trace 记录中保持简洁，将复杂对象、列表或字符串格式化为单行极短摘要。
    """
    if isinstance(value, str):
        return _summarize_text(value)
    if isinstance(value, list):
        return f"items={len(value)}"
    return type(value).__name__


def _summarize_text(text: str) -> str:
    """
    将换行符等多余空白字符替换为空格，并将长文本截断至 120 字符用于可视化日志。
    """
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
    """
    构建并返回一个 DocumentGrade 数据类实例，保存该文档的打分属性。
    """
    return DocumentGrade(
        source_title=doc.source_title,
        source_url=doc.source_url,
        binary_score=binary_score,  # type: ignore[arg-type]
        reason=reason,
        failure_type=failure_type,  # type: ignore[arg-type]
        score=_ranking_score(doc),
        grader=grader,  # type: ignore[arg-type]
        attempt=attempt,
    )


def _fallback_grounded_answer(docs: list[RetrievedDoc]) -> str:
    """
    生成兜底答案。当大模型未配置时，直接抽取前 2 篇最相关的文本分块合成应答。
    """
    lines = ["基于当前知识库命中的资料，可以确认："]
    for doc in docs[:2]:
        lines.append(f"- {doc.chunk_text}")
    return "\n".join(lines)


def _normalize_title(t: str) -> str:
    """
    规范化标题，移除所有标点符号及空白字符，全部转为小写。
    """
    return re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "", t).lower()


def _strip_generated_references(answer: str, docs: list[RetrievedDoc] | None = None) -> str:
    """
    去除 LLM 输出末尾可能自行生成的冗余引用文献格式，以便使用我们规范的 `format_references` 生成。
    """
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
    """
    判断生成的文本中是否包含“未找到依据”、“无法回答”等大模型主动拒绝回答的话术。
    """
    return INSUFFICIENT_ANSWER_PATTERN.search(answer) is not None


def _ranking_score(doc: RetrievedDoc) -> float:
    """
    获取排序分数，优先选用融合后计算得到的 `final_score`，没有则使用原始检索的 `score`。
    """
    return doc.final_score if doc.final_score is not None else doc.score


def _source_identity(doc: RetrievedDoc) -> tuple[str, str]:
    """
    唯一标识一个文档的源。若有 URL 则使用 URL 标识，否则将标题转为小写无空格后标识。
    """
    if doc.source_url:
        return ("url", doc.source_url.strip())
    return ("title", re.sub(r"\s+", "", doc.source_title).strip().lower())


def _keyword_overlap(question: str, evidence: str) -> int:
    """
    计算提问词与文本间的关键词重合个数。
    """
    return _keyword_overlap_coverage(question, evidence)[0]


def _keyword_overlap_coverage(question: str, evidence: str) -> tuple[int, float]:
    """
    计算提问词与证据文本的词汇重合度。
    英文采用基于空白/字母的分词，中文采用二元组（bigram）切分以消除单字高误报。
    """
    eng_tokens = re.findall(r"[a-zA-Z0-9]+", question)
    han_chars = re.findall(r"[\u4e00-\u9fff]", question)
    # 为中文构建二元语法结构 (bigrams)，避免像 “在” “是” 这样的单个常见汉字导致启发式匹配的高误报率。
    han_bigrams = [han_chars[i] + han_chars[i+1] for i in range(len(han_chars)-1)]
    tokens = set(eng_tokens) | set(han_bigrams)
    # 过滤停用词以提高关键词提取的纯净度
    question_tokens = {t.lower() for t in tokens if t not in STOPWORDS}
    evidence_lower = evidence.lower()
    overlap = sum(1 for token in question_tokens if token in evidence_lower)
    return overlap, overlap / len(question_tokens) if question_tokens else 0.0


def _is_number_supported(number_str: str, evidence_text: str) -> bool:
    """
    启发式数字匹配规则：检测生成答案中的数字（如产品参数、时限、比例等）是否能在检索到的证据文本中被验证。
    """
    val = number_str.strip()
    if not val:
        return True

    # 尝试提取出纯数值核心部分（整数/小数）以允许带单位比对，如 "2 小时" 与 "2" 匹配
    num_match = re.match(r"^\d+(?:\.\d+)?", val)
    if not num_match:
        return True

    core_num = num_match.group(0)
    if core_num in evidence_text:
        return True

    # 年份映射：如果是4位数字组成的年份（如2023年），兼容检测其2位简称（23年）是否存在于原文中。
    if len(core_num) == 4 and core_num.isdigit() and (core_num.startswith("19") or core_num.startswith("20")):
        short_year = core_num[2:]
        if short_year in evidence_text:
            return True

    return False
