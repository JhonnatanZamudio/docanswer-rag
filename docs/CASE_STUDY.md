# Case study: DocAnswer, a cited policy assistant

*This is a portfolio demo on a synthetic corpus for a fictional company, not a client project.*

**Problem.** HR and operations teams answer the same policy questions every week, while the answers sit in PDF handbooks. A generic chatbot is not acceptable here: it can invent policy, and it ignores who may see what.

**Approach.**
- Chunks keep their page number and section heading.
- Retrieval is hybrid: BM25 and vectors, merged with Reciprocal Rank Fusion.
- Each role searches only the documents it is cleared for, filtered before ranking.
- Every `[Document p.N]` citation is checked against the passages actually retrieved.
- Below a confidence threshold, calibrated on a separate dev set, the assistant refuses.

**Architecture.**
- Typed Python package with `Embedder` and `LLMProvider` protocols.
- Offline defaults, plus opt-in Anthropic or OpenAI providers.
- FastAPI service with a no-build HTML UI, packaged with Docker.
- Bilingual (EN/ES) evaluation suite.

**Result (demo corpus, 34 questions, offline extractive mode).**
- Retrieval hit@3: 1.00
- Answers citing the expected page: 0.84
- Unanswerable questions refused: 0.80
- Access-restricted questions refused: 1.00
- Access-control leaks: 0
- Tests passing: 36

The failures are documented: paraphrases, cross-lingual questions, and one off-topic answer.

**What I'd do in production.**
- Take roles and ACLs from SSO and the source system.
- Use multilingual embeddings with a reranker.
- Generate answers with an LLM and add an answerability check.
- Add OCR for scanned PDFs.
- Build the evaluation set from real employee questions.
- Monitor refusals and user feedback.
