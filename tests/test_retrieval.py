import pytest

from docanswer.models import AccessLevel, clearance_for
from docanswer.retrieval import HybridRetriever, reciprocal_rank_fusion


@pytest.fixture(scope="module")
def retriever(index):
    return HybridRetriever(index)


@pytest.mark.parametrize("role", ["guest", "employee", "unknown-role"])
@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_access_filter_never_leaks(retriever, index, role, mode):
    """Adversarial: use every restricted chunk's own text as the query."""
    clearance = clearance_for(role)
    restricted = [c for c in index.chunks if c.access_level > clearance]
    assert restricted
    for chunk in restricted:
        result = retriever.search(chunk.text, role, k=10, mode=mode)
        assert all(s.chunk.access_level <= clearance for s in result.results)


def test_manager_can_reach_manager_documents(retriever):
    result = retriever.search("merit increase budget 2026", "manager", k=3)
    assert result.results[0].chunk.doc_id == "manager-compensation-guidelines"
    assert result.results[0].chunk.access_level == AccessLevel.MANAGERS


def test_unknown_role_fails_closed():
    assert clearance_for("admin") == AccessLevel.PUBLIC


def test_hybrid_finds_spanish_and_english_facts(retriever):
    assert retriever.search("hotel cap Bogota per night", "employee").results[0].chunk.page == 2
    top = retriever.search("¿Cuántos periodos de vacaciones puedo acumular?", "employee").results[0].chunk
    assert (top.doc_id, top.page) == ("politica-vacaciones", 2)


def test_confidence_is_low_for_out_of_corpus_questions(retriever):
    assert retriever.search("cryptocurrency investment rules", "manager").confidence < 0.3


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])
    assert [item for item, _ in fused][:2] == [1, 3]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
