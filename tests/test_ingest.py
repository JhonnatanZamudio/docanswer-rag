import itertools

from docanswer.ingest import DocMeta, Index, boilerplate_lines, chunk_page, sections_from_lines
from docanswer.models import AccessLevel

META = DocMeta("doc", "Doc Title", "Doc", AccessLevel.EMPLOYEES, "en")


def test_every_chunk_has_a_valid_page_and_metadata(index):
    pages = {
        "employee-handbook": 4,
        "travel-expense-policy": 4,
        "it-security-policy": 3,
        "onboarding-guide": 3,
        "politica-vacaciones": 3,
        "manager-compensation-guidelines": 3,
    }
    assert {c.doc_id for c in index.chunks} == set(pages)
    for c in index.chunks:
        assert 1 <= c.page <= pages[c.doc_id]
        assert c.chunk_id.startswith(f"{c.doc_id}:p{c.page}:")


def test_facts_land_on_their_source_page(index):
    def pages_with(fact):
        return {(c.doc_id, c.page) for c in index.chunks if fact in c.text}

    assert pages_with("COP 350,000") == {("travel-expense-policy", 2)}
    assert pages_with("15 días hábiles") == {("politica-vacaciones", 1)}
    assert pages_with("6.5%") == {("manager-compensation-guidelines", 1)}


def test_headers_and_footers_are_removed_and_headings_kept(index):
    assert not any("Page " in c.text or "INTERNAL - employees" in c.text for c in index.chunks)
    hotel = next(c for c in index.chunks if "COP 350,000" in c.text)
    assert hotel.heading == "4. Hotels"


def test_boilerplate_detection_masks_page_numbers():
    pages = ["ACME - Page 1\nBody one", "ACME - Page 2\nBody two", "ACME - Page 3\nBody three"]
    assert boilerplate_lines(pages) == {"ACME - Page #"}


def test_chunking_overlaps_within_a_section_and_respects_size():
    body = " ".join(f"Sentence number {i} has five words." for i in range(1, 9))
    sections = sections_from_lines(["2. Rules", body], carry_heading="Doc Title")
    chunks = chunk_page(META, 5, sections, max_words=18, overlap=1)
    assert len(chunks) > 1 and all(c.page == 5 and c.heading == "2. Rules" for c in chunks)
    for a, b in itertools.pairwise(chunks):
        assert a.text.split(". ")[-1].rstrip(".") in b.text  # last sentence repeated
    assert all(len(c.text.split()) <= 18 for c in chunks)


def test_heading_carries_over_to_continuation_page():
    sections = sections_from_lines(["continued text from the previous page."], carry_heading="7. Reports")
    assert sections == [("7. Reports", "continued text from the previous page.")]


def test_index_roundtrip(index, settings):
    loaded = Index.load(settings.index_dir)
    assert loaded.chunks == index.chunks
    assert loaded.embeddings.shape == index.embeddings.shape
    q = "hotel limit"
    assert (loaded.embedder.embed([q]) == index.embedder.embed([q])).all()
