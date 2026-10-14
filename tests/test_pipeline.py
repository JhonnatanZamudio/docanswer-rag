from dataclasses import replace
from types import SimpleNamespace

from docanswer.llm import REFUSAL, AnthropicProvider, GenerationRequest
from docanswer.pipeline import REFUSAL_ES, DocAnswer, verify_citations


class SpyProvider:
    name = "spy"

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def generate(self, request):
        self.calls.append(request)
        return self.reply(request) if callable(self.reply) else self.reply


def make(engine, provider, **overrides):
    return DocAnswer(engine.retriever.index, provider, replace(engine.settings, **overrides))


def test_answer_cites_the_right_page(engine):
    answer = engine.ask("What is the nightly hotel limit in Bogota?", "employee")
    assert not answer.refused
    assert "COP 350,000" in answer.answer and "[Travel Policy p.2]" in answer.answer
    assert (answer.citations[0].doc_id, answer.citations[0].page) == ("travel-expense-policy", 2)


def test_citation_checker_flags_unretrieved_sources(engine):
    contexts = engine.retriever.search("hotel cap in Bogota", "employee").results
    check = verify_citations(
        "Cap is COP 350,000 [Travel Policy p.2]. Also [Travel Policy p.9] [Made Up p.1].", contexts
    )
    assert [(s.chunk.doc_id, s.chunk.page) for s in check.valid] == [("travel-expense-policy", 2)]
    assert check.invalid == ["[Travel Policy p.9]", "[Made Up p.1]"]


def test_fabricated_citations_are_stripped(engine):
    spy = SpyProvider("The cap is COP 350,000 [Travel Policy p.2] [Secret Memo p.4].")
    answer = make(engine, spy).ask("What is the hotel cap in Bogota?", "employee")
    assert "Secret Memo" not in answer.answer and len(answer.citations) == 1


def test_answer_without_valid_citation_is_refused(engine):
    spy = SpyProvider("The cap is probably USD 500 [Imaginary Policy p.1].")
    answer = make(engine, spy).ask("What is the hotel cap in Bogota?", "employee")
    assert answer.refused and answer.answer == REFUSAL and answer.citations == []


def test_low_confidence_refuses_without_calling_the_model(engine):
    spy = SpyProvider("should never be used [Travel Policy p.2]")
    answer = make(engine, spy).ask("Does the company offer a stock option plan?", "employee")
    assert answer.refused and spy.calls == []


def test_restricted_question_is_refused_for_employees(engine):
    answer = engine.ask("What is the merit increase budget for 2026?", "employee")
    assert answer.refused
    assert all(s.chunk.doc_id != "manager-compensation-guidelines" for s in answer.retrieved)


def test_model_refusal_is_respected_and_localised(engine):
    answer = make(engine, SpyProvider(REFUSAL)).ask("¿Cuántos días de vacaciones tengo por año?", "employee")
    assert answer.refused and answer.answer == REFUSAL_ES


def test_anthropic_provider_sends_grounded_prompt_without_network(engine):
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="COP 350,000 [Travel Policy p.2]")])

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    provider = AnthropicProvider(model="test-model", client=client)
    contexts = engine.retriever.search("hotel cap Bogota", "employee").results
    text = provider.generate(GenerationRequest("hotel cap Bogota?", contexts))
    assert text == "COP 350,000 [Travel Policy p.2]"
    assert sent["model"] == "test-model" and sent["temperature"] == 0
    assert "ONLY" in sent["system"] and 'tag="[Travel Policy p.2]"' in sent["messages"][0]["content"]
