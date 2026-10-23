"""Offline tests: network and LLM are mocked, so these run anywhere (pytest -q)."""
import json
from datetime import date

import pytest

from app import agents, llm, tools

PAGE = ("The Example Global Scholarship offers full tuition and a monthly stipend of 1,200 EUR. "
        "Applicants must hold a bachelor degree with a minimum GPA of 7.5 out of 10. "
        "The application deadline is 15 January 2099. International students from India may apply. " * 3)


def fake_results(query):
    n = abs(hash(query)) % 10_000
    return [{"title": f"Global Scholarship {n}-{i}", "url": f"https://www.uni-{n}-{i}.de/scholarship",
             "snippet": "scholarship for international students"} for i in range(3)]


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(tools, "web_search", lambda q, n=6: (fake_results(q), "duckduckgo"))
    monkeypatch.setattr(tools, "fetch_page", lambda url, **kw: {"url": url, "ok": True, "text": PAGE, "method": "http+html",
                                                                  "error": "", "fetched_at": "2026-10-23T00:00:00+00:00"})
    monkeypatch.setattr(tools, "github_profile", lambda ref: {"username": "x", "public_repos": 1, "languages": {"Python": 2},
                                                                "top_repositories": [{"name": "r", "description": "d"}]})


def make_llm(honest=True):
    counter = {"n": 0}

    def gen_json(prompt, **kw):
        if "Plan 8 diverse" in prompt:
            return {"queries": ["alpha scholarship india"]}
        if "Extract a truthful" in prompt:
            return {"summary": "s", "skills": ["Python"], "projects": ["p"], "achievements": [], "academics": {}, "missing_information": []}
        counter["n"] += 1
        quote = "full tuition and a monthly stipend of 1,200 EUR" if honest else "this sentence is invented by the model"
        return {"opportunities": [{
            "name": f"Scholarship #{counter['n']}", "provider": "Uni", "country": ["Germany", "Canada", "Singapore"][counter["n"] % 3],
            "study_level": "Master's", "funding": "full tuition + stipend", "funding_evidence": quote,
            "deadline": "15 January 2099", "deadline_evidence": "The application deadline is 15 January 2099",
            "criteria": [{"requirement": "GPA >= 7.5/10", "student_value": "8.2", "verdict": "met",
                          "evidence_quote": "minimum GPA of 7.5 out of 10" if honest else "totally made up requirement"},
                         {"requirement": "Indian nationals", "student_value": "Indian", "verdict": "met",
                          "evidence_quote": "International students from India may apply"}],
            "fit_reasons": ["fits"], "gaps": [], "application_steps": ["apply"], "confidence": 0.9}]}

    return gen_json, lambda prompt, **kw: "I am Rahul Sharma. My CGPA is 8.2."


PROFILE = {"name": "Rahul", "cgpa": "8.2", "study_level": "Master's", "field": "AI", "nationality": "Indian",
           "target_countries": ["Canada", "Germany", "Singapore"], "ielts_status": "Not taken"}


def run(honest=True):
    gj, g = make_llm(honest)
    llm.generate_json, llm.generate = gj, g
    events = []
    return agents.Swarm(on_event=events.append).run(PROFILE, "Rahul\nSkills: Python"), events


def test_end_to_end_meets_challenge_requirements():
    res, events = run()
    assert res["status"] == "completed"
    assert len(res["opportunities"]) >= 10
    assert len(res["ranked"]) == 5
    assert res["sop"]
    assert {e["agent"] for e in events} >= set(agents.AGENTS)
    top = res["ranked"][0]
    assert top["status"] == "Likely eligible"
    assert top["funding_verified"] and top["deadline_verified"] and top["deadline_date"] == "2099-01-15"
    assert all(o["official_url"].startswith("https://") for o in res["opportunities"])


def test_fabricated_quotes_are_downgraded():
    res, _ = run(honest=False)
    o = res["opportunities"][0]
    assert not o["funding_verified"] and "Not confirmed" in o["funding"]
    assert o["criteria"][0]["verdict"] == "unknown" and not o["criteria"][0]["quote_verified"]  # invented quote
    assert o["criteria"][1]["verdict"] == "met"  # genuine quote survives
    assert o["status"] == "Potentially eligible"  # never "Likely eligible" while a criterion is unverified


def test_recovery_loop_triggers_when_too_few_found():
    gj, _ = make_llm()
    calls = {"n": 0}

    def sparse(prompt, **kw):
        if "Plan 8 diverse" in prompt or "Extract a truthful" in prompt:
            return gj(prompt, **kw)
        calls["n"] += 1
        return gj(prompt, **kw) if calls["n"] % 3 == 0 else {"opportunities": []}

    llm.generate_json, llm.generate = sparse, lambda p, **kw: "draft"
    events = []
    res = agents.Swarm(on_event=events.append).run(PROFILE, "x")
    assert res["stats"]["rounds"] >= 2
    assert any(e["status"] == "recovery" and "broaden" in e["message"] for e in events)


def test_survives_total_search_outage(monkeypatch):
    def boom(q, n=6):
        raise tools.SearchError("down")
    monkeypatch.setattr(tools, "web_search", boom)
    gj, g = make_llm()
    llm.generate_json, llm.generate = gj, g
    res = agents.Swarm().run(PROFILE, "x")
    assert res["status"] == "completed" and res["opportunities"] == []
    assert "No viable" in res["sop"]


def test_survives_llm_outage_with_heuristic_fallback():
    def down(*a, **kw):
        raise llm.LLMError("quota")
    llm.generate_json = llm.generate = down
    res = agents.Swarm().run(PROFILE, "Rahul\nPROJECTS\n- Built a classifier\n")
    assert res["status"] == "completed"
    assert "Built a classifier" in res["profile"]["projects"]
    assert "TEMPLATE" in res["sop"] or res["sop"].startswith("No viable")


def test_page_fetch_failure_falls_back_to_snippet(monkeypatch):
    monkeypatch.setattr(tools, "fetch_page", lambda url, **kw: {"url": url, "ok": False, "text": "", "method": "", "error": "timeout", "fetched_at": ""})
    gj, g = make_llm()
    llm.generate_json, llm.generate = gj, g
    events = []
    res = agents.Swarm(on_event=events.append).run(PROFILE, "x")
    assert any(e["status"] == "recovery" and "snippet" in e["message"] for e in events)
    assert all(o["evidence_level"] == "search snippet only" for o in res["opportunities"])
    assert all(o["confidence"] <= 0.45 for o in res["opportunities"])


def test_status_and_helpers():
    assert agents.derive_status([{"verdict": "not_met"}, {"verdict": "met"}]) == "Not eligible"
    assert agents.derive_status([{"verdict": "met"}, {"verdict": "met"}]) == "Likely eligible"
    assert agents.derive_status([{"verdict": "met"}, {"verdict": "unknown"}]) == "Potentially eligible"
    assert agents.derive_status([]) == "Insufficient information"
    assert agents.quote_in_text("Minimum GPA of 7.5!", agents._key("the minimum   GPA of 7.5 required"))
    assert tools.classify_source("https://www.daad.de/en/x") == "official"
    assert tools.classify_source("https://www.mastersportal.com/x") == "secondary"
    assert tools.classify_source("https://www.reddit.com/r/x") == "junk"
    assert tools.normalize_url("https://www.X.com/a/?utm_source=1#frag") == "https://x.com/a"
    assert tools.parse_dates("Deadline: 3rd March 2027 or April 1, 2027 or 2027-05-09") == [date(2027, 3, 3), date(2027, 4, 1), date(2027, 5, 9)]


def test_json_repair(monkeypatch):
    seq = iter(["{broken json", '{"ok": true}'])
    gen_json = __import__("importlib").reload(llm).generate_json  # restore the real implementation
    monkeypatch.setattr(llm, "generate", lambda *a, **kw: next(seq))
    assert gen_json("x") == {"ok": True}
