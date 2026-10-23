import io
import json
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `app` importable under `streamlit run app/main.py`

import os

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from app import llm, tools
from app.agents import AGENTS, MIN_OPPORTUNITIES, Swarm, draft_sop
from app.samples import PRESETS, sample_text

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
st.set_page_config(page_title="ScholarSwarm | AI Scholarship Finder", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"] {font-family:'DM Sans',sans-serif;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif; letter-spacing:-.035em;}
.block-container {padding-top:1.6rem;max-width:1250px;}
.hero {padding:2rem 2.2rem;border-radius:22px;background:linear-gradient(120deg,#111827 0%,#172554 55%,#312e81 100%);color:white;margin-bottom:1.4rem;}
.hero h1 {font-size:2.65rem;line-height:1.08;color:white;margin:0.35rem 0 0.65rem;}
.hero p {color:#dbeafe;font-size:1.05rem;max-width:750px;}
.kicker {color:#a5b4fc;font-weight:700;font-size:.76rem;letter-spacing:.16em;text-transform:uppercase;}
.card {border:1px solid rgba(128,128,128,.35);border-radius:14px;padding:.9rem 1.05rem;min-height:128px;}
.card .t {font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;opacity:.75;}
.card .s {font-size:1.4rem;font-weight:700;font-family:'Space Grotesk';margin:.15rem 0;}
.card .m {font-size:.82rem;opacity:.8;}
.working {border-color:#6366f1;box-shadow:0 0 0 2px rgba(99,102,241,.25);}
.working .s {color:#6366f1;}
.done {border-color:#16a34a;} .done .s {color:#16a34a;}
.warning {border-color:#d97706;} .warning .s {color:#d97706;}
.error {border-color:#dc2626;} .error .s {color:#dc2626;}
.feed {font-size:.86rem;border-left:3px solid rgba(128,128,128,.4);padding:.15rem .7rem;margin:.25rem 0;}
.feed.recovery {border-left-color:#d97706;background:rgba(217,119,6,.08);}
.feed.success {border-left-color:#16a34a;} .feed.error {border-left-color:#dc2626;} .feed.warning {border-left-color:#d97706;}
.source {font-size:.8rem;opacity:.8;overflow-wrap:anywhere;}
.stButton>button[kind="primary"] {border-radius:11px;font-weight:700;}
</style>
<div class="hero"><div class="kicker">MULTI-AGENT RESEARCH WORKSPACE</div><h1>Your next opportunity<br>is out there.</h1><p>Four specialist AI agents work together: they read your documents, search the web, open the official pages, check every requirement against your profile, and draft your application — with evidence you can verify.</p></div>
""", unsafe_allow_html=True)

ICONS = {"Profile Agent": "🧑‍🎓", "Research Agent": "🌐", "Eligibility Agent": "🛡️", "Ranking & Application Agent": "🏆"}
STATE_LABEL = {"waiting": "Waiting", "working": "Working…", "done": "Done", "warning": "Done · warnings", "error": "Problem"}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🎓 ScholarSwarm")
    st.caption("AI-powered scholarship discovery")

    preset_name = st.selectbox(
        "Load a sample student",
        list(PRESETS),
        help="Swap the student and rerun: same agents, new results."
    )

    use_samples = st.checkbox(
        "Use the sample documents if I upload none",
        value=True
    )

    st.markdown("---")
    st.markdown("**Agent swarm**")
    st.markdown(
        "1. 🧑‍🎓 Profile Agent\n"
        "2. 🌐 Research Agent\n"
        "3. 🛡️ Eligibility Agent\n"
        "4. 🏆 Ranking & Application Agent"
    )

    st.caption("Drafts only. Applications are never submitted automatically.")

P = PRESETS[preset_name]
kp = preset_name.split()[0].lower()


# ------------------------------------------------------------------ form
st.markdown("### 01 / Student profile")
with st.form("profile_form"):
    a, b, c = st.columns(3)
    with a:
        name = st.text_input("Full name", P["name"], key=f"{kp}_name")
        degree = st.text_input("Current degree", P["degree"], key=f"{kp}_degree")
        cgpa = st.text_input("CGPA / GPA", P["cgpa"], key=f"{kp}_cgpa")
        grad = st.text_input("Graduation year", P["grad"], key=f"{kp}_grad")
    with b:
        nationality = st.text_input("Citizenship", P["nationality"], key=f"{kp}_nat")
        field = st.text_input("Target field", P["field"], key=f"{kp}_field")
        level = st.selectbox("Study level", ["Master's", "Bachelor's", "PhD", "Exchange / short course"], index=["Master's", "Bachelor's", "PhD", "Exchange / short course"].index(P["level"]), key=f"{kp}_level")
        all_countries = ["India","Canada", "Germany", "Singapore", "United States", "United Kingdom", "Netherlands", "Australia", "Sweden", "France", "Ireland", "Japan", "South Korea"]
        countries = st.multiselect("Target countries", all_countries, default=P["countries"], key=f"{kp}_countries")
    with c:
        github = st.text_input("Public GitHub profile", P["github"], placeholder="https://github.com/username", key=f"{kp}_gh")
        ielts_opts = ["Not taken / unknown", "IELTS completed", "TOEFL completed", "Waiver may apply"]
        ielts = st.selectbox("English test status", ielts_opts, index=ielts_opts.index(P["ielts"]), key=f"{kp}_ielts")
        fund_opts = ["Full funding needed", "Tuition scholarship needed", "Partial funding acceptable"]
        funding = st.selectbox("Funding need", fund_opts, index=fund_opts.index(P["funding"]), key=f"{kp}_fund")
        intake_opts = ["Fall 2027", "Spring 2027", "Fall 2028", "Flexible / earliest available"]
        intake = st.selectbox("Target intake", intake_opts, index=intake_opts.index(P["intake"]), key=f"{kp}_intake")
    profile_links = st.text_input("Other public profile links (optional)", placeholder="Portfolio, personal site, research profile (space-separated)", key=f"{kp}_links")
    files = st.file_uploader("Upload resume, marksheet / transcript (PDF, TXT, or a photo)", type=["pdf", "txt", "png", "jpg", "jpeg"], accept_multiple_files=True)
    notes = st.text_area("Extra context (optional)", placeholder="Research experience, awards, projects, financial constraints, preferred universities…", height=80, key=f"{kp}_notes")
    submitted = st.form_submit_button("🚀 Launch scholarship agent swarm", type="primary", width="stretch")


def read_files(uploaded):
    text, images = [], []
    for f in uploaded:
        raw, lname = f.getvalue(), f.name.lower()
        if lname.endswith(".pdf"):
            try:
                t = "\n".join(pg.extract_text() or "" for pg in PdfReader(io.BytesIO(raw)).pages)
                if len(t.strip()) < 50:
                    st.warning(f"{f.name} looks scanned (no text layer). Upload it as a photo (PNG/JPG) so Gemini can read it.")
                text.append(f"=== FILE: {f.name} ===\n{t}")
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not read {f.name}: {exc}")
        elif lname.endswith((".png", ".jpg", ".jpeg")):
            images.append((f.name, "image/png" if lname.endswith(".png") else "image/jpeg", raw))
        else:
            text.append(f"=== FILE: {f.name} ===\n" + raw.decode("utf-8", errors="ignore"))
    return "\n\n".join(text), images


def card_states(events):
    s = {a: {"state": "waiting", "msg": "Waiting for upstream agent"} for a in AGENTS}
    for e in events:
        a = e["agent"]
        if a not in s:
            continue
        if e["phase"] == "start":
            s[a]["state"] = "working"
        elif e["phase"] == "done":
            s[a]["state"] = {"error": "error", "warning": "warning"}.get(e["status"], "done")
        s[a]["msg"] = e["message"]
    return s


def render_live(cards_ph, feed_ph, events, started):
    states = card_states(events)
    with cards_ph.container():
        cols = st.columns(4)
        for i, (col, a) in enumerate(zip(cols, AGENTS)):
            s = states[a]
            col.markdown(f"<div class='card {s['state']}'><div class='t'>{ICONS[a]} Agent {i + 1} · {a}</div><div class='s'>{STATE_LABEL[s['state']]}</div><div class='m'>{s['msg'][:120]}</div></div>", unsafe_allow_html=True)
    with feed_ph.container():
        recov = sum(1 for e in events if e["status"] == "recovery")
        st.caption(f"⏱ {time.time() - started:0.0f}s · {len(events)} events · {recov} self-recoveries")
        for e in events[-12:][::-1]:
            icon = {"recovery": "🔁", "success": "✅", "warning": "⚠️", "error": "❌"}.get(e["status"], "•")
            st.markdown(f"<div class='feed {e['status']}'>{icon} <b>{e['time']} · {e['agent']}</b> — {e['message']}</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ run (streams live)
if submitted:
    text, images = read_files(files)
    if not files and use_samples:
        text = sample_text(P)
    profile = {"name": name, "degree": degree, "cgpa": cgpa, "graduation_year": grad, "nationality": nationality, "field": field,
               "study_level": level, "target_countries": countries or ["Canada", "Germany", "Singapore"], "github": github,
               "ielts_status": ielts, "budget_need": funding, "target_intake": intake, "other_profile_links": profile_links, "extra_context": notes}
    st.markdown("---")
    st.markdown("### 02 / Live agent activity")
    cards_ph, feed_ph = st.empty(), st.empty()
    q: queue.Queue = queue.Queue()
    holder: dict = {}

    def work():
        try:
            holder["result"] = Swarm(on_event=q.put).run(profile, text, images)
        except Exception as exc:  # noqa: BLE001
            holder["error"] = str(exc)
        finally:
            q.put(None)

    started = time.time()
    threading.Thread(target=work, daemon=True).start()
    events: list = []
    render_live(cards_ph, feed_ph, events, started)
    while True:
        try:
            ev = q.get(timeout=0.5)
        except queue.Empty:
            render_live(cards_ph, feed_ph, events, started)
            continue
        if ev is None:
            break
        events.append(ev)
        render_live(cards_ph, feed_ph, events, started)
    if "result" in holder:
        st.session_state["swarm_result"] = holder["result"]
        st.session_state["sop_text"] = holder["result"].get("sop", "")
        st.session_state["sop_warnings"] = holder["result"].get("sop_warnings", [])
        st.session_state["sop_version"] = st.session_state.get("sop_version", 0) + 1
    else:
        st.error(f"Workflow failed: {holder.get('error', 'unknown error')}")

# ------------------------------------------------------------------ results
result = st.session_state.get("swarm_result")
if result:
    st.markdown("---")
    st.markdown("### 03 / Run summary")
    stats, logs = result.get("stats", {}), result.get("logs", [])
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Searches run", stats.get("queries", 0))
    s2.metric("Sources found", stats.get("sources", 0))
    s3.metric("Pages read in full", stats.get("pages_read", 0))
    s4.metric("Opportunities", stats.get("opportunities", 0), help=f"Target: {MIN_OPPORTUNITIES}+")
    s5.metric("Research rounds", stats.get("rounds", 1), help="More than 1 means the Eligibility Agent asked the Research Agent to try again")
    with st.expander(f"Full agent event log ({len(logs)} events)"):
        for e in logs:
            st.markdown(f"<div class='feed {e['status']}'><b>{e['time']} · {e['agent']}</b> — {e['message']}</div>", unsafe_allow_html=True)
    for err in result.get("errors", [])[:12]:
        st.warning(err)

    p = result.get("profile", {})
    st.markdown("### 04 / Student intelligence")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CGPA", str(p.get("cgpa", "Not provided")))
    m2.metric("Skills found", len(p.get("skills", [])))
    m3.metric("Projects found", len(p.get("projects", [])))
    m4.metric("Open questions", len(p.get("missing_information", [])))
    if p.get("summary"):
        st.write(p["summary"])
    with st.expander("Extracted profile, data sources and missing information", expanded=True):
        x, y = st.columns(2)
        with x:
            st.markdown("**Skills**"); st.write(p.get("skills") or "None extracted")
            st.markdown("**Projects**"); st.write(p.get("projects") or "None extracted")
            st.markdown("**Achievements**"); st.write(p.get("achievements") or "None extracted — nothing is invented")
        with y:
            st.markdown("**Profile built from**"); st.write(p.get("sources_used") or ["Form fields only"])
            st.markdown("**Missing information to confirm**"); st.write(p.get("missing_information") or [])
            if p.get("github"):
                gh = p["github"]
                st.markdown(f"**GitHub:** [{gh.get('username')}]({gh.get('profile_url')}) · {gh.get('public_repos')} repos · languages: {', '.join(list((gh.get('languages') or {}))[:5])}")

    opps = result.get("opportunities", [])
    st.markdown(f"### 05 / All opportunities found ({len(opps)})")
    st.caption("Every row links to the page the agents actually visited. Deadline and funding show only when a quote from that page was verified.")
    if opps:
        st.dataframe([{"Opportunity": o["name"], "Provider": o["university_or_provider"], "Country": o["country"], "Status": o["status"],
                       "Funding ✓": "✓" if o["funding_verified"] else "—", "Deadline": o["deadline"] if o["deadline_verified"] else "verify",
                       "Source": o["source_type"], "Evidence": o["evidence_level"], "Score": o.get("score"), "URL": o["official_url"]} for o in opps],
                     width="stretch", hide_index=True, column_config={"URL": st.column_config.LinkColumn("URL")})
    else:
        st.info("No opportunities could be verified. Check the event log above, then retry (search outage or missing Gemini key are the usual causes).")

    st.markdown("### 06 / Top 5 matches")
    matches = result.get("ranked", [])
    badge = {"Likely eligible": "🟢", "Potentially eligible": "🟡", "Insufficient information": "⚪", "Not eligible": "🔴"}
    for i, o in enumerate(matches, 1):
        with st.container(border=True):
            head, tag = st.columns([4, 1])
            with head:
                st.markdown(f"#### {i}. {o['name']}")
                st.caption(f"{o['university_or_provider']} · {o['country']} · {o.get('opportunity_type', '')}")
            with tag:
                st.metric("Confidence", f"{o['confidence'] * 100:.0f}%")
            st.markdown(f"{badge.get(o['status'], '⚪')} **{o['status']}** · source: **{o['source_type']}** · evidence: **{o['evidence_level']}**")
            f1, f2 = st.columns(2)
            f1.markdown(f"**Funding** {'✅' if o['funding_verified'] else '⚠️'}\n\n{o['funding']}")
            f2.markdown(f"**Deadline** {'✅' if o['deadline_verified'] else '⚠️'}{' · ⛔ passed' if o['expired'] else ''}\n\n{o['deadline']}")
            if o.get("criteria"):
                st.markdown("**Requirement check (each verdict needs a verified quote from the page)**")
                mark = {"met": "✅ met", "not_met": "❌ not met", "unknown": "❓ unknown"}
                st.dataframe([{"Requirement": c["requirement"], "Your profile": c["student_value"], "Verdict": mark[c["verdict"]],
                               "Quote verified": "✓" if c["quote_verified"] else "—", "Evidence": c["evidence_quote"]} for c in o["criteria"]],
                             width="stretch", hide_index=True)
            l, r = st.columns(2)
            with l:
                st.markdown("**Why it may fit**")
                for x in o.get("fit_reasons", []): st.markdown(f"- {x}")
            with r:
                st.markdown("**Gaps / verify**")
                for x in o.get("gaps", []): st.markdown(f"- {x}")
            if o.get("application_steps"):
                with st.expander("How to apply"):
                    for x in o["application_steps"]: st.markdown(f"- {x}")
            with st.expander("Why this rank? (score breakdown)"):
                st.dataframe(o.get("score_breakdown", []), hide_index=True, width="stretch")
            st.markdown(f"[Open the source page ↗]({o['official_url']})  ·  retrieved {o.get('retrieved_at') or 'n/a'}")

    if result.get("action_items"):
        st.markdown("### 07 / What Rahul still needs to do" if "Rahul" in p.get("name", "") else "### 07 / What you still need to do")
        for x in result["action_items"]:
            st.checkbox(x, key=f"todo_{hash(x)}")

    st.markdown("### 08 / Statement of purpose draft")
    st.warning("Draft only. Review every claim, replace [PERSONALIZE] placeholders and verify program details. Never submit unreviewed AI text.")
    if matches:
        pick = st.selectbox("Draft for", range(len(matches)), format_func=lambda i: f"{i + 1}. {matches[i]['name']}", key="sop_pick")
        if st.button("✍️ Regenerate draft for this opportunity"):
            with st.spinner("Writing…"):
                t, w = draft_sop(p, matches[pick])
            st.session_state.update(sop_text=t, sop_warnings=w, sop_version=st.session_state.get("sop_version", 0) + 1)
            st.rerun()
    for w in st.session_state.get("sop_warnings", []):
        st.warning(w)
    ver = st.session_state.get("sop_version", 0)
    sop = st.text_area("Editable SOP draft", value=st.session_state.get("sop_text", ""), height=340, key=f"sop_edit_{ver}")
    d1, d2 = st.columns(2)
    d1.download_button("Download SOP draft (.txt)", data=sop, file_name="scholarship_sop_draft.txt", mime="text/plain")
    slim = [{k: v for k, v in x.items() if k != "page_text"} for x in result.get("research", [])]
    export = {"profile": {k: v for k, v in p.items() if k != "resume_excerpt"}, "opportunities": opps, "top_matches": matches,
              "sources_visited": slim, "stats": stats, "agent_logs": logs, "errors": result.get("errors", [])}
    d2.download_button("Download research report (.json)", data=json.dumps(export, indent=2, ensure_ascii=False), file_name="scholarship_research_report.json", mime="application/json")
    st.caption("Privacy: document text is sent to Gemini only when a key is configured. Avoid uploading passports or national IDs. Nothing is submitted anywhere.")
else:
    st.markdown("### What you'll get")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**🔎 Real research**\n\nLive web search, pages actually opened and read, every result linked to its source.")
    c2.markdown("**🧭 Evidence, not guesses**\n\nA requirement is marked met or unmet only when a quote from the page is verified.")
    c3.markdown("**✍️ Application support**\n\nSOP draft with placeholders for anything missing — never auto-submitted.")
    with st.expander("How the swarm works"):
        st.graphviz_chart('''digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica"];
          P [label="Profile Agent\\nresume + marksheet + GitHub"]; R [label="Research Agent\\nplan → search → open pages"];
          E [label="Eligibility Agent\\nquote-verified criteria"]; K [label="Ranking & Application\\ntop 5 + SOP"];
          P -> R -> E -> K; E -> R [label="fewer than 10?\\nbroaden (feedback)", color="#d97706", fontcolor="#d97706"]; }''')
    st.info("Add a Gemini key in the sidebar for the full AI workflow. Search works without any key via DuckDuckGo.")
