"""Real-world tools used by the agents: web search, page fetching, GitHub, helpers."""
from __future__ import annotations

import io
import ipaddress
import os
import re
import socket
from datetime import date, datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

UA = "Mozilla/5.0 (compatible; ScholarSwarm/1.0; student research project)"


class SearchError(RuntimeError):
    pass


class GitHubError(RuntimeError):
    pass


# --------------------------------------------------------------------------- URLs
def normalize_url(url: str) -> str:
    p = urlparse(url.strip())
    q = [(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith(("utm_", "fbclid", "gclid"))]
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme.lower(), p.netloc.lower().removeprefix("www."), path, "", urlencode(q), ""))


def domain_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


_OFFICIAL_RE = re.compile(
    r"(\.edu(\.[a-z]{2})?$|\.ac\.[a-z]{2}$|\.gov(\.[a-z]{2})?$|\.gc\.ca$|\.europa\.eu$|\.go\.(jp|kr)$"
    r"|(^|\.)uni-[a-z-]+\.de$|(^|\.)tu-[a-z-]+\.de$|(^|\.)univ-[a-z-]+\.fr$)"
)
_OFFICIAL_SET = {
    "daad.de", "educanada.ca", "si.se", "studyinnl.org", "chevening.org", "campusfrance.org",
    "educationinireland.com", "studyaustralia.gov.au", "jasso.go.jp", "studyinkorea.go.kr",
    "utoronto.ca", "ubc.ca", "uwaterloo.ca", "mcgill.ca", "ualberta.ca", "queensu.ca", "sfu.ca",
    "yorku.ca", "uottawa.ca", "mcmaster.ca", "dal.ca", "ucalgary.ca", "concordia.ca", "uvic.ca",
    "tum.de", "rwth-aachen.de", "kit.edu", "tudelft.nl", "tue.nl", "uva.nl", "uu.nl", "utwente.nl",
    "kth.se", "chalmers.se", "lu.se", "uu.se", "su.se", "tcd.ie", "ucd.ie", "ethz.ch", "epfl.ch",
    "ntu.edu.sg", "nus.edu.sg", "smu.edu.sg", "sutd.edu.sg", "a-star.edu.sg",
}
_AGGREGATORS = {
    "scholars4dev.com", "scholarshipportal.com", "mastersportal.com", "shiksha.com", "leverageedu.com",
    "scholarships.com", "studyportals.com", "internationalscholarships.com", "buddy4study.com",
    "collegedunia.com", "scholarshiproar.com", "findamasters.com", "topuniversities.com",
    "educations.com", "studyabroadaid.com", "afterschoolafrica.com", "scholarshipsads.com",
}
_JUNK = {
    "reddit.com", "quora.com", "youtube.com", "facebook.com", "instagram.com", "pinterest.com",
    "twitter.com", "x.com", "tiktok.com", "linkedin.com", "medium.com", "wikipedia.org",
}


def _suffix_in(dom: str, pool: set[str]) -> bool:
    return any(dom == d or dom.endswith("." + d) for d in pool)


def classify_source(url: str) -> str:
    """official | secondary | unclear | junk (domain heuristic only)."""
    dom = domain_of(url)
    if not dom or _suffix_in(dom, _JUNK):
        return "junk"
    if _OFFICIAL_RE.search(dom) or _suffix_in(dom, _OFFICIAL_SET):
        return "official"
    if _suffix_in(dom, _AGGREGATORS):
        return "secondary"
    return "unclear"


def _safe_url(url: str) -> bool:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    try:
        for info in socket.getaddrinfo(p.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
    except (OSError, ValueError):
        return True  # DNS failure will surface naturally in the request
    return True


# --------------------------------------------------------------------------- search
def search_provider_name() -> str:
    return "Tavily (with DuckDuckGo fallback)" if os.getenv("TAVILY_API_KEY", "").strip() else "DuckDuckGo (no key needed)"


def _tavily_search(query: str, n: int) -> list[dict]:
    r = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {os.getenv('TAVILY_API_KEY', '').strip()}"},
        json={"query": query, "search_depth": "basic", "max_results": n, "include_answer": False},
        timeout=40,
    )
    r.raise_for_status()
    return [
        {"title": x.get("title") or "Untitled", "url": x.get("url", ""), "snippet": (x.get("content") or "")[:1400]}
        for x in r.json().get("results", [])
        if x.get("url")
    ]


def _ddg_search(query: str, n: int) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:  # older package name
        from duckduckgo_search import DDGS
    rows = list(DDGS().text(query, max_results=n))
    return [
        {"title": r.get("title") or "Untitled", "url": r.get("href") or r.get("url", ""), "snippet": (r.get("body") or "")[:1400]}
        for r in rows
        if (r.get("href") or r.get("url"))
    ]


def web_search(query: str, max_results: int = 6) -> tuple[list[dict], str]:
    """Search the web. Returns (results, provider). Tavily first, DuckDuckGo as fallback."""
    errors = []
    if os.getenv("TAVILY_API_KEY", "").strip():
        try:
            return _tavily_search(query, max_results), "tavily"
        except Exception as exc:  # noqa: BLE001 - any failure triggers the fallback
            errors.append(f"tavily: {type(exc).__name__}")
    try:
        return _ddg_search(query, max_results), "duckduckgo"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"duckduckgo: {type(exc).__name__}: {str(exc)[:80]}")
    raise SearchError("; ".join(errors))


# --------------------------------------------------------------------------- page fetch
def _html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "svg", "iframe"]):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = root.get_text("\n")
    if len(text) < 400 and root is not soup:  # <main> was too thin; use the whole page
        text = soup.get_text("\n")
    lines = [re.sub(r"[ \t\u00a0]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _pdf_text(data: bytes, max_pages: int = 6) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join((pg.extract_text() or "") for pg in reader.pages[:max_pages])


def _tavily_extract(url: str) -> str:
    r = requests.post(
        "https://api.tavily.com/extract",
        headers={"Authorization": f"Bearer {os.getenv('TAVILY_API_KEY', '').strip()}"},
        json={"urls": [url]},
        timeout=45,
    )
    r.raise_for_status()
    res = r.json().get("results", [])
    return (res[0].get("raw_content") or "") if res else ""


def fetch_page(url: str, max_chars: int = 9000, timeout: int = 20) -> dict:
    """Visit a page and return readable text. Direct GET first, Tavily extract as recovery."""
    out = {"url": url, "ok": False, "text": "", "method": "", "error": "",
           "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if not _safe_url(url):
        out["error"] = "blocked: non-public address"
        return out
    err = ""
    for attempt in range(2):
        try:
            r = requests.get(url, headers={"User-Agent": UA, "Accept-Language": "en"}, timeout=timeout)
            r.raise_for_status()
            ctype = r.headers.get("content-type", "").lower()
            if "pdf" in ctype or url.lower().split("?")[0].endswith(".pdf"):
                text, method = _pdf_text(r.content), "http+pdf"
            else:
                text, method = _html_text(r.text), "http+html"
            if len(text.strip()) >= 300:
                out.update(ok=True, text=text.strip()[:max_chars], method=method)
                return out
            err = "page had too little readable text (possibly JavaScript-rendered)"
            break
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {str(exc)[:100]}"
    if os.getenv("TAVILY_API_KEY", "").strip():
        try:
            text = _tavily_extract(url)
            if len(text.strip()) >= 300:
                out.update(ok=True, text=text.strip()[:max_chars], method="tavily-extract")
                return out
        except Exception as exc:  # noqa: BLE001
            err += f" | tavily-extract: {type(exc).__name__}"
    out["error"] = err
    return out


# --------------------------------------------------------------------------- GitHub
def github_username(ref: str) -> str:
    m = re.search(r"github\.com/([A-Za-z0-9-]+)", ref or "")
    return m.group(1) if m else (ref or "").strip().strip("/").lstrip("@")


def github_profile(ref: str) -> dict:
    """Public GitHub enrichment: bio, languages, top repos with README excerpts."""
    user = github_username(ref)
    if not user or not re.fullmatch(r"[A-Za-z0-9-]{1,39}", user):
        raise GitHubError("could not read a GitHub username")
    headers = {"Accept": "application/vnd.github+json", "User-Agent": UA}
    if os.getenv("GITHUB_TOKEN", "").strip():
        headers["Authorization"] = f"Bearer {os.getenv('GITHUB_TOKEN').strip()}"
    r = requests.get(f"https://api.github.com/users/{user}", headers=headers, timeout=15)
    if r.status_code == 404:
        raise GitHubError(f"GitHub user '{user}' not found")
    if r.status_code in (403, 429):
        raise GitHubError("GitHub API rate limit reached (set GITHUB_TOKEN to raise it)")
    r.raise_for_status()
    u = r.json()
    rr = requests.get(f"https://api.github.com/users/{user}/repos?per_page=40&sort=updated", headers=headers, timeout=15)
    repos = [x for x in (rr.json() if rr.ok else []) if not x.get("fork")]
    langs: dict[str, int] = {}
    for x in repos:
        if x.get("language"):
            langs[x["language"]] = langs.get(x["language"], 0) + 1
    top = sorted(repos, key=lambda x: (x.get("stargazers_count", 0), x.get("updated_at", "")), reverse=True)[:6]
    out_repos = []
    for i, x in enumerate(top):
        excerpt = ""
        if i < 3:
            try:
                h = dict(headers, Accept="application/vnd.github.raw")
                rd = requests.get(f"https://api.github.com/repos/{user}/{x['name']}/readme", headers=h, timeout=12)
                if rd.ok:
                    excerpt = re.sub(r"\s+", " ", rd.text)[:600]
            except requests.RequestException:
                pass
        out_repos.append({"name": x.get("name"), "description": x.get("description"), "language": x.get("language"),
                          "stars": x.get("stargazers_count", 0), "topics": x.get("topics", []),
                          "url": x.get("html_url"), "readme_excerpt": excerpt})
    return {"username": u.get("login"), "name": u.get("name"), "bio": u.get("bio"),
            "followers": u.get("followers"), "public_repos": u.get("public_repos"),
            "profile_url": u.get("html_url"), "languages": langs, "top_repositories": out_repos}


# --------------------------------------------------------------------------- dates
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], 1)}
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DATE_PATTERNS = [
    re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MON}\.?,?\s+(20\d\d)\b", re.I),   # 15 January 2027
    re.compile(rf"\b{_MON}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(20\d\d)\b", re.I),   # January 15, 2027
    re.compile(r"\b(20\d\d)-(\d{2})-(\d{2})\b"),                                       # 2027-01-15
]


def today() -> date:
    return date.today()


def _month_num(s: str) -> int:
    s = s.lower()
    return next(n for name, n in _MONTHS.items() if name.startswith(s[:3]))


def parse_dates(text: str) -> list[date]:
    found: list[date] = []
    for i, pat in enumerate(_DATE_PATTERNS):
        for m in pat.finditer(text or ""):
            try:
                if i == 0:
                    d = date(int(m.group(3)), _month_num(m.group(2)), int(m.group(1)))
                elif i == 1:
                    d = date(int(m.group(3)), _month_num(m.group(1)), int(m.group(2)))
                else:
                    d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                found.append(d)
            except ValueError:
                continue
    return found


def parse_deadline(text: str):
    """Earliest upcoming date in text, else the latest past date, else None."""
    dates = parse_dates(text)
    if not dates:
        return None
    upcoming = [d for d in dates if d >= today()]
    return min(upcoming) if upcoming else max(dates)
