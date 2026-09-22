"""Fetch and clean 8-K documents from EDGAR Archives for text scoring (stage 3, step 2).

For a filing we want the informative text, not the boilerplate cover: for earnings 8-Ks (Item 2.02) that is the
press release exhibit (EX-99.1); otherwise the Item sections of the primary document. Everything is cached on
disk under data/edgar/docs/<accession>/ so re-runs never hit SEC again. Same User-Agent / rate limits as ljs.edgar.
"""
from __future__ import annotations

import html as htmllib
import json
import re
from pathlib import Path

from loguru import logger

from ljs.edgar import _get

ARCHIVE = "https://www.sec.gov/Archives/edgar/data"
ITEM_RE = re.compile(r"(?i)\bitem\s+(\d\.\d\d)\b")


def filing_index(cik: int, accession: str, cache_dir: Path) -> dict | None:
    """index.json of a filing folder: lists all documents with names/types (via the -index.htm we parse names)."""
    acc = accession.replace("-", "")
    cache_dir = Path(cache_dir) / acc
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / "index.json"
    if p.exists():
        return json.loads(p.read_text())
    r = _get(f"{ARCHIVE}/{cik}/{acc}/index.json")
    if r.status_code != 200:
        logger.warning(f"index {accession}: HTTP {r.status_code}")
        return None
    j = r.json()
    p.write_text(json.dumps(j))
    return j


def fetch_document(cik: int, accession: str, name: str, cache_dir: Path) -> str | None:
    acc = accession.replace("-", "")
    p = Path(cache_dir) / acc / name
    if p.exists():
        return p.read_text(errors="ignore")
    r = _get(f"{ARCHIVE}/{cik}/{acc}/{name}")
    if r.status_code != 200:
        logger.warning(f"doc {accession}/{name}: HTTP {r.status_code}")
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(r.text)
    return r.text


def html_to_text(raw: str) -> str:
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    s = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>|</li>|</h\d>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = htmllib.unescape(s)
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


def pick_exhibit_99(index: dict) -> str | None:
    """Name of the first EX-99* document (press release) in a filing folder, if any."""
    items = index.get("directory", {}).get("item", [])
    names = [it["name"] for it in items]
    for n in names:
        if re.search(r"(?i)ex[-_]?99", n) and n.lower().endswith((".htm", ".html", ".txt")):
            return n
    return None


def item_sections(text: str, max_chars: int = 2500) -> dict[str, str]:
    """Split the primary 8-K document into Item x.xx sections (heuristic), dropping the signature/exhibit tail."""
    text = re.split(r"(?i)\n\s*signature", text)[0]
    hits = list(ITEM_RE.finditer(text))
    out: dict[str, str] = {}
    for i, h in enumerate(hits):
        start = h.end()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        body = text[start:end].strip(" .:\n-")
        if len(body) > 40 and h.group(1) not in out:
            out[h.group(1)] = body[:max_chars]
    return out


def event_text(cik: int, accession: str, items: list[str], cache_dir: Path, primary: str | None = None, max_chars: int = 3000) -> dict:
    """Best-effort informative text for a filing: press release (EX-99) for 2.02/7.01/8.01 filings when present,
    otherwise the concatenated Item sections of the primary document. Returns {'source', 'text'}."""
    idx = filing_index(cik, accession, cache_dir)
    if idx is None:
        return {"source": None, "text": ""}
    ex = pick_exhibit_99(idx) if any(i in items for i in ("2.02", "7.01", "8.01")) else None
    if ex:
        raw = fetch_document(cik, accession, ex, cache_dir)
        if raw:
            t = html_to_text(raw)
            return {"source": ex, "text": t[:max_chars]}
    name = primary or next((it["name"] for it in idx.get("directory", {}).get("item", []) if it["name"].lower().endswith((".htm", ".html"))), None)
    if not name:
        return {"source": None, "text": ""}
    raw = fetch_document(cik, accession, name, cache_dir)
    if not raw:
        return {"source": name, "text": ""}
    secs = item_sections(html_to_text(raw))
    keep = [f"Item {k}: {v}" for k, v in secs.items() if k != "9.01"]
    return {"source": name, "text": "\n".join(keep)[:max_chars]}
