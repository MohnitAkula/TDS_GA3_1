"""A2A 1.0 (HTTP+JSON) invoice action agent.

Layers: (1) protocol/auth  (2) SQLite storage + idempotency  (3) AI decision step.
Run ONE worker process: critical sections are synchronous (no awaits) so they are atomic
within the event loop; SQLite persists state across restarts.

Env:
  PUBLIC_BASE_URL   exact submitted base URL, e.g. https://host/a2a/   (required for the card)
  LLM_BASE_URL      OpenAI-compatible base, e.g. https://api.openai.com/v1
  LLM_API_KEY       key (never put in prompts)
  LLM_MODEL         e.g. gpt-4o-mini / any small model
  API_TOKENS        optional comma list of allowed bearer tokens (else any non-empty token = a user)
  DB_PATH           default ./agent.db
  EVIDENCE_BRACKETS "1" keeps [brackets] in evidenceRefs (default strips them)
"""
import os, re, json, uuid, hashlib, sqlite3, asyncio
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

A2A = "application/a2a+json"
BATCH_MT = "application/vnd.ga5.invoice-claim-batch+json"
PROP_MT = "application/vnd.ga5.invoice-action-proposals+json"
RES_MT = "application/vnd.ga5.invoice-action-results+json"
RECP_MT = "application/vnd.ga5.invoice-action-receipts+json"
ACTIONS = ["settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"]
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "https://example.com/a2a/")
BP = urlparse(PUBLIC_BASE_URL).path.rstrip("/")
LLM_BASE = os.environ.get("LLM_BASE_URL", "").rstrip("/")
LLM_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
ALLOWED = {t.strip() for t in os.environ.get("API_TOKENS", "").split(",") if t.strip()}
KEEP_BRACKETS = os.environ.get("EVIDENCE_BRACKETS", "0") == "1"

db = sqlite3.connect(os.environ.get("DB_PATH", "agent.db"), check_same_thread=False, isolation_level=None)
db.executescript("""
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, principal TEXT, state TEXT, body TEXT, created REAL);
CREATE TABLE IF NOT EXISTS msgs(principal TEXT, message_id TEXT, hash TEXT, task_id TEXT, PRIMARY KEY(principal,message_id));
CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, body TEXT);
CREATE TABLE IF NOT EXISTS executions(principal TEXT, task_id TEXT, package_id TEXT, PRIMARY KEY(principal,task_id,package_id));
""")
events: dict[str, asyncio.Event] = {}


# ---------------------------------------------------------------- helpers
def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

def chash(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

def err(status, code, msg="Request failed.", headers=None):
    return JSONResponse({"error": {"code": status, "status": code, "message": msg,
                                   "details": [{"reason": code}]}},
                        status_code=status, media_type=A2A, headers=headers)

def guard(request: Request, need_ct: bool):
    """Auth -> version -> media type. Returns (principal, error)."""
    auth = request.headers.get("authorization", "")
    m = re.fullmatch(r"Bearer (\S+)", auth)
    if not m or (ALLOWED and m.group(1) not in ALLOWED):
        return None, err(401, "UNAUTHENTICATED", "Authentication required.", {"WWW-Authenticate": "Bearer"})
    if request.headers.get("a2a-version", "").strip() != "1.0":
        return None, err(400, "VERSION_NOT_SUPPORTED", "Unsupported A2A version.")
    if need_ct:
        ct = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ct not in (A2A, "application/json"):
            return None, err(415, "UNSUPPORTED_MEDIA_TYPE", "Unsupported media type.")
    return hashlib.sha256(m.group(1).encode()).hexdigest(), None

def get_task(tid, principal):
    r = db.execute("SELECT body FROM tasks WHERE id=? AND principal=?", (tid, principal)).fetchone()
    return json.loads(r[0]) if r else None

def save_task(principal, t, created=None):
    db.execute("INSERT INTO tasks(id,principal,state,body,created) VALUES(?,?,?,?,?) "
               "ON CONFLICT(id) DO UPDATE SET state=excluded.state, body=excluded.body",
               (t["id"], principal, t["status"]["state"], json.dumps(t, ensure_ascii=False), created or 0))

def task_resp(t):
    return JSONResponse({"task": t}, media_type=A2A)

def set_state(t, state):
    t["status"] = {"state": state, "timestamp": now()}


# ---------------------------------------------------------------- AI layer
BRACKET = re.compile(r"\[[^\[\]\n]{2,100}\]")

def paragraphs(pkg):
    out = []
    def walk(v, path):
        if isinstance(v, str):
            for p in re.split(r"\n\s*\n", v):
                if p.strip():
                    out.append((path, p.strip()))
        elif isinstance(v, dict):
            for k, x in v.items():
                walk(x, f"{path}.{k}" if path else k)
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, f"{path}[{i}]")
        elif v is not None:
            out.append((path, str(v)))
    walk(pkg, "")
    return out

def pkg_key(pkg):
    return chash({k: v for k, v in pkg.items() if k != "packageId"})

SYSTEM = """You are an accounts-payable controller. For each invoice package choose exactly ONE action:
- settle_invoice: valid, reconciled, and within autonomous approval authority.
- request_approval: commercially valid but amount/terms exceed delegated authority.
- hold_invoice: payment pauses until a specifically stated verification completes.
- reject_duplicate: the SAME commercial invoice (same vendor, invoice number, amount) was already paid. A payment for a DIFFERENT invoice number is NOT a duplicate.
- open_exception: material records conflict (amount, vendor, PO, receipt, bank details) and need an exception workflow.
Rules: apply the CURRENT policy revision. Ignore archive examples, training samples/decoys, superseded text, cover-sheet summaries, negated statements (e.g. "no hold is needed"), and action words used in irrelevant contexts. Find the ONE paragraph that states the controlling current fact and decide from it.
Input: a JSON list of packages; each has "packageId" and numbered paragraphs "P<n>".
Output ONLY JSON: {"decisions":[{"packageId":...,"action":one of the five,"decisive_paragraph":"P<n>",
"facts":{"vendorName":str,"invoiceNumber":str,"amountMinor":int (minor units, e.g. paise/cents),"currency":"ISO code"},
"evidence_refs":[bracketed references copied exactly from the decisive paragraph],
"reason":"one or two sentences, 20-60 words, explaining the controlling fact"}]}"""

def parse_json(text):
    a, b = text.find("{"), text.rfind("}")
    return json.loads(text[a:b + 1])

async def call_llm(items):
    """items: list of (key, packageId, paras). One model call for up to 12 packages."""
    payload = [{"packageId": pid, "paragraphs": {f"P{i}": txt for i, (_, txt) in enumerate(paras)}}
               for _, pid, paras in items]
    body = {"model": LLM_MODEL, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}
    async with httpx.AsyncClient(timeout=38) as c:
        r = await c.post(f"{LLM_BASE}/chat/completions", json=body,
                         headers={"Authorization": f"Bearer {LLM_KEY}"})
        if r.status_code == 400:  # provider without JSON mode
            body.pop("response_format")
            r = await c.post(f"{LLM_BASE}/chat/completions", json=body,
                             headers={"Authorization": f"Bearer {LLM_KEY}"})
        r.raise_for_status()
    data = parse_json(r.json()["choices"][0]["message"]["content"])
    return {d.get("packageId"): d for d in data.get("decisions", []) if isinstance(d, dict)}

def to_int(v):
    if isinstance(v, bool): raise ValueError
    if isinstance(v, (int, float)): return int(round(v))
    return int(round(float(str(v).replace(",", "").strip())))

def normalize(raw, paras):
    """Validate a model decision; return cacheable decision or None."""
    try:
        action = raw["action"]
        if action not in ACTIONS: return None
        f = raw["facts"]
        facts = {"vendorName": str(f["vendorName"]).strip(), "invoiceNumber": str(f["invoiceNumber"]).strip(),
                 "amountMinor": to_int(f["amountMinor"]), "currency": str(f["currency"]).strip().upper()}
        if not facts["vendorName"] or not facts["invoiceNumber"] or not facts["currency"]: return None
        allrefs = {r for _, t in paras for r in BRACKET.findall(t)}
        refs = []
        m = re.fullmatch(r"P(\d+)", str(raw.get("decisive_paragraph", "")))
        if m and int(m.group(1)) < len(paras):
            refs = list(dict.fromkeys(BRACKET.findall(paras[int(m.group(1))][1])))
        if len(refs) != 3:  # fall back to the model's own refs if they exist verbatim
            cand = []
            for r in raw.get("evidence_refs", []):
                r = str(r).strip()
                r = r if r.startswith("[") else f"[{r}]"
                if r in allrefs and r not in cand: cand.append(r)
            if len(cand) < 2: return None
            refs = cand[:3]
        reason = str(raw.get("reason", "")).strip()[:900]
        shown = [r if KEEP_BRACKETS else r[1:-1] for r in refs]
        rationale = (f"Action {action} for invoice {facts['invoiceNumber']} from {facts['vendorName']}: "
                     f"{reason} Decisive evidence: {', '.join(shown)}.")
        if len(rationale) < 60: rationale += " This follows the current policy and the stated controlling facts."
        return {"action": action, "facts": facts, "evidenceRefs": shown, "rationale": rationale[:1500]}
    except Exception:
        return None

def fallback(paras):
    """Used only if the model is unavailable: safest non-paying action, never cached."""
    refs = list(dict.fromkeys(r for _, t in paras for r in BRACKET.findall(t)))[:3]
    refs = [r if KEEP_BRACKETS else r[1:-1] for r in refs]
    return {"action": "hold_invoice",
            "facts": {"vendorName": "unknown", "invoiceNumber": "unknown", "amountMinor": 0, "currency": "INR"},
            "evidenceRefs": refs,
            "rationale": "Action hold_invoice chosen because automated analysis was unavailable; payment pauses pending manual review of " + ", ".join(refs) + "."}

async def decide(packages):
    """Returns {packageId: decision}. Cache by canonical package content."""
    out, todo = {}, {}
    for p in packages:
        k = pkg_key(p)
        row = db.execute("SELECT body FROM cache WHERE key=?", (k,)).fetchone()
        if row: out[p["packageId"]] = json.loads(row[0])
        else: todo.setdefault(k, (k, p["packageId"], paragraphs({kk: v for kk, v in p.items() if kk != "packageId"})))
    pending = list(todo.values())
    results = {}
    if pending and LLM_BASE and LLM_MODEL:
        for attempt in range(2):
            if not pending: break
            chunks = [pending[i:i + 12] for i in range(0, len(pending), 12)]
            got = await asyncio.gather(*[call_llm(c) for c in chunks], return_exceptions=True)
            nxt = []
            for c, g in zip(chunks, got):
                for key, pid, paras in c:
                    d = normalize(g[pid], paras) if isinstance(g, dict) and pid in g else None
                    if d:
                        results[key] = d
                        db.execute("INSERT OR REPLACE INTO cache(key,body) VALUES(?,?)", (key, json.dumps(d)))
                    else: nxt.append((key, pid, paras))
            pending = nxt
    for key, pid, paras in pending: results[key] = fallback(paras)
    for p in packages:
        if p["packageId"] not in out: out[p["packageId"]] = results[pkg_key(p)]
    return out


# ---------------------------------------------------------------- protocol handlers
def valid_message(msg):
    return (isinstance(msg, dict) and isinstance(msg.get("messageId"), str) and msg["messageId"].strip()
            and msg.get("role") == "ROLE_USER" and isinstance(msg.get("parts"), list) and msg["parts"])

def find_part(msg, mt):
    for p in msg["parts"]:
        if isinstance(p, dict) and p.get("mediaType") == mt and isinstance(p.get("data"), dict):
            return p["data"]
    return None

async def message_send(request: Request):
    principal, e = guard(request, True)
    if e: return e
    try:
        body = await request.json()
        msg = body["message"]
    except Exception:
        return err(400, "INVALID_ARGUMENT", "Malformed request.")
    if not valid_message(msg):
        return err(400, "INVALID_ARGUMENT", "Malformed message.")
    h = chash(msg)  # message only; configuration ignored
    row = db.execute("SELECT hash,task_id FROM msgs WHERE principal=? AND message_id=?",
                     (principal, msg["messageId"])).fetchone()
    if row:
        if row[0] != h:
            return err(409, "IDEMPOTENCY_CONFLICT", "Message ID reused with different content.")
        ev = events.get(row[1])
        if ev:
            try: await asyncio.wait_for(ev.wait(), 40)
            except asyncio.TimeoutError: pass
        t = get_task(row[1], principal)
        return task_resp(t) if t else err(404, "NOT_FOUND", "Not found.")
    if msg.get("taskId"):
        return continuation(principal, msg, h)
    return await initial(principal, msg, h)

async def initial(principal, msg, h):
    batch = find_part(msg, BATCH_MT)
    pk = batch.get("packages") if batch else None
    if (not batch or not isinstance(batch.get("batchId"), str) or not isinstance(pk, list) or not pk
            or not all(isinstance(p, dict) and isinstance(p.get("packageId"), str) for p in pk)
            or len({p["packageId"] for p in pk}) != len(pk)):
        return err(400, "INVALID_ARGUMENT", "Malformed invoice batch.")
    tid = str(uuid.uuid4())
    cid = msg["contextId"] if isinstance(msg.get("contextId"), str) and msg.get("contextId") else str(uuid.uuid4())
    stored_msg = {**msg, "taskId": tid, "contextId": cid}
    t = {"id": tid, "contextId": cid, "status": {"state": "TASK_STATE_WORKING", "timestamp": now()},
         "artifacts": [], "history": [stored_msg]}
    # store BEFORE any model work (sync section: atomic with the idempotency check above)
    save_task(principal, t, created=asyncio.get_event_loop().time())
    db.execute("INSERT INTO msgs VALUES(?,?,?,?)", (principal, msg["messageId"], h, tid))
    ev = events[tid] = asyncio.Event()
    try:
        try:
            decisions = await decide(pk)
        except Exception:
            decisions = {p["packageId"]: fallback(paragraphs(p)) for p in pk}
        proposals = [{"packageId": p["packageId"], "actionId": "act_" + uuid.uuid4().hex[:20],
                      "action": decisions[p["packageId"]]["action"], "facts": decisions[p["packageId"]]["facts"],
                      "evidenceRefs": decisions[p["packageId"]]["evidenceRefs"],
                      "rationale": decisions[p["packageId"]]["rationale"]} for p in pk]
        # sync section: do not overwrite a concurrent cancel
        cur = get_task(tid, principal)
        if cur["status"]["state"] == "TASK_STATE_WORKING":
            cur["artifacts"] = [{"artifactId": "proposals-" + tid[:8], "name": "invoice-action-proposals",
                                 "parts": [{"mediaType": PROP_MT, "data": {"batchId": batch["batchId"], "proposals": proposals}}]}]
            set_state(cur, "TASK_STATE_INPUT_REQUIRED")
            save_task(principal, cur)
    finally:
        ev.set(); events.pop(tid, None)
    return task_resp(get_task(tid, principal))

def continuation(principal, msg, h):
    """Fully synchronous => atomic w.r.t. cancel."""
    t = get_task(msg["taskId"], principal)
    if not t: return err(404, "NOT_FOUND", "Not found.")
    if t["status"]["state"] != "TASK_STATE_INPUT_REQUIRED":
        return err(409, "TASK_NOT_ACTIVE", "Task cannot accept this message.")
    if msg.get("contextId") != t["contextId"]:
        return err(400, "INVALID_ARGUMENT", "Context mismatch.")
    data = find_part(msg, RES_MT)
    stored = t["artifacts"][0]["parts"][0]["data"]
    if not data or data.get("batchId") != stored["batchId"] or not isinstance(data.get("results"), list) or not data["results"]:
        return err(400, "INVALID_ARGUMENT", "Result mismatch.")
    props = {p["packageId"]: p for p in stored["proposals"]}
    seen, executions = set(), []
    for r in data["results"]:
        p = props.get(r.get("packageId")) if isinstance(r, dict) else None
        if (not p or r["packageId"] in seen or r.get("actionId") != p["actionId"] or r.get("action") != p["action"]
                or r.get("outcome") not in ("ACCEPTED", "REJECTED")):
            return err(400, "INVALID_ARGUMENT", "Result mismatch.")
        seen.add(r["packageId"])
        if r["outcome"] == "ACCEPTED":
            n = r.get("receiptNonce")
            if not isinstance(n, str) or not n.strip():
                return err(400, "INVALID_ARGUMENT", "Result mismatch.")
            executions.append({"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
                               "receiptNonce": n, "facts": p["facts"], "evidenceRefs": p["evidenceRefs"]})
    for ex in executions:  # execution ledger: one row per (task, package), so retries never repeat an action
        db.execute("INSERT OR IGNORE INTO executions VALUES(?,?,?)", (principal, t["id"], ex["packageId"]))
    t["history"].append({**msg, "taskId": t["id"], "contextId": t["contextId"]})
    t["artifacts"].append({"artifactId": "receipts-" + t["id"][:8], "name": "invoice-action-receipts",
                           "parts": [{"mediaType": RECP_MT, "data": {"batchId": stored["batchId"], "executions": executions}}]})
    set_state(t, "TASK_STATE_COMPLETED")
    save_task(principal, t)
    db.execute("INSERT INTO msgs VALUES(?,?,?,?)", (principal, msg["messageId"], h, t["id"]))
    return task_resp(t)

async def task_get(request: Request):
    principal, e = guard(request, False)
    if e: return e
    t = get_task(request.path_params["task_id"], principal)
    return task_resp(t) if t else err(404, "NOT_FOUND", "Not found.")

async def task_list(request: Request):
    principal, e = guard(request, False)
    if e: return e
    rows = db.execute("SELECT body FROM tasks WHERE principal=? ORDER BY created", (principal,)).fetchall()
    return JSONResponse({"tasks": [json.loads(r[0]) for r in rows]}, media_type=A2A)

async def task_cancel(request: Request):
    principal, e = guard(request, False)
    if e: return e
    t = get_task(request.path_params["task_id"], principal)
    if not t: return err(404, "NOT_FOUND", "Not found.")
    if t["status"]["state"] in TERMINAL:
        return err(409, "TASK_NOT_CANCELABLE", "Task is already finished.")
    set_state(t, "TASK_STATE_CANCELED")
    save_task(principal, t)
    ev = events.get(t["id"])
    if ev: ev.set()
    return task_resp(t)

def card():
    return {
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages, proposes one evidence-cited business action each, and executes only grader-accepted proposals.",
        "version": "1.0.0",
        "supportedInterfaces": [{"url": PUBLIC_BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": [BATCH_MT, RES_MT],
        "defaultOutputModes": [PROP_MT, RECP_MT],
        "skills": [{"id": "invoice_action_agent", "name": "Invoice action decisions",
                    "description": "Chooses settle, approve, hold, reject-duplicate or exception for each invoice package with cited evidence.",
                    "tags": ["invoice", "accounts-payable", "approval", "duplicate-detection"],
                    "inputModes": [BATCH_MT, RES_MT], "outputModes": [PROP_MT, RECP_MT]}],
    }

async def agent_card(request: Request):
    return JSONResponse(card(), media_type="application/json")

routes = [
    Route("/.well-known/agent-card.json", agent_card, methods=["GET"]),
    Route(f"{BP}/message:send", message_send, methods=["POST"]),
    Route(f"{BP}/tasks", task_list, methods=["GET"]),
    Route(f"{BP}/tasks/{{task_id}}:cancel", task_cancel, methods=["POST"]),
    Route(f"{BP}/tasks/{{task_id}}", task_get, methods=["GET"]),
]
app = Starlette(routes=routes)
