"""Dashboard Lambda (python3.12, arm64, zip) behind a Lambda Function URL. NOT DEPLOYED YET.
GET /                       -> dashboard page (same index.html as the local Flask demo)
GET /runs                   -> list of processed runs
GET /incident.json?run=ID   -> incident JSON (+ human decisions)
GET /media/<file>?run=ID    -> 302 redirect to a short-lived presigned S3 URL (buckets stay private)
POST /api/review?run=ID     -> store a human reviewer decision (results/ID/reviews.json)
Research demonstration, not a medical device."""
import base64, json, os, re, time
import boto3

from botocore.config import Config
_REGION = os.environ.get("AWS_REGION", "us-east-2")
# Regional endpoint + SigV4 so presigned media URLs work for buckets outside us-east-1 (no 307 redirect).
s3 = boto3.client("s3", region_name=_REGION, endpoint_url=f"https://s3.{_REGION}.amazonaws.com", config=Config(signature_version="s3v4")); BUCKET = os.environ["OUT_BUCKET"]
HTML = open(os.path.join(os.path.dirname(__file__), "index.html")).read()
DECISIONS = {"confirm_now": "Confirmed by human reviewer: needs help now", "downgrade_later": "Downgraded by human reviewer: check later",
             "dismiss": "Dismissed by human reviewer (false alert)"}
SAFE = re.compile(r"^[A-Za-z0-9_.\-/]+$")


def _resp(code, body, ctype="application/json", headers=None):
    return {"statusCode": code, "headers": {"Content-Type": ctype, "Cache-Control": "no-store", **(headers or {})},
            "body": body if isinstance(body, str) else json.dumps(body)}


def _get_json(key, default=None):
    try: return json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    except s3.exceptions.NoSuchKey: return default


def _runs():
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix="results/", Delimiter="/")
    return sorted([p["Prefix"].split("/")[1] for p in r.get("CommonPrefixes", [])], reverse=True)


def handler(event, _ctx):
    path = event.get("rawPath", "/"); method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    q = event.get("queryStringParameters") or {}
    run = q.get("run") or next(iter(_runs()), None)
    if path == "/": return _resp(200, HTML, "text/html; charset=utf-8")
    if path == "/runs": return _resp(200, {"runs": _runs()})
    if not run or not SAFE.match(run) or ".." in run: return _resp(404, {"error": "no run"})
    base = f"results/{run}/"
    if path == "/incident.json":
        inc = _get_json(base + "incident.json")
        if inc is None: return _resp(404, {"error": "not found"})
        rv = _get_json(base + "reviews.json", {})
        for it in inc["review_queue"]:
            if it["id"] in rv: it.update(rv[it["id"]])
        return _resp(200, inc)
    if path.startswith("/media/"):
        fn = path[len("/media/"):]
        if not SAFE.match(fn) or ".." in fn: return _resp(400, {"error": "bad name"})
        url = s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": base + fn}, ExpiresIn=600)
        return {"statusCode": 302, "headers": {"Location": url}, "body": ""}
    if path == "/api/review" and method == "POST":
        body = event.get("body") or "{}"
        if event.get("isBase64Encoded"): body = base64.b64decode(body).decode()
        d = json.loads(body)
        if d.get("decision") not in DECISIONS or not SAFE.match(str(d.get("id", ""))): return _resp(400, {"error": "bad request"})
        rv = _get_json(base + "reviews.json", {})
        rv[d["id"]] = {"status": "reviewed", "decision": d["decision"], "final_label": DECISIONS[d["decision"]],
                       "reviewer": "web reviewer", "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        s3.put_object(Bucket=BUCKET, Key=base + "reviews.json", Body=json.dumps(rv).encode(), ContentType="application/json")
        return _resp(200, rv[d["id"]])
    return _resp(404, {"error": "not found"})
