"""Offline test of aws/dashboard/app.py with an in-memory fake S3 (no AWS account needed).
Run: python tests/test_dashboard_lambda.py <local_run_dir>"""
import io, json, os, sys, types

RUN = sys.argv[1] if len(sys.argv) > 1 else "demo/run"
store = {}
for dp, _, fns in os.walk(RUN):
    for fn in fns:
        store["results/demo-run/" + os.path.relpath(os.path.join(dp, fn), RUN)] = open(os.path.join(dp, fn), "rb").read()


class NoSuchKey(Exception): pass


class FakeS3:
    exceptions = types.SimpleNamespace(NoSuchKey=NoSuchKey)
    def get_object(self, Bucket, Key):
        if Key not in store: raise NoSuchKey(Key)
        return {"Body": io.BytesIO(store[Key])}
    def put_object(self, Bucket, Key, Body, ContentType): store[Key] = Body
    def list_objects_v2(self, Bucket, Prefix, Delimiter):
        pre = sorted({k.split("/")[1] for k in store if k.startswith(Prefix)})
        return {"CommonPrefixes": [{"Prefix": f"results/{p}/"} for p in pre]}
    def generate_presigned_url(self, op, Params, ExpiresIn): return f"https://example-presigned/{Params['Key']}?exp={ExpiresIn}"


import boto3
boto3.client = lambda *_a, **_k: FakeS3()
os.environ["OUT_BUCKET"] = "fake"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "aws", "dashboard"))
import app

ev = lambda path, method="GET", body=None, q=None: {"rawPath": path, "requestContext": {"http": {"method": method}}, "queryStringParameters": q, "body": body}
assert app.handler(ev("/"), None)["statusCode"] == 200
runs = json.loads(app.handler(ev("/runs"), None)["body"])["runs"]; assert runs == ["demo-run"], runs
inc = json.loads(app.handler(ev("/incident.json", q={"run": "demo-run"}), None)["body"])
assert app.handler(ev("/media/" + inc["video"], q={"run": "demo-run"}), None)["statusCode"] == 302
assert app.handler(ev("/media/../x", q={"run": "demo-run"}), None)["statusCode"] == 400
if inc["review_queue"]:
    rid = inc["review_queue"][0]["id"]
    r = app.handler(ev("/api/review", "POST", json.dumps({"id": rid, "decision": "downgrade_later"}), {"run": "demo-run"}), None)
    assert r["statusCode"] == 200
    inc2 = json.loads(app.handler(ev("/incident.json", q={"run": "demo-run"}), None)["body"])
    assert inc2["review_queue"][0]["status"] == "reviewed"
assert app.handler(ev("/api/review", "POST", json.dumps({"id": "R001", "decision": "auto_label"}), {"run": "demo-run"}), None)["statusCode"] == 400
print("dashboard lambda offline test: OK", {"runs": runs, "review_items": len(inc["review_queue"])})
