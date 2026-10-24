"""Local incident dashboard (Flask). Run:  python web/app.py --run demo/run  then open http://127.0.0.1:8050
Shows the annotated video, the agent's incident event list and the human-review queue. Reviewer decisions are
stored in <run>/reviews.json. The same static page also works without Flask (S3/Lambda hosting later):
it then reads incident.json next to it and keeps decisions in the browser's localStorage.
Research demonstration, not a medical device."""
import argparse, json, os, time
from flask import Flask, jsonify, request, send_from_directory, abort

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=None)
RUN = None
DECISIONS = {"confirm_now": "Confirmed by human reviewer: needs help now", "downgrade_later": "Downgraded by human reviewer: check later",
             "dismiss": "Dismissed by human reviewer (false alert)"}


def _reviews():
    p = os.path.join(RUN, "reviews.json")
    return json.load(open(p)) if os.path.exists(p) else {}


@app.get("/")
def index(): return send_from_directory(os.path.join(HERE, "static"), "index.html")


@app.get("/incident.json")
def incident():
    inc = json.load(open(os.path.join(RUN, "incident.json"))); rv = _reviews()
    for it in inc["review_queue"]:
        if it["id"] in rv: it.update(rv[it["id"]])
    return jsonify(inc)


@app.get("/media/<path:fn>")
def media(fn):
    if ".." in fn: abort(400)
    return send_from_directory(RUN, fn)


@app.post("/api/review")
def review():
    d = request.get_json(force=True) or {}
    if d.get("decision") not in DECISIONS or not d.get("id"): abort(400)
    rv = _reviews()
    rv[d["id"]] = {"status": "reviewed", "decision": d["decision"], "final_label": DECISIONS[d["decision"]],
                   "reviewer": (d.get("reviewer") or "local reviewer")[:60], "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    json.dump(rv, open(os.path.join(RUN, "reviews.json"), "w"), indent=1)
    return jsonify(rv[d["id"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default=os.path.join(HERE, "..", "demo", "run"))
    ap.add_argument("--port", type=int, default=8050); a = ap.parse_args()
    RUN = os.path.abspath(a.run); app.run("127.0.0.1", a.port, debug=False)
