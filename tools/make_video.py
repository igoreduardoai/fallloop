"""Draft narrated demo video: matplotlib slides + piper TTS (LJSpeech voice, public-domain data) + ffmpeg.
Output: video/FALLLOOP-2026-10-24[-NO-INTRO].mp4. INTRO=<clip> prepends Igor's on-camera opening; AWS_CLIP=<clip> adds the live AWS capture."""
import json, os, subprocess, sys, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); V = os.path.join(ROOT, "video"); T = os.path.join(V, "_build"); os.makedirs(T, exist_ok=True)
PIPER = os.environ.get("PIPER", "/workspace/ai_media_engine/.venv/bin/piper"); VOICE = os.environ.get("VOICE", "/workspace/ai_media_engine/voices/en_US-ljspeech-high.onnx")
E = json.load(open(os.path.join(ROOT, "results", "EVAL-FALLLOOP-2026-10-24.json")))
te, tf, b4 = E["clip_TEST"]["LS"], E["clip_TEST"]["FALL"], E["latency_benchmark"]["runs"][0]
FOOT = "Research demonstration - not a medical device - a human decides.  Video data: GMDCSA-24 (Alam et al. 2024), CC BY 4.0"

def slide(name, title, lines=(), image=None, bg="#0f1419"):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100); fig.patch.set_facecolor(bg); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.04, 0.93, title, color="white", fontsize=30, fontweight="bold", va="top")
    if image:
        im = mpimg.imread(image); a2 = fig.add_axes([0.05, 0.1, 0.9, 0.72]); a2.imshow(im); a2.axis("off")
    for i, l in enumerate(lines): ax.text(0.05, 0.78 - i*0.085, l, color="#e6e6e6", fontsize=21, va="top")
    ax.text(0.5, 0.025, FOOT, color="#ffb3b3", fontsize=11, ha="center")
    p = os.path.join(T, name + ".png"); fig.savefig(p, facecolor=bg); plt.close(fig); return p

fd = E["fall_alert_delay_s"]
G = json.load(open(os.path.join(ROOT, "results", "aws-graviton-2026-10-24", "incident.json")))["summary"]
AWS_CLIP = os.environ.get("AWS_CLIP", os.path.join(V, "_capture", "aws-dashboard-clip.mp4"))
INTRO = os.environ.get("INTRO")  # Igor's ~20 s on-camera opening (optional); prepended when given
SEG = [
 ("02", slide("02", "FallLoop", ["Agentic fall-alert review", "OpenCV 5  +  AWS Graviton", "", "OpenCV AI Competition 2026"]),
  "FallLoop: agentic fall-alert review, built with OpenCV 5 and AWS.", None),
 ("03", slide("03", "The problem", ["A person lying still on the floor may need help.", "A person asleep on a bed is lying still too.", "FallLoop flags POSSIBLE falls for a human reviewer.", "It never makes the call itself."]),
  "A person lying still on the floor is the cue a caregiver looks for. But a person sleeping on a bed is lying still too. FallLoop flags possible falls for a human reviewer. It is a research demonstration, not a medical device.", None),
 ("04", slide("04", "Perceive - decide - act", image=os.path.join(ROOT, "docs", "agent-workflow.png")),
  "Every frame, OpenCV 5 decodes the video, runs a YOLO26 pose model through c v 2 d n n, and measures motion with Farneback optical flow. An agent reads that output and chooses the next vision action: lower the detection threshold, re-detect a person on a zoomed region, wait for more frames, ask for a new view, or escalate to a human.", None),
 ("05", None, "Here a person falls and lies still. After half a second, the agent escalates a possible fall. At night, keypoints are weak, so the agent re-detects on a zoomed crop, and lowers its threshold when the person disappears. The third person is asleep on a bed. Also lying still.", os.path.join(ROOT, "demo", "run", "annotated.mp4")),
 ("06a", slide("06a", "Human review queue", image=os.path.join(ROOT, "docs", "screens", "dashboard-pending.png")),
  "The dashboard shows the video, the agent's timeline, and the review queue. Nothing is decided until a human confirms, downgrades, or dismisses it.", None),
 ("06b", slide("06b", "Only a human decides", image=os.path.join(ROOT, "docs", "screens", "dashboard-after-review.png")),
  "Here the reviewer dismisses the person sleeping on the bed.", None),
 ("07", slide("07", "Results (GMDCSA-24, held-out actors)", [f"Lying + still alerts: precision {te['precision']:.2f}, recall {te['recall']:.2f}", f"Judged as fall-only: precision {tf['precision']:.2f}, recall {tf['recall']:.2f}", f"Median fall-to-alert time: {fd['median']:.1f} s (n={fd['n']})", f"x86 CPU: about {b4['throughput_fps']:.0f} fps (4 threads); AWS Graviton Lambda: {G['throughput_fps']:.1f} fps", "Staged indoor clips, 4 actors. Not clinical accuracy."]),
  f"I evaluated on G M D C S A 24, a public fall dataset, choosing thresholds on two actors and testing on the other two. On held-out actors, lying and still alerts reached {te['precision']:.2f} precision and {te['recall']:.2f} recall, with a median of {fd['median']:.1f} seconds from the fall to the alert. On a CPU, the pipeline runs at about {b4['throughput_fps']:.0f} frames per second, and at {G['throughput_fps']:.1f} on a Graviton Lambda.", None),
 ("08", slide("08", "Where it fails", ["Detector loses people once they are on the floor", "Push-ups and bending flagged as lying", "No bed-versus-floor context", "Rooms with several people not tested"]),
  "It fails too. People lost by the detector once on the floor, push-ups flagged as lying, and no bed versus floor context. Rooms with several people were not tested.", None),
 ("09", slide("09", "AWS architecture (deployed, us-east-2)", image=os.path.join(ROOT, "docs", "architecture.png")),
  "On AWS, uploads to a private S 3 bucket trigger a Graviton Lambda running the same OpenCV 5 pipeline, and a second Lambda serves the dashboard, with a one dollar budget alert and a teardown script.", None),
] + ([("09b", None, "This is the live dashboard on AWS, showing a run processed by the Graviton Lambda. The reviewer jumps to the person on the bed and dismisses it as a false alert.", AWS_CLIP)] if os.path.exists(AWS_CLIP) else []) + [
 ("10", slide("10", "FallLoop", ["Vision that knows when to ask a human.", "", "Thank you."]), "FallLoop. Vision that knows when to ask a human. Thank you.", None),
]

def dur(p): return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p]).decode().strip())
parts = []
for sid, img, text, vid in SEG:
    wav = os.path.join(T, sid + ".wav")
    subprocess.run([PIPER, "--model", VOICE, "--output_file", wav], input=text.encode(), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    d = dur(wav) + 0.8; out = os.path.join(T, sid + ".mp4")
    if vid:
        vd = dur(vid); d = max(d, vd)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", vid, "-i", wav, "-filter_complex",
                        f"[0:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30,tpad=stop_mode=clone:stop_duration={max(d-vd,0):.2f},setsar=1[v];[1:a]apad[a]",
                        "-map", "[v]", "-map", "[a]", "-t", f"{d:.2f}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "44100", "-ac", "1", out], check=True)
    else:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", img, "-i", wav, "-filter_complex", "[0:v]scale=1280:720,fps=30,setsar=1[v];[1:a]apad[a]",
                        "-map", "[v]", "-map", "[a]", "-t", f"{d:.2f}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "44100", "-ac", "1", out], check=True)
    parts.append(out); print(sid, round(d, 1), flush=True)
lst = os.path.join(T, "list.txt"); open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
if INTRO:  # normalise Igor's clip to the same format and put it first
    io = os.path.join(T, "01-intro.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", INTRO, "-filter_complex", "[0:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30,setsar=1[v];[0:a]aresample=44100[a]",
                    "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "44100", "-ac", "1", io], check=True)
    parts.insert(0, io); open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
final = os.path.join(V, "FALLLOOP-2026-10-24" + ("" if INTRO else "-NO-INTRO") + ".mp4")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", final], check=True)
print("final", final, round(dur(final), 1), "s")
