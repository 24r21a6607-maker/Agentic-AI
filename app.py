import os, sys, json, re, time
import gradio as gr
from openai import OpenAI
 
# ---------- Config (set via environment variables) ----------
API_KEY = os.environ.get("LLM_API_KEY", "")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
 
 
def safe_json(text):
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except Exception:
        return {}
 
 
# ---------- Monitoring ----------
class MonitoringClient:
    def __init__(self):
        self.events = []
 
    def log(self, agent, seconds, ok=True):
        self.events.append({"agent": agent, "seconds": seconds, "ok": ok})
 
    def report(self):
        if not self.events:
            return "No LLM calls yet."
        total = sum(e["seconds"] for e in self.events)
        fails = sum(1 for e in self.events if not e["ok"])
        return f"LLM calls: {len(self.events)} | total time: {total:.1f}s | failures: {fails}"
 
 
# ---------- Base agent ----------
class BaseAgent:
    def __init__(self, name, system_prompt, monitor):
        self.name = name
        self.system_prompt = system_prompt
        self.monitor = monitor
 
    def ask(self, prompt, want_json=False, temperature=0.4):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        kwargs = {"model": MODEL, "messages": messages, "temperature": temperature}
        t0 = time.time()
        try:
            if want_json:
                try:
                    resp = client.chat.completions.create(
                        response_format={"type": "json_object"}, **kwargs)
                except Exception:
                    resp = client.chat.completions.create(**kwargs)
            else:
                resp = client.chat.completions.create(**kwargs)
            self.monitor.log(self.name, time.time() - t0, True)
            return resp.choices[0].message.content or ""
        except Exception:
            self.monitor.log(self.name, time.time() - t0, False)
            raise
 
 
# ---------- Placement coach agent ----------
class PlacementCoachAgent(BaseAgent):
    def __init__(self, monitor):
        super().__init__(
            "PlacementCoach",
            "You are an experienced campus-placement interviewer and career coach. "
            "Be specific, fair and encouraging.",
            monitor,
        )
 
    def generate_questions(self, company, role, round_type, n):
        prompt = (
            f'Create {n} {round_type} interview questions that {company} typically asks '
            f'for the role "{role}". Return JSON: {{"questions": ["...", "..."]}}'
        )
        data = safe_json(self.ask(prompt, want_json=True))
        return [q for q in data.get("questions", []) if isinstance(q, str)][:n]
 
    def evaluate_answer(self, company, question, answer):
        prompt = (
            f'An interviewer at {company} asked: "{question}"\n'
            f'Candidate answer: "{answer}"\n'
            "Score the answer out of 10. Return JSON with keys: score (number), "
            "strengths (list of short strings), improvements (list of short strings), "
            "model_answer (string, a concise ideal answer)."
        )
        data = safe_json(self.ask(prompt, want_json=True, temperature=0.2))
        try:
            data["score"] = max(0.0, min(10.0, float(data.get("score", 0))))
        except Exception:
            data["score"] = 0.0
        return data
 
 
# ---------- Evaluation & scoring module ----------
def readiness_report(history, monitor):
    if not history:
        return "Answer at least one question to get a readiness score."
    avg = sum(h["score"] for h in history) / len(history)
    pct = round(avg * 10)
    level = "Interview-ready" if pct >= 75 else "Almost there" if pct >= 50 else "Needs more practice"
    rows = "\n".join(f"- {h['question']} : **{h['score']:.1f}/10**" for h in history)
    return (f"### Placement readiness: {pct}% ({level})\n"
            f"Answers evaluated: {len(history)}\n\n{rows}\n\n_{monitor.report()}_")
 
 
monitor = MonitoringClient()
coach = PlacementCoachAgent(monitor)
 
 
# ---------- UI handlers ----------
def ui_generate(company, role, round_type, n):
    try:
        qs = coach.generate_questions(company, role, round_type, int(n))
    except Exception as e:
        return gr.update(choices=[], value=None), f"Error: {e}"
    if not qs:
        return gr.update(choices=[], value=None), "Could not generate questions. Try again."
    return (gr.update(choices=qs, value=qs[0]),
            "\n".join(f"{i + 1}. {q}" for i, q in enumerate(qs)))
 
 
def ui_evaluate(company, question, answer, history):
    history = history or []
    if not question or not (answer or "").strip():
        return "Pick a question and write an answer first.", history
    try:
        fb = coach.evaluate_answer(company, question, answer)
    except Exception as e:
        return f"Error: {e}", history
    history = history + [{"question": question, "score": fb["score"]}]
    strengths = "\n".join(f"- {s}" for s in fb.get("strengths", []))
    improve = "\n".join(f"- {s}" for s in fb.get("improvements", []))
    md = (f"### Score: {fb['score']:.1f}/10\n**Strengths**\n{strengths}\n\n"
          f"**To improve**\n{improve}\n\n**Model answer**\n{fb.get('model_answer', '')}")
    return md, history
 
 
def ui_report(history):
    return readiness_report(history, monitor)
 
 
with gr.Blocks(title="Placement-Ready Career Coach") as demo:
    gr.Markdown("# Placement-Ready Career Coach\nGenerate company-specific interview "
                "questions, practise answers and get scored feedback.")
    history = gr.State([])
    with gr.Row():
        company = gr.Textbox(label="Company", value="TCS")
        role = gr.Textbox(label="Role", value="Software Engineer")
        round_type = gr.Dropdown(["HR", "Technical", "Behavioral"], value="HR", label="Round")
        n = gr.Slider(3, 10, value=5, step=1, label="Number of questions")
    gen_btn = gr.Button("Generate questions", variant="primary")
    q_list = gr.Markdown()
    question = gr.Dropdown(label="Pick a question to practise", choices=[])
    answer = gr.Textbox(label="Your answer", lines=5)
    eval_btn = gr.Button("Evaluate my answer")
    feedback = gr.Markdown()
    report_btn = gr.Button("Show readiness score")
    report = gr.Markdown()
 
    gen_btn.click(ui_generate, [company, role, round_type, n], [question, q_list])
    eval_btn.click(ui_evaluate, [company, question, answer, history], [feedback, history])
    report_btn.click(ui_report, [history], [report])
 
 
from fastapi import FastAPI
 
api = FastAPI()
 
 
@api.get("/health")
def health():
    return {"status": "ok"}
 
 
app = gr.mount_gradio_app(api, demo, path="/")
