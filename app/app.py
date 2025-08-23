# app/app.py
# UI de Gradio para sugerencia de acordes (igual a tu notebook)
from pathlib import Path
import json, re, math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr

# ------------------ Rutas portables ------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR   = ROOT / "data"
PROC_DIR   = DATA_DIR / "processed"   # aquí están los JSON de vocab
MODELS_DIR = ROOT / "models"          # aquí están config.json + checkpoint_best.pt

# ------------------ Carga de vocab y config ------------------
with open(PROC_DIR / "chord_to_idx.json", "r", encoding="utf-8") as f:
    CH2IDX = json.load(f)
with open(PROC_DIR / "idx_to_chord.json", "r", encoding="utf-8") as f:
    IDX2CH = json.load(f)
PAD = CH2IDX["[PAD]"]; UNK = CH2IDX["[UNK]"]

with open(MODELS_DIR / "config.json", "r", encoding="utf-8") as f:
    CFG_JSON = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = int(CFG_JSON.get("max_len", 112))

# ------------------ Modelo (mismos nombres que el checkpoint) ------------------
@dataclass
class ModelConfig:
    vocab_size: int; pad_idx: int; unk_idx: int; max_len: int = MAX_LEN
    d_model: int = 256; n_layers: int = 4; n_heads: int = 8; d_ff: int = 1024; dropout: float = 0.1

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__(); self.pos_emb = nn.Embedding(max_len, d_model)
    def forward(self, x):
        B,T = x.size(); pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B,T)
        return self.pos_emb(pos)

class CausalTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_idx)
        self.pos_emb = PositionalEmbedding(cfg.max_len, cfg.d_model)
        enc = nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.n_heads,
                                         dim_feedforward=cfg.d_ff, dropout=cfg.dropout,
                                         activation="gelu", batch_first=True, norm_first=True)
        self.trf = nn.TransformerEncoder(enc, num_layers=cfg.n_layers)
        self.drop = nn.Dropout(cfg.dropout)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.apply(self._init_w)
    def _init_w(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
    def _causal_mask(self, T, device):
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    def forward(self, x, attention_mask=None):
        B,T = x.shape
        h = self.drop(self.tok_emb(x) + self.pos_emb(x))
        causal = self._causal_mask(T, x.device)
        key_pad = (attention_mask==0) if attention_mask is not None else None
        h = self.trf(h, mask=causal, src_key_padding_mask=key_pad)
        return self.lm_head(h)

CFG = ModelConfig(
    vocab_size=len(CH2IDX), pad_idx=PAD, unk_idx=UNK,
    max_len=int(CFG_JSON.get("max_len", MAX_LEN)),
    d_model=int(CFG_JSON.get("d_model", 256)),
    n_layers=int(CFG_JSON.get("n_layers", 4)),
    n_heads=int(CFG_JSON.get("n_heads", 8)),
    d_ff=int(CFG_JSON.get("d_ff", 1024)),
    dropout=float(CFG_JSON.get("dropout", 0.1))
)

MODEL = CausalTransformer(CFG).to(device)
CKPT  = torch.load(MODELS_DIR / "checkpoint_best.pt", map_location=device)
MODEL.load_state_dict(CKPT["model_state"]); MODEL.eval()

# ------------------ Helpers de notación y transposición ------------------
NOTE_SH = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
VAL = {n:i for i,n in enumerate(NOTE_SH)}; VAL.update({'Db':1,'Eb':3,'Gb':6,'Ab':8,'Bb':10})
REV = {i:n for i,n in enumerate(NOTE_SH)}

def norm_ch(s:str)->str:
    return s.replace("min","m").replace("maj","M").replace("s","#").strip()

class ExtChord:
    __slots__=("root","quality","ext")
    def __init__(self, c:str):
        m = re.match(r"^([A-G][b#]?)(.*)$", c)
        if not m: self.root=self.quality=self.ext=None; return
        self.root = m.group(1); rem = m.group(2)
        if rem.startswith("m"): self.quality="m"; self.ext=rem[1:]
        else: self.quality=""; self.ext=rem
    def as_str(self): return f"{self.root}{self.quality}{self.ext}"

def transpose_one(c:str, interval:int)->str:
    e = ExtChord(c); rv = VAL.get(e.root, None)
    if rv is None: return c
    new_root = REV[(rv - interval) % 12]
    return f"{new_root}{e.quality}{e.ext}"

def auto_interval_to_C(seq:List[str]):
    roots = [VAL.get(ExtChord(c).root) for c in seq if VAL.get(ExtChord(c).root) is not None]
    if not roots: return 0
    mean = int(round(sum(roots)/len(roots))) % 12
    return mean

def encode(seq:List[str]):
    ids = [CH2IDX.get(c, UNK) for c in seq][:MAX_LEN]
    if len(ids)<MAX_LEN: ids += [PAD]*(MAX_LEN-len(ids))
    return ids

def attn_mask(ids:List[int]): return [1 if t!=PAD else 0 for t in ids]

# ------------------ Predicción con clave fija (igual a notebook) ------------------
@torch.no_grad()
def predict_topk_with_interval(seq_orig:List[str], k:int=10, temperature:float=1.0, interval:Optional[int]=None):
    seq_norm = [norm_ch(x) for x in seq_orig]
    use_interval = auto_interval_to_C(seq_norm) if interval is None else int(interval)
    seq_C = [transpose_one(x, use_interval) for x in seq_norm]
    ids = encode(seq_C); msk = attn_mask(ids)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    a = torch.tensor([msk], dtype=torch.long, device=device)
    T = int(sum(msk)) if any(msk) else 1
    logits = MODEL(x, attention_mask=a)[0, T-1, :]
    if temperature>0: logits = logits / float(temperature)
    probs = F.softmax(logits, dim=-1)
    top = torch.topk(probs, k=min(k, probs.numel()))
    idxs = top.indices.tolist(); pr = [float(v) for v in top.values.tolist()]
    chords_C = [IDX2CH.get(str(i), f"<{i}>") for i in idxs]
    chords_orig = [transpose_one(c, -use_interval) for c in chords_C]
    return list(zip(chords_orig, pr))

# ------------------ UI (réplica visual 2x5 + barra con separadores) ------------------
# === Raíces unificadas (enarmónicos) ===
ROOT_OPTIONS = [
    "C (Do)",
    "C#/Db (Do#/Reb)",
    "D (Re)",
    "D#/Eb (Re#/Mib)",
    "E (Mi)",
    "F (Fa)",
    "F#/Gb (Fa#/Solb)",
    "G (Sol)",
    "G#/Ab (Sol#/Lab)",
    "A (La)",
    "A#/Bb (La#/Sib)",
    "B (Si)"
]
# Etiqueta unificada -> raíz canónica que usará el modelo (preferimos sostenidos)
ROOT_LABEL_TO_CANON = {
    "C (Do)": "C",
    "C#/Db (Do#/Reb)": "C#",
    "D (Re)": "D",
    "D#/Eb (Re#/Mib)": "D#",
    "E (Mi)": "E",
    "F (Fa)": "F",
    "F#/Gb (Fa#/Solb)": "F#",
    "G (Sol)": "G",
    "G#/Ab (Sol#/Lab)": "G#",
    "A (La)": "A",
    "A#/Bb (La#/Sib)": "A#",
    "B (Si)": "B",
}
# === Variantes (diccionario exacto y ordenado) ===
# Orden: mayor, m, 7, maj7, dim, dim7, aug, add9, madd9
VARIANTS = [
    ("mayor",  ""),     # mayor -> sin sufijo
    ("menor",  "m"),    # menor
    ("7",      "7"),
    ("m7",     "m7"),
    ("maj7",   "maj7"),
    ("dim",    "dim"),
    ("dim7",   "dim7"),
    ("aug",    "aug"),
    ("add9",   "add9"),
    ("madd9",  "madd9"),
    ("sus2",   "sus2"),
    ("sus4",   "sus4"),
    ("msus2",  "msus2"),
    ("msus4",  "msus4")
]
def render_seq_bar(seq): return " | ".join(seq) if seq else ""

def medals_topk(chords_probs):
    medals = ["🥇","🥈","🥉"] + [""]*7
    out = []
    for i,(c,p) in enumerate(chords_probs[:10]):
        out.append(f"{medals[i]} {c} ({p*100:.2f}%)".strip())
    return (out + [""]*10)[:10]

def fanout_button_labels(labels10):
    labels10 = (labels10 + [""]*10)[:10]
    return [gr.update(value=l) for l in labels10]

def labels_topk(seq, temp, key_interval):
    top = predict_topk_with_interval(seq, k=10, temperature=temp, interval=key_interval)
    return medals_topk(top), top

def add_by_dropdown(seq, key_interval, root_label, var_label, temp):
    root = ROOT_LABEL_TO_CANON[root_label]   # usa la raíz canónica (C#, F#, etc.)
    suf  = dict(VARIANTS)[var_label]
    chord = f"{root}{suf}"

    seq2 = seq + [chord]
    key2 = VAL[ExtChord(chord).root] if key_interval is None else key_interval
    labels, top = labels_topk(seq2, temp, key2)
    return [seq2, key2, render_seq_bar(seq2), *fanout_button_labels(labels), top]

def add_from_top(seq, key_interval, top, which, temp):
    if not top or which >= len(top):
        labels, top2 = labels_topk(seq, temp, key_interval) if seq else ([""]*10, [])
        return [seq, key_interval, render_seq_bar(seq), *fanout_button_labels(labels), top2]
    chosen = top[which][0]
    seq2 = seq + [chosen]
    key2 = key_interval if key_interval is not None else VAL.get(ExtChord(chosen).root, 0)
    labels, top2 = labels_topk(seq2, temp, key2)
    return [seq2, key2, render_seq_bar(seq2), *fanout_button_labels(labels), top2]

def pop_one(seq, key_interval, temp):
    seq2 = seq[:-1] if seq else []
    key2 = None if len(seq2)==0 else key_interval
    labels, top = labels_topk(seq2, temp, key2) if seq2 else ([""]*10, [])
    return [seq2, key2, render_seq_bar(seq2), *fanout_button_labels(labels), top]

def reset_all():
    return [[], None, "", *fanout_button_labels([""]*10), []]

def on_temp_change(seq, key_interval, t):
    if not seq: return [*fanout_button_labels([""]*10), []]
    labels, top = labels_topk(seq, t, key_interval)
    return [*fanout_button_labels(labels), top]

def init_buttons():
    labels, top = labels_topk([], 1.0, None)
    return [*fanout_button_labels(labels), top]

with gr.Blocks(title="armonIA — Chord Prediction") as demo:
    # --- CSS para grid 2x5 estable ---
    gr.HTML("""
    <style>
    #row_top1, #row_top2 { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
    #row_top1 button, #row_top2 button { width: 100%; }
    .title-center { text-align:center; margin: 0 0 6px 0; }
    .subtitle { text-align:center; opacity:.85; margin: 0 0 16px 0; }
    .desc { font-size:0.95rem; line-height:1.4rem; }
    </style>
    """)

    # --- Título + descripción bilingüe ---
    gr.HTML("""
    <h1 class="title-center">armonIA</h1>
    <h3 class="subtitle">Predicción de acordes con Transformer · Transformer‑based Chord Prediction</h3>
    <div class="desc">
      <p><b>ES:</b> Construye una secuencia de acordes y el modelo sugerirá el siguiente. Selecciona raíz y variante, añade desde el Top‑10 o manualmente, ajusta la temperatura y explora las opciones más probables.</p>
      <p><b>EN:</b> Build a chord sequence and the model will suggest the next one. Choose root and quality, add from the Top‑10 or manually, tweak temperature, and explore the most likely options.</p>
    </div>
    """)

    # --- Controles y layout (idéntico a tu versión previa) ---
    with gr.Row():
        # Izquierda: selects + acciones
        with gr.Column(scale=6):
            root_dd = gr.Dropdown(ROOT_OPTIONS, value="C (Do)", label="Raíz / Root")
            var_dd  = gr.Dropdown([v for v,_ in VARIANTS], value="mayor", label="Variante / Quality")

            with gr.Row():
                btn_add = gr.Button("Añadir / Add", variant="primary")
                btn_pop = gr.Button("Borrar / Undo")
                btn_rst = gr.Button("Reset")
        # Derecha: Top-10 (2x5) + barra "Secuencia construida"
        with gr.Column(scale=6):
            gr.Markdown("#### Añadir / Add")
            with gr.Row(elem_id="row_top1"):
                btns_row1 = [gr.Button("", scale=1) for _ in range(5)]
            with gr.Row(elem_id="row_top2"):
                btns_row2 = [gr.Button("", scale=1) for _ in range(5)]
            btns = btns_row1 + btns_row2
            seq_bar = gr.Textbox(value="", label="Secuencia construida / Built sequence", interactive=False)

    # Slider de temperatura
    with gr.Row():
        temp = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Temperature")

    # Estados
    seq_state = gr.State([])      # acordes en tonalidad original
    key_state = gr.State(None)    # intervalo fijo (se fija al primer acorde)
    top_state = gr.State([])      # [(ch, prob), ...]

    # Eventos y lógica (sin cambios)
    demo.load(init_buttons, inputs=None, outputs=[*btns, top_state])

    btn_add.click(add_by_dropdown,
                  inputs=[seq_state, key_state, root_dd, var_dd, temp],
                  outputs=[seq_state, key_state, seq_bar, *btns, top_state])

    btn_pop.click(lambda s,k,t: pop_one(s,k,t),
                  inputs=[seq_state, key_state, temp],
                  outputs=[seq_state, key_state, seq_bar, *btns, top_state])

    btn_rst.click(lambda: reset_all(),
                  inputs=None,
                  outputs=[seq_state, key_state, seq_bar, *btns, top_state])

    for i, b in enumerate(btns):
        b.click(lambda s,k,top,t,i=i: add_from_top(s, k, top, i, t),
                inputs=[seq_state, key_state, top_state, temp],
                outputs=[seq_state, key_state, seq_bar, *btns, top_state])

    temp.change(on_temp_change,
                inputs=[seq_state, key_state, temp],
                outputs=[*btns, top_state])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

