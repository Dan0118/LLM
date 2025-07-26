#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
olmoe_text_only.py  · 纯文本生成（1–100 段）

- 本地 OLMoE，baseline / CoT 两组
- 判重：与同轮历史完全一致则重试
- 仅生成，不评估；支持断点续跑 & Early‑Stop
- tqdm 进度条显示段落平均时延
"""

import os, re, json, time, logging, signal, sys
from pathlib import Path
from collections import defaultdict
import pandas as pd, numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ========== 0. 全局参数 ==========
ROUNDS        = 200
PARA_START    = 1          # 1–100 段
PARAS         = 100
MAX_NEW_TOK   = 450
MAX_RETRY_GEN = 10
MAX_RETRY_DUP = 5  # 专门为重复设置的重试次数

STYLES = [("baseline", "low"), ("cot", "low")]

RESULT_DIR   = Path("/home/linux/olmoe/results/baseline_vs_cot_experiment_olmoe")
RAW_CSV      = RESULT_DIR / "olmoe_text_generation.csv"
LOG_FILE     = RESULT_DIR / "generation.log"
EARLY_STOP_FILE = RESULT_DIR / "early_stop_status.json"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ========== 1. 日志 ==========
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE)])
logger = logging.getLogger(__name__)
signal.signal(signal.SIGINT, lambda s, f: (logger.warning("Ctrl‑C 捕获，退出"), sys.exit(0)))

# ========== 2. 本地 OLMoE ==========
OLMOE_DIR = "/home/linux/olmoe/OLMoE-1B-7B-0125-Instruct"  # ← 修改为实际权重路径
logger.info("Loading OLMoE from %s", OLMOE_DIR)
olmoe_tok = AutoTokenizer.from_pretrained(
    OLMOE_DIR,
    use_fast=True,          # 
    local_files_only=True,   # 
    trust_remote_code=True   # 
)
olmoe_mod = AutoModelForCausalLM.from_pretrained(
    OLMOE_DIR,
    torch_dtype=torch.float16,
    device_map={"":0},
    local_files_only=True,
    trust_remote_code=True   # 同理
).eval()

SYSTEM_PROMPT = (
    "You are OLMo 2, a helpful, imaginative and concise AI assistant created by "
    "the Allen Institute for AI."
)

# ========== 3. Token 预算 ==========
_GPT2 = AutoTokenizer.from_pretrained("gpt2")
num_tokens = lambda t: len(_GPT2.encode(t, add_special_tokens=False))
MAX_CTX, SAFE_HEAD, HIST_BUDGET = 4096, 128, 3500
def _trim(prompt:str,budget:int)->str:
    lines=prompt.splitlines()
    while lines and num_tokens("\n".join(lines))>budget:
        lines.pop(0)
    return "\n".join(lines)

# ========== 4. Early‑Stop ==========
class EarlyStop:
    def __init__(s,p:Path): s.p=p; s.stopped=s._load()
    def _load(s):
        if s.p.exists():
            try: return set(json.load(open(s.p))['stopped_groups'])
            except Exception: pass
        return set()
    def _save(s): json.dump({'stopped_groups':list(s.stopped)},open(s.p,'w'),indent=2)
    def is_stop(s,tag): return tag in s.stopped
    def stop(s,tag,r,p):
        if tag not in s.stopped:
            s.stopped.add(tag); s._save()
            logger.warning("🛑 EARLY‑STOP %s at R%d P%d",tag,r+1,p)
early = EarlyStop(EARLY_STOP_FILE)

# ========== 5. 词库 & 缺词 ==========
TOKEN_RE=re.compile(r"[A-Za-z']+"); words=lambda t:TOKEN_RE.findall(t.lower())
import inflect, nltk; from nltk.stem import WordNetLemmatizer
infl,wn=inflect.engine(),WordNetLemmatizer()
def _vars(w):
    b=w.lower(); v={b,infl.plural(b) or b+"s", infl.singular_noun(b) or b}
    if len(b)>2:v.update({b+"s",b+"ed",b[:-1]+"ing" if b.endswith("e") else b+"ing"})
    v.update({wn.lemmatize(b,pos="n"),wn.lemmatize(b,pos="v")}); return {x for x in v if x}
def missing(lst,txt):
    tok={t.lower() for t in TOKEN_RE.findall(txt)}
    tok|={wn.lemmatize(t,pos="n") for t in tok}
    tok|={wn.lemmatize(t,pos="v") for t in tok}
    return [w for w in lst if not _vars(w)&tok]

df_vocab=pd.read_csv("vocab.csv"); rng=np.random.default_rng(42)
def sample_five(r):
    s={r.choice(df_vocab[df_vocab.category.str.startswith("noun")].word),
       r.choice(df_vocab[df_vocab.category=="verb_action"].word)}
    while len(s)<5: s.add(r.choice(df_vocab.word))
    return list(s)
GLOBAL_FIVE=[sample_five(rng) for _ in range(ROUNDS)]

# ========== 6. 调 OLMoE ==========
def _fmt(p): return f"<|endoftext|><|user|>\n{SYSTEM_PROMPT}\n\n{p}\n<|assistant|>\n"
@torch.inference_mode()
def call_olmoe(prompt,max_new=MAX_NEW_TOK):
    prompt=_trim(prompt,HIST_BUDGET)
    if num_tokens(prompt)+max_new>MAX_CTX-SAFE_HEAD:
        prompt=_trim(prompt,int(HIST_BUDGET*0.9))
    inp=olmoe_tok(_fmt(prompt),return_tensors="pt",truncation=True)
    inp={k:v.to(olmoe_mod.device) for k,v in inp.items()}
    out=olmoe_mod.generate(**inp,max_new_tokens=max_new,temperature=1.2,
                           do_sample=True,top_p=0.85,
                           pad_token_id=olmoe_tok.eos_token_id)
    return olmoe_tok.decode(out[0],skip_special_tokens=True).split("<|assistant|>")[-1].strip()

# ========== 7. 生成段落 ==========
# JSON_RE=re.compile(r"\{.*?\}",re.S)
# def extract_json(raw):
#     m=JSON_RE.search(raw)
#     if not m: return ""
#     try: return (json.loads(m.group(0)).get("output") or "").strip()
#     except json.JSONDecodeError: return ""

from sentence_transformers import SentenceTransformer, util
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
def embed(t: str):
    return embedder.encode(t, convert_to_tensor=True, normalize_embeddings=True)

# ---------- util: cosine 判重 ----------
def is_sem_dup(vec, hist_vecs, thresh=0.90):
    """vec 与历史向量最大余弦 > thresh ?"""
    if not hist_vecs:
        return False
    from sentence_transformers.util import cos_sim
    return bool((cos_sim(vec, hist_vecs).max() > thresh).item())


# ---------- 主函数 ----------
def generate_para(five, hist, style, eff):
    """
    返回 (paragraph, total_tries, duplicate_tries)
    - 语义重复: 余弦 > 0.9 视为复读
    - 缺词 / 超长: 反馈后重生
    - 模型只需输出纯文本，无 JSON
    """
    # —— 先把历史段落一次性嵌入，避免重复 encode
    hist_vecs = torch.stack([embed(h) for h in hist]) if hist else None

    total_tries, dup_tries = 0, 0
    MISSING_TEMPLATE = (
        "❗ You **MUST** include ALL five words. "
        "Missing last time: {words}\n"
    )

    def build_prompt(miss_words=None, duplicate_warning=False):
        # —— 共用头部 —— #
        head = (
            "Write an imaginative English paragraph **under 180 words** "
            f"that uses **ALL** of these words (any tense): "
            f"{', '.join(five)}.\n\n"
        )
        # 最近 6 段历史（只 baseline / cot 自己的）——再多会稀释 token 预算
        if hist:
            head += "Previous paragraphs (latest 6):\n"
            head += "\n".join(f"- {s}" for s in hist[-6:]) + "\n\n"

        # —— 额外提醒 —— #
        extra = ""
        if duplicate_warning:
            extra += "⚠️ Your last try was **too similar** to earlier text. " \
                     "Change setting, perspective and sentence rhythm!\n"
        if miss_words:
            extra += MISSING_TEMPLATE.format(
                words=", ".join(f"**{w}**" for w in miss_words))
        if extra:
            extra += "\n"

        # —— baseline / cot 分叉 —— #
        if style == "baseline":
            body = (
                extra +
                "Focus on originality: switch genre, setting or narrator if needed.\n\n"
                "Return ONLY the paragraph text – no headings, no explanations."
            )
        else:               # ----- CoT 精简版 -----
            body = (
                extra +
                "Think **silently** in three quick steps:\n"
                "1) Recall themes & tone already used.\n"
                "2) Pick a fresh, contrasting idea & weave the five words in.\n"
                "3) Write the final paragraph, ≤180 words, no repetitions.\n\n"
                "Return ONLY the paragraph text – do NOT reveal steps."
            )
        return head + body
    # ---------- end build_prompt ----------

    miss_feedback = None
    for attempt in range(1, MAX_RETRY_GEN + 1):
        total_tries = attempt

        prompt = build_prompt(
            miss_words=miss_feedback,
            duplicate_warning=(dup_tries > 0)
        )
        raw = call_olmoe(prompt)
        txt = raw.strip()

        # --------- 校验 ---------
        # 1. 缺词
        miss = missing(five, txt)
        if miss:
            miss_feedback = miss      # 下一轮告知缺词
            continue

        # 2. 超长
        if len(words(txt)) > 180:
            miss_feedback = None      # 不再重复缺词提示
            dup_tries += 1            # 也算一次“内容不合规”重试
            continue

        # 3. 语义重复
        vec = embed(txt)
        if is_sem_dup(vec, hist_vecs):
            dup_tries += 1
            if dup_tries >= MAX_RETRY_DUP:
                return None, total_tries, dup_tries
            continue

        # ✅ 通过全部检查
        if hist_vecs is not None:
            hist_vecs = torch.cat([hist_vecs, vec.unsqueeze(0)], dim=0)
        return txt, total_tries, dup_tries

    # 全部尝试失败
    return None, total_tries, dup_tries

    # 主生成循环
    feedback = ""
    for attempt in range(1, MAX_RETRY_GEN + 1):
        total_tries = attempt
        
        # 构建 prompt（传入当前重复次数）
        prompt = build_prompt(duplicate_count=duplicate_tries)
        if feedback:
            prompt = prompt.replace("Requirements:", f"{feedback}\nRequirements:")
        
        raw = call_olmoe(prompt)
        txt = extract_json(raw) if style == "cot" else (extract_json(raw) or raw)
        if "\n\n" in txt: 
            txt = [p.strip() for p in txt.split("\n\n") if p.strip()][-1]
        
        # 检查缺词
        mis = missing(five, txt)
        if mis:
            feedback = f"Missing words: {', '.join(mis)}. You MUST include ALL five words."
            continue
            
        # 检查长度
        if len(words(txt)) > 180:
            feedback = "Paragraph exceeds 180 words. Make it shorter."
            continue
            
        # 检查重复
        if txt in hist:
            duplicate_tries += 1
            if duplicate_tries >= MAX_RETRY_DUP:
                # 达到重复上限，返回失败
                logger.warning(f"Reached max duplicate retries ({MAX_RETRY_DUP}) for {style}")
                return None, total_tries, duplicate_tries
            feedback = f"This paragraph is IDENTICAL to a previous one! (Duplicate #{duplicate_tries})"
            continue
            
        # 成功生成
        return txt, total_tries, duplicate_tries
    
    # 所有尝试失败
    return None, total_tries, duplicate_tries

# ========== 8. 历史恢复 ==========
def load_prev(prev_df,rid):
    hist=defaultdict(list); max_idx=0
    if prev_df is not None and not prev_df.empty:
        ok=prev_df[(prev_df.status=="ok")&(prev_df.round==rid)]
        if not ok.empty:
            max_idx=int(ok.idx.max())
            for tag,g in ok.groupby("exp"):
                hist[tag].extend(g.para_text.tolist())
    return hist,max_idx

# ========== 9. CSV ==========
def append_rows(rows):
    pd.DataFrame(rows).to_csv(RAW_CSV, mode="a",
                              header=not RAW_CSV.exists(), index=False)

# ========== 10. 单轮 ==========
def run_round(rid: int, start: int, prev_df):
    five = GLOBAL_FIVE[rid]

    # 载入历史、确定起点
    hist, max_idx = load_prev(prev_df, rid)
    start = max(start, max_idx + 1)

    # 进度条准备
    pbar = tqdm(total=PARAS - start + 1, desc=f"Round {rid+1}")
    last_time = {"low_base": 0.0, "low_cot": 0.0}      # 记录最新一次耗时

    for idx in range(start, PARAS + 1):
        rows = []; info = []

        for sty, eff in STYLES:
            tag = f"{eff}_{'base' if sty == 'baseline' else 'cot'}"

            # 若已 Early-Stop
            if early.is_stop(tag):
                rows.append(dict(exp=tag, round=rid, idx=idx,
                                 status="early_stop", para_text="",
                                 words_used=json.dumps(five), 
                                 tries=0, duplicate_tries=0))
                info.append(f"{tag}=STOP")
                last_time[tag] = 0.0          # 不算时间
                continue

            # ⏱ 计时开始
            t0 = time.time()

            # 生成段落
            para, total_tries, dup_tries = generate_para(five, hist[tag], sty, eff)

            # ⏱ 计时结束
            dt = time.time() - t0
            last_time[tag] = dt               # 记录该风格本段耗时

            if para is None:
                early.stop(tag, rid, idx)
                rows.append(dict(exp=tag, round=rid, idx=idx,
                                 status="early_stop", para_text="",
                                 words_used=json.dumps(five), 
                                 tries=total_tries,
                                 duplicate_tries=dup_tries))
                info.append(f"{tag}=STOP")
            else:
                hist[tag].append(para)
                rows.append(dict(exp=tag, round=rid, idx=idx, status="ok",
                                 para_text=para, words_used=json.dumps(five),
                                 tries=total_tries,
                                 duplicate_tries=dup_tries))
                info.append(f"{tag}=OK(t{total_tries},d{dup_tries})")

        append_rows(rows)

        # 更新 tqdm：显示最新 baseline / CoT 耗时
        pbar.update(1)
        pbar.set_postfix(
            base=f"{last_time['low_base']:.2f}s",
            cot =f"{last_time['low_cot']:.2f}s"
        )

        # 若两组都停了，提前结束
        if all(early.is_stop(f"{e}_{'base' if s=='baseline' else 'cot'}")
               for s, e in STYLES):
            pbar.close()
            logger.warning("All groups stopped — experiment halted.")
            return False

    pbar.close()
    return True

# ========== 11. 主入口 ==========
def main():
    logger.info("🚀 OLMoE text‑only 1‑100 start")
    done_round,prev_rows={},{}
    if RAW_CSV.exists():
        df=pd.read_csv(RAW_CSV)
        for rid,g in df.groupby("round"):
            done_round[rid]=int(g.idx.max())
            prev_rows[rid]=g
    for rid in range(ROUNDS):
        start=done_round.get(rid,PARA_START-1)+1
        if start>PARAS: continue
        cont=run_round(rid,start,prev_rows.get(rid))
        if not cont: break
    logger.info("✔ Finished — CSV at %s", RAW_CSV)
    print("🎉 All done. →", RAW_CSV)

if __name__=="__main__":
    main()
