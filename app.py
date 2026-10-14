import re
import logging
import streamlit as st
from datetime import datetime

from database import init_database, save_chat, log_unanswered, save_feedback, log_error
from rag_engine import get_engine
from llm_engine import llm_available
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger("app")

st.set_page_config(
    page_title="ALRN Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    init_database()
except Exception as e:
    st.error(f"⚠️ Gagal menginisialisasi database: {e}")
    logger.critical(f"init_database gagal: {e}")

if "sessions"     not in st.session_state: st.session_state.sessions = {"Sesi 1": []}
if "active"       not in st.session_state: st.session_state.active = "Sesi 1"
if "chip_trigger" not in st.session_state: st.session_state.chip_trigger = None

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:         #f7f7f5;
  --sb-bg:      #f0efec;
  --sb-border:  #e4e3df;
  --bot-bubble: #ffffff;
  --usr-bubble: #e8e7e3;
  --border:     #e4e3df;
  --text:       #1c1c1a;
  --text-2:     #57574f;
  --text-3:     #a0a098;
  --accent:     #1c1c1a;
  --radius:     14px;
  --sb-width:   260px;
  --header-h:   54px;
  --sb-top-h:   148px;
  --sb-bot-h:   56px;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stApp"] {
  font-family: 'Sora', sans-serif !important;
  background:  var(--bg) !important;
  overflow:    hidden !important;
}

/* Sembunyikan chrome Streamlit + tombol collapse */
#MainMenu, footer, header,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebarHeader"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
.stDeployButton,
button[kind="header"],
[data-testid="stSidebar"] > div > button,
section[data-testid="stSidebar"] > div > button {
  display:        none !important;
  visibility:     hidden !important;
  width:          0 !important;
  height:         0 !important;
  pointer-events: none !important;
}

/* ── Sidebar container ── */
[data-testid="stSidebar"] {
  background:   var(--sb-bg) !important;
  border-right: 1px solid var(--sb-border) !important;
  min-width:    var(--sb-width) !important;
  max-width:    var(--sb-width) !important;
  height:       100vh !important;
  overflow:     hidden !important;
  position:     relative !important;
}

[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"],
[data-testid="stSidebarUserContent"] > div {
  height:   100vh !important;
  padding:  0 !important;
  margin:   0 !important;
  overflow: visible !important;
  position: relative !important;
  display:  block !important;
}

/* ── 3 Zona sidebar (dibuat via JS) ── */
.sb-zone-top {
  position:      absolute !important;
  top:           0 !important;
  left:          0 !important;
  right:         0 !important;
  background:    var(--sb-bg) !important;
  z-index:       10 !important;
  overflow:      hidden !important;
}

.sb-zone-mid {
  position:   absolute !important;
  top:        var(--sb-top-h) !important;
  bottom:     0 !important;
  left:       0 !important;
  right:      0 !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
  background: var(--sb-bg) !important;
  padding:    4px 8px !important;
  padding-bottom: 70px !important;
  z-index:    5 !important;
}
.sb-zone-mid::-webkit-scrollbar       { width: 3px; }
.sb-zone-mid::-webkit-scrollbar-thumb { background: #ccc; border-radius: 3px; }

.sb-zone-bottom {
  position:   absolute !important;
  bottom:     0 !important;
  left:       0 !important;
  right:      0 !important;
  height:     var(--sb-bot-h) !important;
  border-top: 1px solid var(--sb-border) !important;
  background: var(--sb-bg) !important;
  padding:    8px 8px 12px !important;
  z-index:    999 !important;
}

/* ── Isi zona atas ── */
.sb-brand {
  padding:       14px 16px 10px !important;
  border-bottom: 1px solid var(--sb-border) !important;
  text-align: center;
}
.sb-brand-title {
  font-size: 2rem; font-weight: 700;
  color: var(--text); letter-spacing: -0.3px; margin: 0 0 2px 0;
}
.sb-brand-sub { font-size: 1rem; color: var(--text-3); margin: 0; }

.sb-newchat-wrap { padding: 8px 8px 4px !important; }

.sb-history-label {
  font-size: 1.00rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 1.1px;
  color: var(--text-3); padding: 4px 16px 6px;
}

/* ── Sidebar buttons ── */
[data-testid="stSidebarUserContent"] .stButton > button {
  font-family: 'Sora', sans-serif !important;
  font-size: 0.775rem !important; font-weight: 500 !important;
  border-radius: 8px !important; border: 1px solid transparent !important;
  background: transparent !important; color: var(--text-2) !important;
  text-align: left !important; width: 100% !important;
  padding: 7px 10px !important; line-height: 1.4 !important;
  transition: background 0.12s !important;
}
[data-testid="stSidebarUserContent"] .stButton > button:hover {
  background: var(--sb-border) !important; color: var(--text) !important;
}

/* ── Main area ── */
.main .block-container {
  padding: 0 !important; max-width: 100% !important;
  height: 100vh !important; display: flex !important;
  flex-direction: column !important; overflow: hidden !important;
}

/* ── Header fixed full width ── */
.chat-header {
  position: fixed !important; top: 0 !important;
  left: var(--sb-width) !important; right: 0 !important;
  height: var(--header-h) !important; background: #ffffff !important;
  border-bottom: 1px solid var(--border) !important;
  display: flex !important; align-items: center !important;
  justify-content: center !important; flex-direction: column !important;
  gap: 2px !important; box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
  z-index: 50 !important;
}
.hdr-title { font-size: 0.93rem; font-weight: 700; letter-spacing: -0.3px; color: var(--text); }
            
/* ── Chat area ── */
.chat-scroll-area {
  padding-bottom: 140px !important; /* biar ada space buat input floating */
}
.chat-scroll-area::-webkit-scrollbar       { width: 4px; }
.chat-scroll-area::-webkit-scrollbar-thumb { background: #dddcd8; border-radius: 4px; }
.msg-wrap { max-width: 700px; margin: 0 auto; padding: 0 24px; }

.msg-group { margin-bottom: 18px; }
.msg-meta  { font-size: 0.61rem; color: var(--text-3); font-family: 'JetBrains Mono', monospace; margin-bottom: 5px; }
.msg-meta.user { text-align: right; }
.msg-row       { display: flex; align-items: flex-end; gap: 9px; }
.msg-row.user  { flex-direction: row-reverse; }

.avatar {
  width: 28px; height: 28px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.72rem; flex-shrink: 0; border: 1px solid var(--border);
}
.avatar.bot  { background: var(--accent); color: #fff; font-size: 0.85rem; }
.avatar.user { background: var(--usr-bubble); color: var(--text-2); }

.bubble {
  max-width: min(78%, 520px); padding: 10px 14px;
  border-radius: var(--radius); font-size: 0.845rem;
  line-height: 1.68; word-break: break-word;
}
.bubble.bot {
  background: var(--bot-bubble); color: var(--text);
  border-bottom-left-radius: 4px; border: 1px solid var(--border);
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.bubble.user {
  background: var(--usr-bubble); color: var(--text);
  border-bottom-right-radius: 4px; border: 1px solid #dddcd8;
}
.bubble strong { font-weight: 600; color: #111; }

.rag-badge {
  display: inline-block; font-size: 0.57rem; font-weight: 600;
  padding: 2px 7px; border-radius: 20px; margin-bottom: 6px;
  letter-spacing: 0.4px; font-family: 'JetBrains Mono', monospace;
}
.rag-on      { background: #edf7ed; color: #2e7d32; border: 1px solid #c8e6c9; }
.rag-off     { background: #fff8e1; color: #e65100; border: 1px solid #ffe082; }
.llm-on      { background: #ede7f6; color: #4527a0; border: 1px solid #d1c4e9; }
.error-badge { background: #fce4ec; color: #b71c1c; border: 1px solid #ef9a9a; }

.empty-state {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  padding: 0px 24px 20px; gap: 8px; text-align: center;
}
.empty-icon  { font-size: 2.2rem; opacity: 0.15; }
.empty-title { font-size: 1rem; font-weight: 700; color: var(--text-2); }
.empty-sub   { font-size: 0.76rem; color: var(--text-3); line-height: 1.75; max-width: 300px; }

.input-hint {
  flex-shrink: 0; text-align: center; font-size: 0.6rem;
  color: var(--text-3); font-family: 'JetBrains Mono', monospace;
  padding: 3px 0 4px;
}
/* ── CHAT INPUT FLOATING (FIX TOTAL) ── */
[data-testid="stChatInput"] {
  position: fixed !important;
  bottom: 20px !important;

  /* geser ke kanan karena ada sidebar */
  left: calc(var(--sb-width) + 40px) !important;
  right: 40px !important;

  background: transparent !important;
  border: none !important;
  z-index: 100 !important;
}

/* container dalamnya */
[data-testid="stChatInput"] > div {
  background: #ffffff !important;
  border: 1px solid var(--border) !important;
  border-radius: 20px !important;
  box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
  padding: 6px !important;
}
[data-testid="stChatInput"] textarea {
  font-family: 'Sora', sans-serif !important; font-size: 0.855rem !important;
  border-radius: 12px !important; background: #ffffff !important;
}

.main .stButton > button {
  font-family: 'Sora', sans-serif !important; font-size: 0.775rem !important;
  font-weight: 500 !important; border-radius: 10px !important;
  border: 1px solid var(--border) !important; background: #ffffff !important;
  color: var(--text-2) !important; text-align: left !important;
  width: 100% !important; padding: 10px 13px !important;
  line-height: 1.45 !important; transition: background 0.12s ease !important;
}
.main .stButton > button:hover {
  background: var(--usr-bubble) !important; color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load engine ───────────────────────────────────────────────
if "engine_ready" not in st.session_state:
    try:
        with st.spinner("Memuat knowledge base..."):
            engine = get_engine()
        st.session_state.engine_ready = engine.is_ready
        if not engine.is_ready:
            st.warning("⚠️ Knowledge base tidak dapat dimuat sepenuhnya.")
    except Exception as e:
        st.session_state.engine_ready = False
        st.error(f"❌ Gagal memuat RAG engine: {e}")
        logger.critical(f"RAG engine gagal: {e}")


def now_str(): return datetime.now().strftime("%H:%M")
def cur_msgs(): return st.session_state.sessions.get(st.session_state.active, [])

def format_html(content: str) -> str:
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
    html = re.sub(r'^\#{1,3} (.+)$', r'<strong>\1</strong>', html, flags=re.MULTILINE)
    html = re.sub(r'^\* (.+)$',  r'• \1', html, flags=re.MULTILINE)
    html = re.sub(r'^\+ (.+)$',  r'&nbsp;&nbsp;◦ \1', html, flags=re.MULTILINE)
    html = re.sub(r'^\- (.+)$',  r'• \1', html, flags=re.MULTILINE)
    html = re.sub(r'^\d+\. (.+)$', lambda m: f'• {m.group(1)}', html, flags=re.MULTILINE)
    html = html.replace('\n', '<br>')
    return html

def do_send(text: str):
    if not text.strip(): return
    msgs = st.session_state.sessions.get(st.session_state.active)
    if msgs is None:
        st.error("Sesi chat tidak valid.")
        return

    msgs.append({"role": "user", "content": text, "time": now_str()})
    try: save_chat(st.session_state.active, "user", text)
    except Exception as e: logger.warning(f"Gagal simpan user: {e}")

    chat_history_for_llm = [
        {"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]}
        for m in msgs[:-1] if m.get("content")
    ]

    answer, docs, is_answered, used_llm, is_error = "Maaf, terjadi kesalahan.", [], False, False, False
    try:
        engine = get_engine()
        with st.spinner("Sedang memproses..."):
            result = engine.answer(text, chat_history=chat_history_for_llm)
        if len(result) == 4: answer, docs, is_answered, used_llm = result
        else: answer, docs, is_answered = result
    except MemoryError:
        answer, is_error = "⚠️ Server kehabisan memori.", True
    except Exception as e:
        logger.error(f"do_send error: {type(e).__name__}: {e}")
        answer, is_error = f"⚠️ Kesalahan: `{type(e).__name__}`.", True
        try: log_error("do_send", e)
        except: pass

    try: save_chat(st.session_state.active, "bot", answer)
    except Exception as e: logger.warning(f"Gagal simpan bot: {e}")

    if not is_answered and not is_error:
        try: log_unanswered(text)
        except Exception as e: logger.warning(f"log_unanswered gagal: {e}")

    msgs.append({
        "role": "bot", "content": answer, "time": now_str(),
        "is_answered": is_answered, "used_llm": used_llm,
        "is_error": is_error, "n_docs": len(docs),
    })


# ═════════════════════════════════════════════════════════════
# SIDEBAR
# Pendekatan: render semua elemen dengan id unik,
# JS mengambil elemen-elemen itu dan merestrukturisasi ke 3 zona
# ═════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Blok brand (zona atas) ────────────────────────────────
    st.markdown("""
    <div id="_sb_brand">
      <div class="sb-brand">
        <div class="sb-brand-title">ALRN</div>
        <div class="sb-brand-sub">AI Academic Assistant</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tombol Chat Baru ──────────────────────────────────────
    # Bungkus dengan id agar JS bisa menemukannya
    st.markdown('<span id="_sb_newchat_before"></span>', unsafe_allow_html=True)
    if st.button("＋  Chat Baru", key="new_chat_btn"):
        n = f"Sesi {len(st.session_state.sessions) + 1}"
        st.session_state.sessions[n] = []
        st.session_state.active = n
        st.rerun()
    st.markdown('<span id="_sb_newchat_after"></span>', unsafe_allow_html=True)

    # ── Label Riwayat ─────────────────────────────────────────
    st.markdown("""
    <div id="_sb_label">
      <div class="sb-history-label">Riwayat</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Marker mulai daftar sesi ──────────────────────────────
    st.markdown('<span id="_sb_mid_start"></span>', unsafe_allow_html=True)

    # ── Daftar sesi ───────────────────────────────────────────
    for name in list(st.session_state.sessions.keys()):
        msgs_sb = st.session_state.sessions[name]
        preview = "Chat kosong"
        for m in msgs_sb:
            if m["role"] == "user":
                preview = m["content"][:30] + ("…" if len(m["content"]) > 30 else "")
                break
        bullet = "●" if name == st.session_state.active else "○"
        if st.button(f"{bullet}  {preview}", key=f"sess_{name}"):
            st.session_state.active = name
            st.rerun()

    # ── Marker akhir daftar sesi ──────────────────────────────
    st.markdown('<span id="_sb_mid_end"></span>', unsafe_allow_html=True)

    # ── Tombol Hapus Chat ─────────────────────────────────────
    st.markdown('<span id="_sb_del_before"></span>', unsafe_allow_html=True)
    if st.button("🗑️  Hapus Chat Ini", key="clear_btn"):
        if len(st.session_state.sessions) > 1:
            del st.session_state.sessions[st.session_state.active]
            n = f"Sesi {len(st.session_state.sessions) + 1}"
            st.session_state.sessions[n] = []
            st.session_state.active = n
        else:
            st.session_state.sessions[st.session_state.active] = []
        st.rerun()
    st.markdown('<span id="_sb_del_after"></span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CHIP TRIGGER
# ─────────────────────────────────────────────────────────────
if st.session_state.chip_trigger:
    do_send(st.session_state.chip_trigger)
    st.session_state.chip_trigger = None


# ═════════════════════════════════════════════════════════════
# HEADER — fixed, dari tepi sidebar ke ujung kanan
# ═════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="chat-header">
  <div class="hdr-title">ALRN | AI Assistant Campus</div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# CHAT AREA — scrollable
# ═════════════════════════════════════════════════════════════
msgs = cur_msgs()
st.markdown('<div class="chat-scroll-area"><div class="msg-wrap">', unsafe_allow_html=True)

if not msgs:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">💠</div>
      <div class="empty-title">Apa yang ingin kamu tanyakan?</div>
      <div class="empty-sub">Tanyakan tentang jurusan, pendaftaran,
      fasilitas, dosen, atau kalender akademik.</div>
    </div>
    """, unsafe_allow_html=True)

    left, center, right = st.columns([1,2,1])

    with center:
        c1, c2 = st.columns(2)
    with c1:
        if st.button("🎓  Daftar jurusan", key="qa1"):
            st.session_state.chip_trigger = "Jurusan apa saja yang tersedia?"
            st.rerun()
        if st.button("🏛️  Fasilitas kampus", key="qa3"):
            st.session_state.chip_trigger = "Fasilitas apa saja di kampus?"
            st.rerun()
    with c2:
        if st.button("📋  Cara pendaftaran", key="qa2"):
            st.session_state.chip_trigger = "Bagaimana cara mendaftar ke kampus?"
            st.rerun()
        if st.button("💰  Biaya kuliah", key="qa4"):
            st.session_state.chip_trigger = "Berapa biaya pendaftaran kuliah?"
            st.rerun()

else:
    for i, msg in enumerate(msgs):
        role    = msg["role"]
        content = msg["content"]
        mtime   = msg.get("time", "")
        is_last = (i == len(msgs) - 1)

        if role == "user":
            st.markdown(f"""
            <div class="msg-group">
              <div class="msg-meta user">You· {mtime}</div>
              <div class="msg-row user">
                <div class="avatar user">👤</div>
                <div class="bubble user">{content}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            is_answered = msg.get("is_answered", True)
            used_llm    = msg.get("used_llm", False)
            is_error    = msg.get("is_error", False)
            n_docs      = msg.get("n_docs", 0)

            if is_error:
                badge = '<span class="rag-badge error-badge">⚠ Error</span>'
            elif used_llm:
                badge = '<span class="rag-badge llm-on">✦ RAG + LLM</span>'
            elif is_answered:
                badge = f'<span class="rag-badge rag-on">✓ RAG · {n_docs} docs</span>'
            else:
                badge = '<span class="rag-badge rag-off">⚠ Tidak ditemukan</span>'

            st.markdown(f"""
            <div class="msg-group">
              <div class="msg-meta">ALRN· {mtime}</div>
              <div class="msg-row">
                <div class="avatar bot">💠</div>
                <div class="bubble bot">{badge}<br>{format_html(content)}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if is_last:
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                fb_c1, fb_c2, _ = st.columns([1, 1, 5])
                with fb_c1:
                    if st.button("👍  Membantu", key=f"fb_good_{i}"):
                        try:
                            save_feedback(st.session_state.active, "", content, 5)
                            st.toast("Terima kasih! 😊")
                        except Exception as e:
                            logger.warning(f"Gagal feedback: {e}")
                with fb_c2:
                    if st.button("👎  Kurang", key=f"fb_bad_{i}"):
                        try:
                            save_feedback(st.session_state.active, "", content, 2)
                            st.toast("Terima kasih, akan kami tingkatkan!")
                        except Exception as e:
                            logger.warning(f"Gagal feedback: {e}")

st.markdown('</div></div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# INPUT
# ═════════════════════════════════════════════════════════════
prompt = st.chat_input("Tanya ALRN...")
if prompt:
    do_send(prompt)
    st.rerun()