import os
import re
import logging

logger = logging.getLogger("llm_engine")

# ── Daftar query conversational yang tidak perlu RAG ─────────────────────────
CONVERSATIONAL_PATTERNS = re.compile(
    r"^\s*(ya|yaa+|iya|oke|ok|oke+|lanjut|lanjutkan|boleh|baik|"
    r"tentu|silahkan|silakan|benar|betul|terima kasih|makasih|thx|"
    r"thanks|sip|siap|noted|paham|ngerti|mengerti|hmm+|oh+|"
    r"halo|hai|hello|hi|hey)\s*[!?.]*\s*$",
    re.IGNORECASE,
)

# ── Kata kunci yang harus muncul agar RAG context dianggap relevan ────────────
CAMPUS_KEYWORDS = re.compile(
    r"(mahasiswa|dosen|jurusan|program studi|biaya|kuliah|beasiswa|"
    r"pendaftaran|kampus|universitas|president|fakultas|akademik|"
    r"uts|uas|krs|jadwal|semester|lab|laboratorium|asrama|nim|nip|"
    r"mata kuliah|ipk|transkrip|wisuda|sidang|skripsi|prodi)",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """Kamu adalah ALRN Chatbot, asisten akademik virtual untuk kampus President University (perguruan tinggi di Indonesia).

Tugasmu adalah membantu mahasiswa dan calon mahasiswa dengan pertanyaan seputar:
- Jurusan dan program studi yang tersedia
- Proses dan jalur pendaftaran mahasiswa baru
- Biaya kuliah, beasiswa, dan bantuan finansial
- Fasilitas kampus (laboratorium, perpustakaan, asrama, dll.)
- Data dan informasi dosen
- Kalender akademik (jadwal UTS, UAS, KRS, liburan, dll.)
- Program perkuliahan dan kebijakan akademik

Panduan menjawab:
1. Jawab dalam Bahasa Indonesia yang ramah, jelas, dan informatif.
2. Jika ada konteks dari knowledge base (RAG), gunakan konteks tersebut sebagai referensi utama.
3. JANGAN mengarang fakta spesifik yang tidak ada dalam konteks. Jika konteks tidak cukup untuk menjawab, katakan bahwa kamu tidak memiliki informasi tersebut.
4. Format jawaban menggunakan markdown (bold untuk judul, bullet points untuk daftar).
5. Jawab ringkas dan langsung ke poin.
6. Akhiri jawaban dengan kalimat penutup yang ramah dan membantu.

ATURAN KETAT — WAJIB DIPATUHI:
- JANGAN PERNAH mengasumsikan atau menyimpulkan identitas pengguna (nama, NIM, jurusan, dll.) berdasarkan data dalam konteks.
- Jika pengguna bertanya tentang data dirinya (NIM, nilai, jadwal, dll.), HANYA jawab jika datanya secara eksplisit disebutkan dalam konteks DAN pengguna sudah menyebutkan identitasnya sendiri.
- Jika konteks berisi data seseorang tetapi pengguna tidak menyebutkan namanya, JANGAN gunakan data tersebut untuk mengidentifikasi pengguna.
- Jika tidak yakin apakah konteks relevan dengan pertanyaan, katakan tidak memiliki informasi tersebut dan arahkan ke pihak kampus.
- Hanya tampilkan informasi yang benar-benar ada dalam konteks yang diberikan. 
- JANGAN PERNAH menambahkan kalimat penutup yang merujuk ke data yang tidak ditampilkan.
"""


def _get_api_key() -> str:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    return api_key


def llm_available() -> bool:
    api_key = _get_api_key()
    if not api_key:
        logger.warning("GROQ_API_KEY tidak ditemukan.")
        return False
    if not api_key.startswith("gsk_"):
        logger.warning("GROQ_API_KEY format tidak valid.")
        return False
    return True


def _is_conversational(query: str) -> bool:
    """Cek apakah query hanya respons conversational pendek (ya/oke/lanjut/dll.)"""
    return bool(CONVERSATIONAL_PATTERNS.match(query))


def _is_rag_relevant(query: str, rag_context: str) -> bool:
    """
    Cek apakah RAG context relevan dengan query.
    Strategi: context harus mengandung kata kunci kampus ATAU
    ada overlap signifikan antara kata kunci query dan context.
    """
    if not rag_context.strip():
        return False

    # Jika context mengandung kata kunci kampus → dianggap relevan
    if CAMPUS_KEYWORDS.search(rag_context):
        return True

    # Fallback: cek overlap kata (≥2 kata dari query muncul di context)
    query_words = set(re.findall(r"\b\w{4,}\b", query.lower()))
    context_words = set(re.findall(r"\b\w{4,}\b", rag_context.lower()))
    overlap = query_words & context_words
    if len(overlap) >= 2:
        return True

    logger.info(f"RAG context dianggap tidak relevan. Overlap kata: {overlap}")
    return False


def ask_llm(
    query: str,
    rag_context: str = "",
    chat_history: list[dict] | None = None,
    max_tokens: int = 600,
) -> tuple[str, bool]:
    api_key = _get_api_key()
    if not api_key:
        return _offline_fallback(query), False

    # ── Layer 1: Filter query conversational ─────────────────────────────────
    if _is_conversational(query):
        logger.info(f"Query '{query}' dikenali sebagai conversational. Tangani langsung.")
        return _conversational_reply(query, chat_history), False

    # ── Layer 2: RAG kosong → langsung fallback ───────────────────────────────
    if not rag_context.strip():
        logger.info("RAG context kosong. Hard fallback tanpa panggil LLM.")
        return _offline_fallback(query), False

    # ── Layer 3: RAG tidak relevan → fallback ────────────────────────────────
    if not _is_rag_relevant(query, rag_context):
        logger.info("RAG context tidak relevan dengan query. Fallback.")
        return _offline_fallback(query), False

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        # Susun user message
        user_content = (
            f"[KONTEKS DARI KNOWLEDGE BASE KAMPUS]\n{rag_context}\n\n"
            f"[PERTANYAAN PENGGUNA]\n{query}"
        )

        # Bangun messages dengan history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if chat_history:
            for turn in chat_history[-6:]:
                role    = turn.get("role", "user")
                content = turn.get("content", "")
                if content and role in ("user", "assistant"):
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,   # Diturunkan dari 0.5 → lebih faktual, kurang improvisasi
        )

        answer = response.choices[0].message.content or ""
        if not answer:
            raise ValueError("Respons Groq kosong.")

        logger.info(f"Groq OK. Tokens: {response.usage.prompt_tokens}/{response.usage.completion_tokens}")
        return answer, True

    except Exception as e:
        msg = str(e).lower()
        logger.error(f"Groq error ({type(e).__name__}): {e}")
        if "authentication" in msg or "api_key" in msg or "invalid" in msg:
            return _offline_fallback(query, error="autentikasi"), False
        if "rate" in msg or "quota" in msg or "429" in msg:
            return _offline_fallback(query, error="rate_limit"), False
        if "connect" in msg or "timeout" in msg:
            return _offline_fallback(query, error="koneksi"), False
        return _offline_fallback(query, error="umum"), False


def _conversational_reply(query: str, chat_history: list[dict] | None = None) -> str:
    """
    Tangani query conversational (ya/oke/lanjut) tanpa panggil LLM atau RAG.
    Coba inferensikan konteks dari history terakhir.
    """
    last_topic = None
    if chat_history:
        for turn in reversed(chat_history):
            if turn.get("role") == "assistant":
                last_topic = turn.get("content", "")[:120]
                break

    if last_topic:
        return (
            "Baik! Apakah ada hal spesifik yang ingin kamu ketahui lebih lanjut? "
            "Misalnya, kamu bisa tanyakan tentang **jurusan**, **biaya kuliah**, "
            "**beasiswa**, atau **fasilitas kampus** President University. 😊"
        )
    return (
        "Halo! Ada yang bisa saya bantu? Silakan ajukan pertanyaanmu seputar "
        "**pendaftaran**, **program studi**, **biaya kuliah**, atau informasi kampus lainnya. 😊"
    )


def _offline_fallback(query: str, error: str = "") -> str:
    error_messages = {
        "autentikasi": "_(Catatan: GROQ_API_KEY tidak valid.)_\n\n",
        "rate_limit":  "_(Catatan: Layanan Groq sedang sibuk, coba lagi sebentar.)_\n\n",
        "koneksi":     "_(Catatan: Tidak dapat terhubung ke Groq saat ini.)_\n\n",
        "umum":        "_(Catatan: Terjadi kendala teknis pada layanan Groq.)_\n\n",
    }
    prefix = error_messages.get(error, "")
    return (
        f"{prefix}"
        f"Maaf, saya belum memiliki informasi spesifik tentang **\"{query}\"**.\n\n"
        "Silakan hubungi pihak kampus:\n"
        "📞 (+62) 859 4236 3468 | ✉️ presuniv_pekanbaru@president.ac.id\n\n"
        "atau kamu bisa follow dan DM IG untuk lebih lanjut:\n"
        "📷 @presunivpekanbaru"
    )