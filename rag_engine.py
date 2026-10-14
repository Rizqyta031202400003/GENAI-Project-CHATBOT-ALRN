import os
import re
import pickle
import logging
import numpy as np
from difflib import get_close_matches

logger = logging.getLogger("rag_engine")

# ── scikit-learn ──────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    logger.warning("scikit-learn tidak terinstall.")

# ── FAISS ─────────────────────────────────────────────────────
try:
    import faiss
    FAISS_OK = True
except ImportError:
    FAISS_OK = False
    logger.warning("faiss-cpu tidak terinstall.")

# ── LangChain ─────────────────────────────────────────────────
try:
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_OK = True
except ImportError:
    LANGCHAIN_OK = False
    logger.warning("langchain tidak terinstall. Install: pip install langchain langchain-community")

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TF  = os.path.join(CACHE_DIR, "tfidf.pkl")

# ── [PERUBAHAN 2] Threshold dinaikkan dari 0.05/0.08 → 0.20 ──
SCORE_THRESHOLD = 0.20


# ════════════════════════════════════════════════════════════════
# [PERUBAHAN 1] QUERY REFORMULATION DARI HISTORY
# ════════════════════════════════════════════════════════════════

_AMBIGUOUS = re.compile(
    r"^\s*(ya|iya|yaa+|oke|ok+|lanjut|lanjutkan|boleh|baik|tentu|"
    r"silakan|silahkan|benar|betul|sip|siap|noted|paham|ngerti|"
    r"mengerti|hmm+|oh+|yep|yups?)\s*[!?.]*\s*$",
    re.IGNORECASE,
)


def _reformulate_from_history(query: str, chat_history: list[dict] | None) -> str:
    """
    Kalau query ambigu (ya/oke/lanjut/dll), cari query USER terakhir
    yang tidak ambigu dari history dan gunakan itu untuk RAG retrieval.

    Kenapa pakai query user (bukan pesan bot)?
    → Query user mengandung kata kunci topik yang relevan untuk TF-IDF/FAISS.
    → Pesan bot terlalu panjang dan tidak cocok sebagai search query.
    """
    if not _AMBIGUOUS.match(query.strip()):
        return query  # Query sudah jelas, tidak perlu diubah

    if not chat_history:
        return query

    # Cari query user terakhir yang tidak ambigu
    for turn in reversed(chat_history):
        if turn.get("role") == "user":
            last_user_query = turn.get("content", "").strip()
            if last_user_query and not _AMBIGUOUS.match(last_user_query):
                logger.info(f"Query direformulasi: '{query}' → '{last_user_query}'")
                return last_user_query

    return query


# ════════════════════════════════════════════════════════════════
# LANGCHAIN DOCUMENT LOADERS
# ════════════════════════════════════════════════════════════════

def load_csv_documents() -> list[dict]:
    """
    Load data.csv menggunakan pandas + konversi ke LangChain Document,
    lalu di-split dengan RecursiveCharacterTextSplitter.
    """
    docs = []
    path = os.path.join(DATA_DIR, "data.csv")
    if not os.path.exists(path):
        logger.warning(f"data.csv tidak ditemukan: {path}")
        return docs

    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas tidak terinstall.")
        return docs

    # Text splitter dari LangChain
    splitter = None
    if LANGCHAIN_OK:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "]
        )

    try:
        df = pd.read_csv(path, dtype=str).fillna("")
        df = df[df["Jenis"] != "Jenis"].reset_index(drop=True)

        for _, row in df.iterrows():
            try:
                jenis = row.get("Jenis", "").strip()
                col1  = row.get("Program Studi", "").strip()
                col2  = row.get("Jenjang", "").strip()
                col3  = row.get("Akreditasi", "").strip()
                col4  = row.get("Deskripsi", "").strip()
                col5  = row.get("Fakultas", "").strip()
                col6  = row.get("Gelar", "").strip()
                col7  = row.get("Lama Studi", "").strip()

                if jenis == "Jurusan":
                    text   = f"jurusan program studi {col1} {col2} akreditasi {col3} {col4} {col5} gelar {col6} lama studi {col7}"
                    answer = (f"Program Studi: {col1} ({col2})\n"
                              f"Fakultas: {col5}\n"
                              f"Akreditasi: {col3}\n"
                              f"Gelar: {col6}\n"
                              f"Lama Studi: {col7}\n"
                              f"Deskripsi: {col4}")
                    docs.append({"text": text, "answer": answer, "category": "jurusan", "source": "csv"})

                elif jenis == "Pendaftaran":
                    text   = f"pendaftaran jalur {col1} syarat {col2} biaya {col3} cara daftar {col4} jadwal {col5}"
                    answer = (f"Jalur Pendaftaran: {col1}\n"
                              f"Syarat: {col2}\n"
                              f"Biaya: {col3}\n"
                              f"Cara Daftar: {col4}\n"
                              f"Jadwal: {col5}")
                    docs.append({"text": text, "answer": answer, "category": "pendaftaran", "source": "csv"})

                elif jenis == "Dosen":
                    text   = f"dosen {col1} {col2} {col3} {col5}"
                    answer = (f"Nama: {col1}\n"
                              f"Jurusan: {col2}\n"
                              f"Jabatan: {col3}\n"
                              f"Email: {col4}\n"
                              f"Mata Kuliah: {col5}")
                    docs.append({"text": text, "answer": answer, "category": "dosen", "source": "csv"})

                elif jenis == "Fasilitas":
                    text   = f"fasilitas {col1} {col2} {col3} {col4}"
                    answer = (f"Fasilitas: {col1}\n"
                              f"Lokasi: {col2}\n"
                              f"Keterangan: {col3}\n"
                              f"Jam Operasional: {col4}")
                    docs.append({"text": text, "answer": answer, "category": "fasilitas", "source": "csv"})

                elif jenis == "Program Perkuliahan":
                    text   = f"program perkuliahan {col1} {col2} {col3} {col4} {col5}"
                    answer = (f"Program: {col1}\n"
                              f"Target: {col2}\n"
                              f"Fleksibilitas: {col3}\n"
                              f"Jenis Kelas: {col4}\n"
                              f"Waktu Kuliah: {col5}")
                    docs.append({"text": text, "answer": answer, "category": "program", "source": "csv"})

                elif jenis == "Kalender Akademik":
                    text   = f"kalender akademik jadwal {col1} {col2} {col3} {col4}"
                    answer = (f"Kegiatan: {col1}\n"
                              f"Tanggal Mulai: {col2}\n"
                              f"Tanggal Selesai: {col3}\n"
                              f"Keterangan: {col4}")
                    docs.append({"text": text, "answer": answer, "category": "kalender", "source": "csv"})

                elif jenis == "Benefit":
                    text   = f"benefit keuntungan {col1} {col2}"
                    answer = f"Benefit: {col1}\nKeterangan: {col2}"
                    docs.append({"text": text, "answer": answer, "category": "benefit", "source": "csv"})

                # ── LangChain: split answer jika terlalu panjang ──
                if docs and splitter and LANGCHAIN_OK:
                    last = docs[-1]
                    if len(last["answer"]) > 300:
                        lc_doc   = Document(page_content=last["answer"],
                                            metadata={"category": last["category"], "source": "csv"})
                        chunks   = splitter.split_documents([lc_doc])
                        for chunk in chunks[1:]:
                            docs.append({
                                "text":     last["text"],
                                "answer":   chunk.page_content,
                                "category": last["category"],
                                "source":   "csv"
                            })

            except Exception as row_err:
                logger.warning(f"Baris CSV dilewati: {row_err}")
                continue

        logger.info(f"CSV loaded: {len(docs)} dokumen.")

    except Exception as e:
        logger.error(f"CSV Loader error: {type(e).__name__}: {e}")

    return docs


def load_pdf_documents() -> list[dict]:
    """
    Load data.pdf menggunakan LangChain PyPDFLoader jika tersedia,
    fallback ke pypdf langsung. Hasil di-split dengan
    RecursiveCharacterTextSplitter dari LangChain.
    """
    docs = []
    path = os.path.join(DATA_DIR, "data.pdf")
    if not os.path.exists(path):
        logger.warning(f"data.pdf tidak ditemukan: {path}")
        return docs

    full_text = ""
    n_pages   = 0

    if LANGCHAIN_OK:
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader   = PyPDFLoader(path)
            lc_pages = loader.load()
            full_text = "\n".join(p.page_content for p in lc_pages if p.page_content)
            n_pages   = len(lc_pages)
            logger.info(f"PDF dibaca via LangChain PyPDFLoader: {n_pages} halaman.")
        except Exception as e:
            logger.warning(f"LangChain PyPDFLoader gagal: {e}, fallback ke pypdf...")

    if not full_text.strip():
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            if reader.is_encrypted:
                logger.warning("data.pdf terenkripsi.")
                return docs
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    full_text += extracted + "\n"
            n_pages = len(reader.pages)
            logger.info(f"PDF dibaca via pypdf: {n_pages} halaman.")
        except Exception as e:
            logger.error(f"pypdf error: {e}")
            return docs

    if not full_text.strip():
        logger.warning("data.pdf tidak mengandung teks (mungkin image-based).")
        return docs

    if LANGCHAIN_OK:
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=80,
                separators=["\n\n", "\n", ".", " "]
            )
            lc_doc = Document(
                page_content=full_text,
                metadata={"source": "pdf", "category": "kalender_akademik_pdf"}
            )
            chunks = splitter.split_documents([lc_doc])
            for chunk in chunks:
                para = chunk.page_content.strip()
                if len(para) >= 20:
                    docs.append({
                        "text":     para.lower(),
                        "answer":   para,
                        "category": "kalender_akademik_pdf",
                        "source":   "pdf"
                    })
            logger.info(f"PDF loaded: {len(docs)} chunks via LangChain TextSplitter.")
            return docs
        except Exception as e:
            logger.warning(f"LangChain splitter gagal: {e}, fallback manual...")

    count = 0
    for para in re.split(r'\n{2,}', full_text):
        para = para.strip()
        if len(para) < 20:
            continue
        docs.append({
            "text":     para.lower(),
            "answer":   para,
            "category": "kalender_akademik_pdf",
            "source":   "pdf"
        })
        count += 1
    logger.info(f"PDF loaded: {count} chunks (manual split).")
    return docs


# ════════════════════════════════════════════════════════════════
# TYPO CORRECTION
# ════════════════════════════════════════════════════════════════

VOCAB = [
    "jurusan", "program", "studi", "teknik", "informatika", "sistem", "informasi",
    "fakultas", "akreditasi", "gelar", "sarjana", "unggul",
    "pendaftaran", "daftar", "mendaftar", "jalur", "mandiri", "beasiswa", "prestasi",
    "rapor", "utbk", "syarat", "ijazah", "biaya", "gratis", "seleksi", "wawancara",
    "fasilitas", "laboratorium", "lab", "komputer", "perpustakaan", "asrama",
    "kantin", "internet", "wifi", "lapangan", "olahraga", "diskusi", "ruang", "kelas",
    "dosen", "pengajar", "guru", "profesor", "jabatan", "email", "mata", "kuliah",
    "kalender", "akademik", "jadwal", "tanggal", "uts", "uas", "ujian", "semester",
    "liburan", "gelombang", "mulai", "selesai", "kapan", "deadline",
    "benefit", "keuntungan", "laptop", "magang", "english", "club", "kursus",
    "apa", "siapa", "bagaimana", "cara", "ada", "tersedia", "berapa", "dimana",
]


def correct_typos(text: str) -> str:
    try:
        words = text.lower().split()
        corrected = []
        for word in words:
            if len(word) <= 2:
                corrected.append(word)
                continue
            matches = get_close_matches(word, VOCAB, n=1, cutoff=0.75)
            corrected.append(matches[0] if matches and matches[0] != word else word)
        return " ".join(corrected)
    except Exception as e:
        logger.warning(f"correct_typos gagal: {e}")
        return text


# ════════════════════════════════════════════════════════════════
# INTENT DETECTION
# ════════════════════════════════════════════════════════════════

INTENT_MAP = {
    "dosen":       ["dosen", "pengajar", "siapa yang mengajar", "guru", "profesor", "staf pengajar",
                    "siapa saja dosen", "list dosen", "daftar dosen"],
    "jurusan":     ["jurusan apa", "program studi", "ada jurusan", "jurusan yang",
                    "teknik informatika", "sistem informasi", "apa itu jurusan", "jurusan mana"],
    "pendaftaran": ["cara daftar", "mendaftar", "jalur", "syarat masuk", "biaya daftar",
                    "pendaftaran", "pmb", "beasiswa", "utbk", "rapor"],
    "fasilitas":   ["fasilitas", "lab ", "laboratorium", "perpustakaan", "asrama",
                    "kantin", "wifi", "internet", "lapangan", "ruang kelas"],
    "kalender":    ["jadwal", "kapan", "uts", "uas", "ujian", "liburan", "semester",
                    "tanggal", "gelombang", "krs", "deadline"],
    "program":     ["kuliah malam", "kelas malam", "non reguler", "karyawan",
                    "reguler", "program kuliah", "waktu kuliah"],
    "benefit":     ["benefit", "keuntungan", "fasilitas gratis", "laptop",
                    "magang", "english club", "gratis apa"],
}


def detect_intent(query: str) -> str | None:
    try:
        q = query.lower()
        for category, keywords in INTENT_MAP.items():
            if any(kw in q for kw in keywords):
                return category
    except Exception as e:
        logger.warning(f"detect_intent gagal: {e}")
    return None


# ════════════════════════════════════════════════════════════════
# RAG ENGINE
# ════════════════════════════════════════════════════════════════

class RAGEngine:
    def __init__(self):
        self.docs       = []
        self.vectorizer = None
        self.tfidf_mat  = None
        self.faiss_idx  = None
        self._ready     = False
        self._build()

    def _build(self):
        try:
            self.docs = load_csv_documents() + load_pdf_documents()
        except Exception as e:
            logger.error(f"Gagal memuat dokumen: {e}")
            self.docs = []

        if not self.docs:
            logger.error("Tidak ada dokumen. Cek folder data/.")
            return

        texts = [d["text"] for d in self.docs]

        if self._try_load_cache(texts):
            self._ready = True
            return

        if not SKLEARN_OK:
            logger.error("scikit-learn diperlukan.")
            return

        try:
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4),
                max_features=12000, sublinear_tf=True
            )
            self.tfidf_mat = self.vectorizer.fit_transform(texts)
            self._save_cache(texts)
            self._build_faiss()
            self._ready = True
            logger.info(f"RAG Engine siap: {len(self.docs)} dokumen.")
        except Exception as e:
            logger.error(f"Gagal membangun TF-IDF: {e}")

    def _try_load_cache(self, texts: list) -> bool:
        if not os.path.exists(CACHE_TF):
            return False
        try:
            with open(CACHE_TF, "rb") as f:
                saved = pickle.load(f)
            if saved.get("n_docs") != len(texts):
                logger.info("Cache tidak sinkron, rebuild...")
                return False
            self.vectorizer = saved["vectorizer"]
            self.tfidf_mat  = saved["matrix"]
            self._build_faiss()
            logger.info("Cache TF-IDF dimuat.")
            return True
        except Exception as e:
            logger.warning(f"Cache rusak, rebuild: {e}")
            try:
                os.remove(CACHE_TF)
            except OSError:
                pass
            return False

    def _save_cache(self, texts: list):
        try:
            with open(CACHE_TF, "wb") as f:
                pickle.dump({
                    "vectorizer": self.vectorizer,
                    "matrix":     self.tfidf_mat,
                    "n_docs":     len(texts)
                }, f)
        except Exception as e:
            logger.warning(f"Gagal simpan cache: {e}")

    def _build_faiss(self):
        if not FAISS_OK or self.tfidf_mat is None:
            return
        try:
            mat   = self.tfidf_mat.toarray().astype(np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1
            mat   = mat / norms
            idx   = faiss.IndexFlatIP(mat.shape[1])
            self.faiss_idx = faiss.IndexIDMap(idx)
            self.faiss_idx.add_with_ids(mat, np.arange(len(self.docs), dtype=np.int64))
            logger.info("FAISS index siap.")
        except Exception as e:
            logger.warning(f"FAISS gagal: {e}")
            self.faiss_idx = None

    # ── Retrieval ─────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[float, dict]]:
        if not self.docs:
            return []
        try:
            corrected        = correct_typos(query)
            intent           = detect_intent(corrected) or detect_intent(query)
            query_for_search = corrected if corrected != query.lower() else query.lower()
            results          = []

            # FAISS
            if self.faiss_idx is not None and self.vectorizer is not None:
                try:
                    q    = self.vectorizer.transform([query_for_search]).toarray().astype(np.float32)
                    norm = np.linalg.norm(q)
                    if norm > 0:
                        q /= norm
                    scores, indices = self.faiss_idx.search(q, top_k * 2)
                    results = [
                        (float(s), self.docs[i])
                        for s, i in zip(scores[0], indices[0])
                        if i >= 0 and s > SCORE_THRESHOLD  # [PERUBAHAN 2]
                    ]
                except Exception as e:
                    logger.warning(f"FAISS search gagal: {e}")

            # TF-IDF fallback
            if not results and self.vectorizer is not None and self.tfidf_mat is not None:
                try:
                    q    = self.vectorizer.transform([query_for_search])
                    sims = cosine_similarity(q, self.tfidf_mat)[0]
                    top  = np.argsort(sims)[::-1][:top_k * 2]
                    results = [(float(sims[i]), self.docs[i]) for i in top if sims[i] > SCORE_THRESHOLD]  # [PERUBAHAN 2]
                except Exception as e:
                    logger.warning(f"TF-IDF search gagal: {e}")

            # Keyword fallback
            if not results:
                results = self._keyword_search(query, top_k * 2)

            # Intent boosting
            if intent:
                results = [
                    (s * 1.5 if d.get("category") == intent else s, d)
                    for s, d in results
                ]
                results.sort(key=lambda x: -x[0])

            return results[:top_k]

        except Exception as e:
            logger.error(f"retrieve() error: {e}")
            return []

    def _keyword_search(self, query: str, top_k: int) -> list[tuple[float, dict]]:
        try:
            qw = set(query.lower().split())
            if not qw:
                return []
            scored = [
                (len(qw & set(d["text"].lower().split())) / max(len(qw), 1), d)
                for d in self.docs
                if qw & set(d["text"].lower().split())
            ]
            return sorted(scored, key=lambda x: -x[0])[:top_k]
        except Exception as e:
            logger.error(f"_keyword_search error: {e}")
            return []

    # ── Answer ────────────────────────────────────────────────

    def answer(self, query: str, chat_history: list[dict] | None = None) -> tuple[str, list, bool, bool]:
        from llm_engine import ask_llm, llm_available

        q_low = query.lower().strip()
        if not q_low:
            return "Silakan ketik pertanyaanmu 😊", [], True, False

        # ── [PERUBAHAN 1] Reformulasi query ambigu dari history ──────────────
        query = _reformulate_from_history(query, chat_history)
        q_low = query.lower().strip()

        # ── Normalisasi alias jurusan ────────────────────────────────
        ALIAS_MAP = {
            r'\bit\b':                      'teknik informatika',
            r'\binformation technology\b':  'teknik informatika',
            r'\btekin\b':                   'teknik informatika',
            r'\bteknik info\b':             'teknik informatika',
            r'\bis\b':                      'sistem informasi',
            r'\binformation system\b':      'sistem informasi',
            r'\binfo system\b':             'sistem informasi',
            r'\bsistem info\b':             'sistem informasi',
        }
        for pattern, replacement in ALIAS_MAP.items():
            q_low = re.sub(pattern, replacement, q_low)
        q_low = re.sub(r'\bteknik\s+teknik\s+informatika\b', 'teknik informatika', q_low)
        q_low = re.sub(r'\bsistem\s+sistem\s+informasi\b', 'sistem informasi', q_low)
        query = q_low

        # Greetings
        if any(w in q_low for w in ["halo", "hai", "hi ", "hello", "selamat pagi",
                                     "selamat siang", "selamat malam", "hei"]):
            return (
                "Halo! Selamat datang di **ALRN Chatbot** 👋\n\n"
                "Saya siap membantu pertanyaan seputar kampus. "
                "Tanyakan tentang **jurusan**, **pendaftaran**, **biaya kuliah**, "
                "**fasilitas**, **dosen**, atau **kalender akademik**.",
                [], True, False
            )
        if any(w in q_low for w in ["terima kasih", "makasih", "thanks", "thx"]):
            return "Sama-sama! Senang bisa membantu. Ada pertanyaan lain? 😊", [], True, False
        if any(w in q_low for w in ["siapa kamu", "kamu siapa", "apa itu alrn"]):
            return (
                "Saya adalah **ALRN Chatbot** 💠 — asisten virtual kampus berbasis "
                "RAG (LangChain + FAISS + TF-IDF) yang siap membantu pertanyaan "
                "seputar akademik, pendaftaran, fasilitas, dosen, dan kalender kampus.",
                [], True, False
            )
        
        # RAG Retrieval — top_k lebih besar untuk dosen & fasilitas
        try:
            intent = detect_intent(query)
            top_k  = 15 if intent in ("dosen", "fasilitas") else 5
            results = self.retrieve(query, top_k=top_k)

            # Kalau query dosen + ada nama jurusan → filter langsung dari semua docs
            if intent == "dosen":
                jurusan_filter = None
                if "teknik informatika" in query:
                    jurusan_filter = "teknik informatika"
                elif "sistem informasi" in query:
                    jurusan_filter = "sistem informasi"

                if jurusan_filter:
                    filtered = [
                        (1.0, d) for d in self.docs
                        if d.get("category") == "dosen"
                        and jurusan_filter in d.get("text", "").lower()
                    ]
                    if filtered:
                        results = filtered
        except Exception as e:
            logger.error(f"Retrieve error: {e}")
            results = []

        # Ada hasil RAG → kirim ke LLM
        if results and results[0][0] >= SCORE_THRESHOLD:  # [PERUBAHAN 2]
            ctx = "\n\n".join(d["answer"] for _, d in results if _ > SCORE_THRESHOLD)  # [PERUBAHAN 2]

            if llm_available():
                llm_answer, llm_ok = ask_llm(query, rag_context=ctx, chat_history=chat_history)
                if llm_ok:
                    return llm_answer, [d for _, d in results], True, True

            # LLM tidak tersedia → raw
            combined = "\n\n".join(d["answer"] for _, d in results if _ > SCORE_THRESHOLD)  # [PERUBAHAN 2]
            return combined, [d for _, d in results], True, False

        # RAG kosong → LLM tanpa konteks
        if llm_available():
            # [PERUBAHAN 3] partial_ctx hanya dikirim jika score >= SCORE_THRESHOLD
            # Kalau tidak ada yang lolos, partial_ctx = "" → llm_engine guard
            # akan tangkap dan kembalikan _offline_fallback, tidak ngarang
            partial_ctx = "\n\n".join(
                d["answer"] for _, d in results[:2]
                if _ >= SCORE_THRESHOLD  # [PERUBAHAN 3]
            ) if results else ""

            llm_answer, llm_ok = ask_llm(query, rag_context=partial_ctx, chat_history=chat_history)
            if llm_ok:
                note = "_(Dijawab dengan bantuan AI — informasi bersifat umum)_\n\n"
                return note + llm_answer, [], True, True

        # Semua gagal
        logger.warning(f"Tidak ada jawaban untuk: '{query}'")
        return self._fallback(query), [], False, False

    def _fallback(self, query: str) -> str:
        return (
            f"Maaf, saya belum memiliki informasi untuk pertanyaan **\"{query}\"**.\n\n"
            "Silakan hubungi pihak kampus:\n"
            "📞 (+62) 859 4236 3468 | ✉️ presuniv_pekanbaru@president.ac.id\n"
            "atau follow dan DM Instagram:\n"
            "📷 @presunivpekanbaru"
        )

    @property
    def is_ready(self) -> bool:
        return self._ready


# ── Singleton ──────────────────────────────────────────────────
_engine: RAGEngine | None = None


def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine