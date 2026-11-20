"""Pre-processing: normalisation, markup removal, sentence split, length check (Sec. 3.3.2, 4.3.1)."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from dataclasses import dataclass, field

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)*")
_TAG_RE = re.compile(r"<[^>]+>")
_MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MD_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_EMPH_RE = re.compile(r"(\*\*|__|\*|`{1,3})")
_MD_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_SPACES_RE = re.compile(r"[ \t\f\v ]+")
_NEWLINES_RE = re.compile(r"\n{3,}")
_REGEX_SENT_RE = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")

# Phrases that reveal the source trivially (Step 2 of the methodology). Removed from the corpus.
BOILERPLATE_PATTERNS = [
    r"as an ai(?: language model)?[^.?!]*[.?!]",
    r"i(?:'m| am) (?:just )?an ai[^.?!]*[.?!]",
    r"i(?:'m| am) sorry,? but (?:i|as)[^.?!]*[.?!]",
    r"i hope (?:this|that) helps[^.?!]*[.?!]",
    r"(?:please )?(?:let me know|feel free to ask) if you have any (?:other|further|more) questions[^.?!]*[.?!]",
    r"i do not have (?:personal|access to)[^.?!]*[.?!]",
    r"(?:certainly|sure|of course)[!,.]\s*here(?:'s| is)[^.?!:]*[.?!:]",
]
_BOILER_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)


def normalise(text: str) -> str:
    """NFKC normalisation, HTML/Markdown stripping and whitespace collapsing."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = html.unescape(t)
    t = _TAG_RE.sub(" ", t)
    t = _MD_LINK_RE.sub(r"\1", t)
    t = _MD_HEADER_RE.sub("", t)
    t = _MD_BULLET_RE.sub("", t)
    t = _MD_EMPH_RE.sub("", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _SPACES_RE.sub(" ", t)
    t = "\n".join(line.strip() for line in t.split("\n"))
    t = _NEWLINES_RE.sub("\n\n", t)
    return t.strip()


def remove_boilerplate(text: str) -> str:
    """Remove assistant boilerplate (refusals, sign-offs) that would make detection trivial."""
    return _SPACES_RE.sub(" ", _BOILER_RE.sub(" ", text)).strip()


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def word_count(text: str) -> int:
    return len(words(text))


def text_hash(text: str) -> str:
    """Hash of aggressively normalised text, used for near-duplicate removal."""
    key = " ".join(w.lower() for w in words(text))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


@dataclass
class Sentence:
    start: int
    end: int
    text: str
    words: list[str]
    pos: list[str] = field(default_factory=list)


@dataclass
class Analysis:
    """Output of the pre-processor consumed by the feature extractor."""

    text: str
    sentences: list[Sentence]
    words: list[str]

    @property
    def n_words(self) -> int:
        return len(self.words)


class Preprocessor:
    """Cleans text, splits sentences (spaCy) and checks minimum length."""

    def __init__(self, min_words: int = 100, spacy_model: str = "en_core_web_sm", use_spacy: bool = True):
        self.min_words = min_words
        self.spacy_model = spacy_model
        self.use_spacy = use_spacy
        self._nlp = None
        self._spacy_failed = False

    # -- public API (class diagram, Fig. 5.4) ------------------------------------------
    def clean(self, text: str) -> str:
        return normalise(text)

    def is_sufficient(self, text: str) -> bool:
        return word_count(text) >= self.min_words

    def sentences(self, text: str) -> list[str]:
        return [s.text for s in self.analyze(text).sentences]

    def analyze(self, text: str) -> Analysis:
        nlp = self._get_nlp()
        if nlp is None:
            return self._regex_analyze(text)
        doc = nlp(text)
        sents: list[Sentence] = []
        for s in doc.sents:
            toks = [t for t in s if not t.is_space]
            if not toks:
                continue
            ws = [t.text for t in toks if WORD_RE.fullmatch(t.text) or t.is_alpha or t.like_num]
            sents.append(Sentence(s.start_char, s.end_char, s.text, ws, [t.pos_ for t in toks]))
        return Analysis(text, sents, words(text))

    # -- internals -------------------------------------------------------------------
    def _get_nlp(self):
        if not self.use_spacy or self._spacy_failed:
            return None
        if self._nlp is None:
            try:
                import spacy

                nlp = spacy.load(self.spacy_model, exclude=["ner", "lemmatizer", "parser"])
                if "senter" in nlp.pipe_names or "senter" in nlp.disabled:
                    nlp.enable_pipe("senter")
                else:  # pragma: no cover
                    nlp.add_pipe("sentencizer")
                nlp.max_length = 2_000_000
                self._nlp = nlp
            except (ImportError, OSError):
                self._spacy_failed = True
                return None
        return self._nlp

    @staticmethod
    def _regex_analyze(text: str) -> Analysis:
        sents: list[Sentence] = []
        pos = 0
        for block in re.split(r"\n\s*\n", text):
            for piece in _REGEX_SENT_RE.split(block):
                piece = piece.strip()
                if not piece:
                    continue
                start = text.find(piece, pos)
                start = max(start, 0)
                end = start + len(piece)
                pos = end
                sents.append(Sentence(start, end, piece, words(piece)))
        return Analysis(text, sents, words(text))
