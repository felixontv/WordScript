"""Groq Whisper transcription and LLM post-correction."""

import logging
import re
import threading
import time

import httpx
from groq import Groq

from .config import Config


def _make_groq_client(api_key: str, timeout: float) -> Groq:
    """Create a Groq client with forced IPv4 to avoid IPv6 connect timeouts."""
    return Groq(
        api_key=api_key,
        max_retries=0,
        timeout=timeout,
        http_client=httpx.Client(
            transport=httpx.HTTPTransport(local_address="0.0.0.0"),
            limits=httpx.Limits(
                max_connections=5,
                max_keepalive_connections=2,
                keepalive_expiry=30,
            ),
        ),
    )


class TranscriptionService:
    """Sends audio to the Groq Whisper API and returns transcribed text."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("TranscriptionService")
        self._client = (
            _make_groq_client(config.groq_api_key, timeout=55.0)
            if config.groq_api_key else None
        )
        self._correction_client = (
            _make_groq_client(config.groq_api_key, timeout=8.0)
            if config.groq_api_key else None
        )
        if not config.groq_api_key:
            self.logger.warning("No Groq API key set — open Settings (chevron) to enter yours.")

    def reload_api_key(self) -> None:
        """Re-initialise the Groq client after an API-key change at runtime."""
        # Close old httpx clients to prevent file-descriptor / connection leaks
        self._close_clients()
        if self.config.groq_api_key:
            self._client = _make_groq_client(self.config.groq_api_key, timeout=55.0)
            self._correction_client = _make_groq_client(self.config.groq_api_key, timeout=8.0)
            self.logger.info("Groq client reloaded with new API key.")
        else:
            self._client = None
            self._correction_client = None

    def _close_clients(self) -> None:
        """Close existing httpx clients to free connections."""
        for client in (self._client, self._correction_client):
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    def transcribe(self, wav_bytes: bytes) -> str:
        """Send WAV audio bytes to Groq and return the transcription text.

        Uses a daemon threading.Thread with join(timeout=60s) as a wall-clock
        guard against servers that stall after 100-Continue.
        """
        if not wav_bytes:
            return ""
        if self.config.backend != "groq":
            self.logger.error(
                "Backend '%s' is not yet implemented. Only 'groq' is supported.",
                self.config.backend,
            )
            return f"[Backend '{self.config.backend}' not supported yet]"
        if not self._client:
            self.logger.error(
                "NO API KEY SET. Open Settings ('>' button on the overlay) "
                "and enter your Groq API key from console.groq.com"
            )
            return "[NO_API_KEY]"

        audio_size_kb = len(wav_bytes) / 1024
        self.logger.info(
            "Sending %.1f KB to Groq Whisper (model: %s)...",
            audio_size_kb,
            self.config.model,
        )
        start = time.perf_counter()

        result_holder: list = [None]
        exc_holder: list = [None]

        def _worker():
            try:
                result_holder[0] = self._do_whisper_request(wav_bytes, start)
            except Exception as e:
                exc_holder[0] = e

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=60.0)

        if t.is_alive():
            elapsed = time.perf_counter() - start
            self.logger.error(
                "⚠ Groq Whisper timed out after %.0fs — likely rate-limited. "
                "Check quota at console.groq.com/usage",
                elapsed,
            )
            return "[TIMEOUT]"

        if exc_holder[0] is not None:
            raise exc_holder[0]
        return result_holder[0] if result_holder[0] is not None else ""

    def _do_whisper_request(self, wav_bytes: bytes, start: float) -> str:
        """Perform the actual Groq Whisper API call (runs inside worker thread)."""
        try:
            params = {
                "file": ("recording.wav", wav_bytes),
                "model": self.config.model,
                "response_format": "text",
                "temperature": 0.0,
            }
            if self.config.language:
                params["language"] = self.config.language
            if self.config.prompt:
                params["prompt"] = self.config.prompt

            transcription = self._client.audio.transcriptions.create(**params)
            elapsed = time.perf_counter() - start
            text = (
                transcription.strip()
                if isinstance(transcription, str)
                else transcription.text.strip()
            )

            if self._is_hallucination(text):
                self.logger.info("Filtered likely hallucination: '%s'", text)
                return ""

            if elapsed > 8.0:
                self.logger.warning(
                    "⚠ Groq API slow (%.0fs) — possible rate limiting. "
                    "Check console.groq.com/usage",
                    elapsed,
                )
            self.logger.info("✓ Whisper (%.2fs): %s", elapsed, text[:100])
            return text

        except Exception as exc:
            error_str = str(exc)
            if "429" in error_str or "rate_limit" in error_str.lower():
                self.logger.error("⚠ RATE LIMIT EXCEEDED: %s", exc)
                return "[Rate limit erreicht - bitte warten]"
            else:
                self.logger.error("❌ Groq API error: %s", exc)
                return f"[Transcription error: {exc}]"

    # ── Hallucination detection ───────────────────────────────────────────────

    _HALLUCINATION_EXACT: frozenset = frozenset({
        "", ".", "..", "...",
        "thanks for watching", "thank you for watching",
        "thank you", "thanks",
        "vielen dank", "vielen dank fürs zuschauen",
        "vielen dank für ihre aufmerksamkeit",
        "danke schön", "danke fürs zuschauen", "danke",
        "bitte abonnieren", "nicht vergessen zu abonnieren",
        "untertitel von", "untertitel der amara.org-community",
        "merci d'avoir regardé", "merci pour votre attention",
        "gracias por ver", "gracias", "subtítulos",
    })

    _HALLUCINATION_PATTERNS: list = [
        re.compile(r"^\s*(thanks? for (watching|listening)|thank you for (watching|listening))\s*[.!]?\s*$", re.I),
        re.compile(r"^\s*(subscribe|like and subscribe|don't forget to subscribe)\s*[.!]?\s*$", re.I),
        re.compile(r"^\s*(untertitel|subtitles?|subtítulos?|sous-titres?)\b", re.I),
        re.compile(r"^\s*(vielen dank|danke schön|danke)\s*[.!]?\s*$", re.I),
        re.compile(r"^\s*(bye|goodbye|tschüss|auf wiedersehen)\s*[.!]?\s*$", re.I),
        re.compile(r"^[.…,!?;:\-–—\s]+$"),
        re.compile(r"^\s*\[?(musik|music|applause|laughter|♪|♫)\]?\s*$", re.I),
    ]

    def _is_hallucination(self, text: str) -> bool:
        low = text.strip().lower()
        if low in self._HALLUCINATION_EXACT:
            return True
        for pat in self._HALLUCINATION_PATTERNS:
            if pat.search(low):
                return True
        return False

    # ── Post-correction ───────────────────────────────────────────────────────

    def correct(
        self,
        text: str,
        model: str = None,
        timeout: float = 8.0,
        filter_fillers: bool = None,
        professionalize: bool = None,
    ) -> str:
        """Run LLM post-correction with a wall-clock timeout."""
        if not text:
            return text

        _filter = filter_fillers  if filter_fillers  is not None else self.config.filter_fillers
        _prof   = professionalize if professionalize is not None else self.config.professionalize
        _model  = model or self.config.correction_model
        result_holder: list = [None]

        def _worker():
            result_holder[0] = self._correct_with_llm(
                text, model=_model, filter_fillers=_filter, professionalize=_prof
            )

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            self.logger.warning("LLM correction timed out (%.0fs), using original text", timeout)
            return text

        return result_holder[0] if result_holder[0] is not None else text

    @staticmethod
    def _word_overlap_ok(original: str, corrected: str, threshold: float = 0.55) -> bool:
        """Return True if corrected shares enough words with original."""
        orig_words = set(original.lower().split())
        if len(orig_words) < 5:
            return True
        corr_words = set(corrected.lower().split())
        overlap = len(orig_words & corr_words) / len(orig_words)
        return overlap >= threshold

    def _correct_with_llm(
        self,
        text: str,
        model: str = None,
        filter_fillers: bool = False,
        professionalize: bool = False,
    ) -> str:
        """Use Groq LLM to correct transcription errors."""
        try:
            start = time.perf_counter()

            if professionalize and filter_fillers:
                system_prompt = (
                    "Du bist ein stummer Textverarbeitungs-Filter. "
                    "Gib AUSSCHLIESSLICH den verarbeiteten Text zurück — "
                    "KEINE Kommentare, Erklärungen oder Antworten. "
                    "Sprache beibehalten (DE/EN/gemischt), niemals übersetzen. "
                    "Aufgaben: "
                    "(1) Füllwörter entfernen: ähm, äh, öh, ähh, hmm, uh, um, er, mhm; "
                    "(2) Tippfehler und Grammatik korrigieren; "
                    "(3) Text professionell und klar formulieren — "
                    "Satzstruktur verbessern, Redundanzen entfernen, sachlich und präzise. "
                    "Du bist ein Filter, kein Assistent."
                )
            elif professionalize:
                system_prompt = (
                    "Du bist ein stummer Textverarbeitungs-Filter. "
                    "Gib AUSSCHLIESSLICH den verarbeiteten Text zurück — "
                    "KEINE Kommentare, Erklärungen oder Antworten. "
                    "Sprache beibehalten (DE/EN/gemischt), niemals übersetzen. "
                    "Aufgaben: "
                    "(1) Tippfehler und Grammatik korrigieren; "
                    "(2) Text professionell und klar formulieren — "
                    "Satzstruktur verbessern, Redundanzen entfernen, sachlich und präzise. "
                    "Bedeutung erhalten, keine neuen Informationen hinzufügen. "
                    "Du bist ein Filter, kein Assistent."
                )
            elif filter_fillers:
                system_prompt = (
                    "Du bist ein stummer Textkorrektur-Filter. "
                    "Gib AUSSCHLIESSLICH den korrigierten Text zurück — "
                    "KEINE Kommentare, Erklärungen oder Antworten. "
                    "Sprache beibehalten (DE/EN/gemischt), niemals übersetzen. "
                    "Aufgaben: "
                    "(1) Füllwörter entfernen: ähm, äh, öh, ähh, hmm, uh, um, er, mhm; "
                    "(2) Tippfehler und Grammatik korrigieren. "
                    "Sonst nichts verändern. Bedeutung und Stil beibehalten. "
                    "Du bist ein Filter, kein Assistent."
                )
            else:
                system_prompt = (
                    "Du bist ein stummer Textkorrektur-Filter. "
                    "Gib AUSSCHLIESSLICH den korrigierten Text zurück — "
                    "KEINE Kommentare, Erklärungen oder Antworten. "
                    "Sprache beibehalten (DE/EN/gemischt), niemals übersetzen. "
                    "Nur Tippfehler und Grammatik korrigieren; "
                    "niemals Wörter entfernen, kürzen oder umformulieren. "
                    "Kurzer Input (1-5 Wörter): exakt zurückgeben. "
                    "Bei korrektem Text: Originaltext Zeichen für Zeichen zurück. "
                    "Du bist ein Filter, kein Assistent."
                )

            response = self._correction_client.chat.completions.create(
                model=model or self.config.correction_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=max(len(text) * 2, 40),
            )

            corrected = response.choices[0].message.content.strip()
            elapsed = time.perf_counter() - start

            # Layer 1: Length guard (professionalize/filter_fillers can shrink text)
            min_ratio = 0.4 if professionalize else (0.5 if filter_fillers else 0.85)
            if len(corrected) > len(text) * 1.5 + 50:
                self.logger.warning(
                    "LLM-Antwort zu lang (%d→%d), ignoriert", len(text), len(corrected)
                )
                return text
            if len(text) > 20 and len(corrected) < len(text) * min_ratio:
                self.logger.warning(
                    "LLM hat Text zu stark gekürzt (%.0f%%), ignoriert",
                    len(corrected) / len(text) * 100,
                )
                return text

            corrected_lower = corrected.lower()
            text_lower = text.lower()

            # Layer 2: Assistant-phrase blacklist
            assistant_phrases = [
                "ich verstehe", "hier ist", "der text lautet", "ich bin bereit",
                "als ki", "als sprachmodell", "entschuldigung", "leider",
                "möchtest du", "danke für", "bitte geben", "bitte eingeben",
                "bitte gib", "damit ich", "ich benötige", "ich brauche",
                "es gibt nichts", "kein text", "keinen text", "keine eingabe",
                "gerne helfe", "gerne korrigiere", "hier der korrigier",
                "natürlich,", "selbstverständlich,",
                "bitte beachte", "bitte beachten",
                "i understand", "here is the", "here's the", "i'm ready",
                "as an ai", "as a language model", "sorry,", "unfortunately,",
                "thank you for", "too short", "not enough",
                "please enter", "please provide", "please type",
                "i need", "i require", "there is nothing",
                "no text", "no input", "of course,", "certainly,",
            ]
            for phrase in assistant_phrases:
                if phrase in corrected_lower and phrase not in text_lower:
                    self.logger.warning(
                        "LLM antwortet als Assistent [phrase='%s'], ignoriert", phrase
                    )
                    return text

            # Layer 3: Suspicious starts (skip when professionalize restructures sentences)
            if not professionalize:
                suspicious_starts = [
                    "ich ", "sie ", "du ", "bitte ", "danke", "vielen",
                    "here ", "i ", "you ", "please ", "thank", "sure,",
                    "of course", "certainly", "natürlich,", "selbstverständlich,",
                ]
                for start_phrase in suspicious_starts:
                    if (corrected_lower.startswith(start_phrase)
                            and not text_lower.startswith(start_phrase)):
                        self.logger.warning(
                            "LLM beginnt Antwort verdächtig [start='%s'], ignoriert", start_phrase
                        )
                        return text

            # Layer 4: Word overlap (relaxed when professionalize or filter_fillers)
            overlap_threshold = 0.25 if professionalize else (0.4 if filter_fillers else 0.55)
            if not self._word_overlap_ok(text, corrected, threshold=overlap_threshold):
                self.logger.warning(
                    "LLM-Wortüberlappung zu gering (threshold=%.0f%%), ignoriert",
                    overlap_threshold * 100,
                )
                return text

            if corrected and corrected != text:
                self.logger.info(
                    "✓ LLM-Korrektur (%.2fs): %s → %s", elapsed, text[:50], corrected[:50]
                )
                return corrected
            else:
                self.logger.info("✓ LLM: Keine Korrektur nötig (%.2fs)", elapsed)
                return text

        except Exception as exc:
            self.logger.warning("LLM-Korrektur fehlgeschlagen: %s", exc)
            return text
