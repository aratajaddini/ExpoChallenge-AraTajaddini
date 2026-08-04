"""Unit tests for the small_talk canned‑reply module."""

import pytest

from backend.chat.small_talk import _norm, small_talk


class TestSmallTalk:
    """Test suite for the small_talk function and its normalisation."""

    # ─── Exact alias matches ──────────────────────────────────────

    def test_greetings(self):
        assert small_talk("hello") is not None
        assert "documentation assistant" in small_talk("hello")
        assert small_talk("hi") is not None
        assert small_talk("hey") is not None
        assert small_talk("good morning") is not None

    def test_thanks(self):
        reply = small_talk("thanks")
        assert reply is not None
        assert "welcome" in reply.lower()
        assert small_talk("thank you") is not None
        assert small_talk("merci") is not None

    def test_goodbye(self):
        reply = small_talk("bye")
        assert reply is not None
        assert "goodbye" in reply.lower()
        assert small_talk("see you later") is not None

    def test_identity(self):
        reply = small_talk("who are you")
        assert reply is not None
        assert "documentation assistant" in reply
        assert small_talk("what are you") is not None

    def test_help(self):
        reply = small_talk("help")
        assert reply is not None
        assert "waste classes" in reply or "documented" in reply
        assert small_talk("can you help me") is not None

    def test_capabilities(self):
        reply = small_talk("what can you do")
        assert reply is not None
        assert "documented questions" in reply

    def test_how_are_you(self):
        reply = small_talk("how are you")
        assert reply is not None
        assert "running fine" in reply.lower()
        assert small_talk("how's it going") is not None

    def test_persian_aliases(self):
        # سلام (salam) and ممنون (mamnun) should match
        assert small_talk("سلام") is not None
        assert "documentation assistant" in small_talk("سلام")
        assert small_talk("ممنون") is not None
        assert "welcome" in small_talk("ممنون").lower()

    # ─── Regex pattern matches ────────────────────────────────────

    def test_regex_good_morning(self):
        reply = small_talk("good morning, how are you?")
        assert reply is not None
        assert "documentation assistant" in reply

    def test_regex_good_afternoon(self):
        reply = small_talk("good afternoon")
        assert reply is not None
        assert "documentation assistant" in reply

    def test_regex_thanks_with_extra(self):
        reply = small_talk("thanks a lot!")
        assert reply is not None
        assert "welcome" in reply.lower()

    def test_regex_bye(self):
        reply = small_talk("bye for now")
        assert reply is not None
        assert "goodbye" in reply.lower()

    def test_regex_who_are_you(self):
        reply = small_talk("who are you?")
        assert reply is not None
        assert "documentation assistant" in reply

    def test_regex_can_you_help(self):
        reply = small_talk("can you help?")
        assert reply is not None
        assert "waste classes" in reply or "documented" in reply

    def test_regex_how_are_you_variant(self):
        reply = small_talk("how are you doing?")
        assert reply is not None
        assert "running fine" in reply.lower()

    # ─── Unmatched (should return None – fall through) ──────────

    def test_unmatched_question(self):
        # These should not be caught by small_talk
        assert small_talk("what is the model architecture") is None
        assert small_talk("how to install") is None
        assert small_talk("what are the classes") is None

    def test_critical_falls_through(self):
        # This specific question must NOT be matched by a regex
        # because it is a real knowledge‑base query.
        assert small_talk("help me understand why it says uncertain") is None

    def test_out_of_scope(self):
        # Things like "what time is it" should return the scope reply
        reply = small_talk("what time is it")
        assert reply is not None
        assert "don't have access to the clock" in reply

    # ─── Normalisation (optional) ─────────────────────────────────

    def test_norm_handles_persian(self):
        # _norm should keep Persian letters (isalnum returns True)
        normal = _norm("سلام")
        assert normal == "سلام"
        normal = _norm("ممنون")
        assert normal == "ممنون"
        # Mixed with punctuation should be cleaned
        normal = _norm("سلام, چطوری؟")
        assert normal == "سلام چطوری"