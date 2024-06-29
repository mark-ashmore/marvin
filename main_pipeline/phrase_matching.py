import spacy
from spacy.matcher import PhraseMatcher

class AssistantPhraseMatcher:
    """A class to encapsulate phrase matching functionality."""

    def __init__(self) -> None:
        self.nlp = spacy.load('en_core_web_sm')
        self.matcher = PhraseMatcher(self.nlp.vocab)

    def add_terms(self, terms: list[str]|tuple[str], id: str) -> None:
        """Add list of terms to matcher with ID."""
        patterns = [self.nlp.make_doc(text) for text in terms]
        self.matcher.add(id, patterns)

    def get_matches(self, doc) -> list[tuple[str, str, int, int]]:
        """Get matches from a doc object. Returns text, id, start, and end."""
        matches = self.matcher(doc)
        match_list = []
        for match_id, start, end in matches:
            span = doc[start:end]
            match_list.append(
                (span.text, self.nlp.vocab.strings[match_id], start, end)
            )
        return match_list
