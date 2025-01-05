import nltk
import re
from typing import List, Dict, Optional
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import heapq

class NewsSummarizer:
    def __init__(self,max_length: int = 3,language: str = 'english',prioritize_recency: bool = True):
        self._download_resources()
        # config
        self.max_length = max_length
        self.language = language
        self.prioritize_recency = prioritize_recency
        # processing
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def _download_resources(self):
        """Download necessary NLTK resources."""
        resources = ['punkt', 'stopwords', 'wordnet']
        for resource in resources:
            try:
                if not nltk.download(resource, quiet=True):
                    raise ValueError(f"Failed to download {resource}.")
            except Exception as e:
                print(f"Resource download warning for {resource}: {e}")

    def _preprocess_text(self, text: str) -> str:
        """
        Text preprocessing: remove special patterns, symbols, and normalize spaces.

        :param text: Input text
        :return: Cleaned and preprocessed text
        """
        text = re.sub(r'\(CNN\)|\(Reuters\)|\(AP\)|[©®™]', '', text)
        text = re.sub(r'[^a-zA-Z\s.]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_named_entities(self, text: str) -> List[str]:
        entities = set()
        for sentence in sent_tokenize(text):
            entities.update(
                word for word in sentence.split() if word[0].isupper() and len(word) > 2
            )
        return list(entities)

    def _calculate_word_importance(self,
                                   words: List[str],
                                   named_entities: List[str]) -> Dict[str, float]:

        filtered_words = [
            word.lower() for word in words if word.lower() not in self.stop_words
        ]

        word_frequencies = {}
        for word in filtered_words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

        for entity in named_entities:
            entity_lower = entity.lower()
            if entity_lower in word_frequencies:
                word_frequencies[entity_lower] *= 2.0

        max_frequency = max(word_frequencies.values(), default=1)
        for word in word_frequencies:
            word_frequencies[word] /= max_frequency

        return word_frequencies

    def _score_sentences(self,
                         sentences: List[str],
                         word_importance: Dict[str, float],
                         named_entities: List[str]) -> Dict[str, float]:

        sentence_scores = {}
        total_sentences = len(sentences)

        for idx, sentence in enumerate(sentences):
            words = word_tokenize(sentence.lower())
            score = sum(word_importance.get(word, 0) for word in words)
            entity_boost = sum(
                1.5 for entity in named_entities if entity.lower() in sentence.lower()
            )

            position_boost = 0.5 if idx < total_sentences * 0.1 or idx > total_sentences * 0.9 else 0
            sentence_scores[sentence] = score * (1 + entity_boost + position_boost)

        return sentence_scores

    def summarize(self,
                  text: str,
                  summary_length: Optional[int] = None) -> str:

        if not text or len(text.split()) < 10:
            return text

        current_max_length = summary_length if summary_length is not None else self.max_length
        processed_text = self._preprocess_text(text)

        sentences = sent_tokenize(text)
        words = word_tokenize(processed_text)
        named_entities = self._extract_named_entities(text)

        word_importance = self._calculate_word_importance(words, named_entities)
        sentence_scores = self._score_sentences(sentences, word_importance, named_entities)

        current_max_length = min(current_max_length, len(sentences))
        summary_sentences = heapq.nlargest(
            current_max_length, sentence_scores, key=sentence_scores.get
        )

        return ' '.join(sentence+"\n" for sentence in sentences if sentence in summary_sentences)

def main():
    # Example usage
    text = """
    Artificial Intelligence is rapidly transforming various industries. 
    Machine learning algorithms are becoming more sophisticated and capable of handling complex tasks. 
    From healthcare to finance, AI is making significant impacts. 
    Researchers are continuously developing new techniques to improve AI's capabilities. 
    Ethical considerations remain a crucial aspect of AI development. 
    The future of AI looks promising but requires careful management and oversight.
    """

    summarizer = NewsSummarizer(max_length=3)
    print("Default 3-sentence summary:")
    print(summarizer.summarize(text))

    print("\n2-sentence summary:")
    print(summarizer.summarize(text, summary_length=2))

    print("\n4-sentence summary:")
    print(summarizer.summarize(text, summary_length=4))

if __name__ == "__main__":
    main()
