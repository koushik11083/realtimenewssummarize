import torch
from transformers import pipeline
from typing import List, Dict, Union


class ZeroShotNewsClassifier:
    def __init__(self, model: str = "facebook/bart-large-mnli"):
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model,
            device=device
        )

    def classify_text(self,
                      text: str,
                      candidate_labels: List[str] = None,
                      multi_label: bool = True) -> Dict[str, Union[List[str], List[float]]]:

        # Default labels if not provided
        if candidate_labels is None:
            candidate_labels = [
                'politics',
                'technology',
                'sports',
                'entertainment',
                'science',
                'health',
                'business',
                'world news'
            ]

        # Truncate very long texts to prevent processing issues
        max_length = 1000
        truncated_text = text[:max_length]

        # Perform zero-shot classification
        classification_result = self.classifier(
            truncated_text,
            candidate_labels,
            multi_label=multi_label
        )

        return classification_result

    def classify_multiple_texts(self,
                                texts: List[str],
                                candidate_labels: List[str] = None) -> List[Dict]:

        results = []
        for text in texts:
            classification = self.classify_text(
                text,
                candidate_labels=candidate_labels
            )
            results.append(classification)

        return results


def main():
    # Example usage
    classifier = ZeroShotNewsClassifier()

    # Example text
    sample_text = """
    Artificial Intelligence continues to make significant strides in various industries. 
    Researchers have developed new machine learning algorithms that can process complex data 
    more efficiently than ever before. The potential applications range from healthcare 
    diagnostics to autonomous vehicle technology, promising to revolutionize multiple sectors.
    """

    # Custom label categories
    custom_labels = [
        'politics',
        'technology',
        'sports',
        'entertainment',
        'science',
        'health',
        'business',
        'world news'
    ]

    # Classify the text
    results = classifier.classify_text(
        sample_text,
        candidate_labels=custom_labels
    )

    # Print classification results
    print("Classification Results:")
    for label, score in zip(results['labels'], results['scores']):
        print(f"{label}: {score:.2%}")


if __name__ == "__main__":
    main()