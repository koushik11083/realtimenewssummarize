from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect
import re


def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^\w\s]', '', text)  # punctuation special characters
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)


def filter_short_or_empty_articles(articles, min_length=50):

    return [
        article for article in articles
        if len(preprocess(article['content'])) >= min_length
    ]


def english(articles):
    english_articles = []
    for article in articles:
        try:
            title_language = detect(article['title'])
            content_language = detect(article['content'])
            if title_language == 'en' and content_language == 'en':
                english_articles.append(article)
        except Exception as e:
            # If detection fails, skip the article
            #print(f"Language detection failed for article: {article['title']}. Error: {e}")
            continue
    return english_articles


def process_articles(articles_list):
    def remove_similar_articles(articles, threshold=0.2):
        texts = [f"{preprocess(article['title'])} {preprocess(article['content'])}" for article in articles]
        tfidf_matrix = TfidfVectorizer().fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        unique_indices = set()
        unique_articles = []

        for i, article in enumerate(articles):
            if i in unique_indices:
                continue
            unique_articles.append(article)
            for j in range(len(articles)):
                if similarity_matrix[i][j] > threshold:
                    unique_indices.add(j)

        return unique_articles

    # Step 1: Remove short or empty articles
    filtered_articles = filter_short_or_empty_articles(articles_list)

    # Step 2: Remove non-English articles
    filtered_articles = english(filtered_articles)

    # Step 3: Remove similar articles
    return remove_similar_articles(filtered_articles)


if __name__ == "__main__":
    from fetch_articles import fetch_news_articles

    articles = fetch_news_articles("vinod-kambli")
    print(f"Original articles count: {len(articles)}")
    processed_articles = process_articles(articles)
    print(f"Filtered articles count: {len(processed_articles)}")
    for i in enumerate(processed_articles):
        print(i)
