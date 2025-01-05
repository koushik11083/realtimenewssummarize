from fetch_trending import fetch_trending_topics
from fetch_articles import fetch_news_articles
from process_articles import process_articles
from news_summarizer import NewsSummarizer
from article_classifier import ZeroShotNewsClassifier

def main():
    # Step 1: Fetch trending topics
    print("Fetching trending topics...")
    topics = fetch_trending_topics()
    print(f"Trending topics: {topics}")

    # Step 2: Fetch news articles
    print("\nFetching news articles...")
    all_articles = []
    for topic in topics[:5]:  # Limit to top 5 topics for demo
        all_articles.extend(fetch_news_articles(topic))

    # Step 3: Process articles
    print("\nProcessing articles...")
    articles = process_articles(all_articles)
    print(f"Filtered {len(articles)} articles.")

    # Step 5: Classify articles
    print("\nClassifying articles...")
    classifier = ZeroShotNewsClassifier()
    categories = ["politics", "technology", "sports", "entertainment", "science", "health", "business", "world news"]
    for article in articles:
        try:
            classification = classifier.classify_text(article['content'], categories)
            article['category'] = classification['labels'][0] if classification['labels'] else "Uncategorized"
        except Exception as e:
            print(f"Error classifying article '{article['title']}': {e}")
            article['category'] = "Uncategorized"

    # Step 4: Summarize articles
    print("\nSummarizing articles...")
    summarizer = NewsSummarizer()
    for article in articles:
        try:
            article['summary'] = summarizer.summarize(article['content'])
        except Exception as e:
            print(f"Error summarizing article '{article['title']}': {e}")
            article['summary'] = "No content available"



    # Step 6: Output results
    for article in articles:
        print(f"\nTitle: {article['title']}")
        print(f"Category: {article['category']}")
        print(f"Summary: {article['summary']}")

if __name__ == "__main__":
    main()
