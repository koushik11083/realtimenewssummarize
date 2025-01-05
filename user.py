from fetch_articles import fetch_news_articles
from process_articles import process_articles
from news_summarizer import NewsSummarizer
from article_classifier import ZeroShotNewsClassifier

def main():
    topic=input("Please Enter the Topic that you would like to see the News of:")
    topic=topic.replace(' ','-')
    while True:
        try:
            num = input("Please enter the number of articles to be fetched (1-9): ").strip()
            if not num:
                print("Input cannot be empty. Please enter a number between 1 and 9.")
                continue
            num = int(num)
            if 1 <= num <= 9:
                break
            else:
                print("Please enter a number between 1 and 9.")
        except ValueError:
            print("Invalid input. Please enter a valid number between 1 and 9.")
    while True:
        try:
            length = input("Please enter the length of the summary (1-9): ").strip()
            if not length:
                print("Input cannot be empty. Please enter a number between 1 and 9.")
                continue
            length = int(length)
            if 1 <= length <= 9:
                break
            else:
                print("Please enter a number between 1 and 9.")
        except ValueError:
            print("Invalid input. Please enter a valid number between 1 and 9.")

    print("\nFetching News Articles...")
    all_articles=[]
    all_articles.extend(fetch_news_articles(topic,max_articles=num))

    print("\n Processing Articles...")
    articles=process_articles(all_articles)
    print(f"\nFiltered Articles:{len(articles)}")

    print("\n Classifying Articles...")
    classifier = ZeroShotNewsClassifier()
    categories = ["politics", "technology", "sports", "entertainment", "science", "health", "business", "world news"]
    for article in articles:
        try:
            classification = classifier.classify_text(article['content'], categories)
            article['category'] = classification['labels'][0] if classification['labels'] else "Uncategorized"
        except Exception as e:
            print(f"Error classifying article '{article['title']}': {e}")
            article['category'] = "Uncategorized"

    print("\nSummarizing articles...")
    summarizer = NewsSummarizer(max_length=length)
    for article in articles:
        try:
            article['summary'] = summarizer.summarize(article['content'])
        except Exception as e:
            print(f"Error summarizing article '{article['title']}': {e}")
            article['summary'] = "No content available"

    print("------------------------------------------------------------------------------------------")
    for article in articles:
        print(f"\nTitle: {article['title']}")
        print(f"Category: {article['category']}")
        print(f"Summary: {article['summary']}")
        print("------------------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()


