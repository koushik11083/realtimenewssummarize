from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from newspaper import Article, ArticleException
import requests

def fetch_news_articles(topic, max_articles=10):
    root = "https://www.google.com/"
    search_url = f"https://www.google.com/search?q={topic}&hl=en&tbm=nws&tbs=qdr:d"

    req = Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req)
    soup = BeautifulSoup(webpage, 'html5lib')

    articles_list = []
    for item in soup.find_all('div', attrs={'class': 'Gx5Zad xpd EtOod pkphOe'}):
        if len(articles_list) >= max_articles:
            break
        raw_link = item.find('a', href=True)['href']
        final_link = (raw_link.split("/url?q=")[1]).split("&sa=U&")[0]

        try:
            article = Article(final_link)
            article.download()
            article.parse()
            articles_list.append({
                'url': final_link,
                'title': article.title,
                'content': article.text
            })
        except ArticleException:
            pass

    return articles_list

if __name__ == "__main__":
    print(fetch_news_articles("Artificial-Intelligence"))
