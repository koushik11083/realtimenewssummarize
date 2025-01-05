from pytrends.request import TrendReq

def fetch_trending_topics(country="india"):
    pytrends = TrendReq()
    trending = pytrends.trending_searches(pn=country)
    trending_list = trending[0].tolist()
    trending_list = [topic.replace(" ", "-") for topic in trending_list]
    return trending_list[:20]

if __name__ == "__main__":
    print(fetch_trending_topics())
