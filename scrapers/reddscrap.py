from urllib.parse import quote
import requests
from bs4 import BeautifulSoup as bsoup

url = r"https://old.reddit.com/r/datascience"
base_url = r"https://old.reddit.com"
# Headers to mimic a browser visit
headers = {
    'User-Agent': r'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
        AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 \
            Safari/537.36'
}
# sanitize urls
url = quote(url, safe=":/?&")
base_url = quote(base_url, safe=":/?&")
# Returns a requests.models.Response object 
page = requests.get(url, headers=headers)
soup = bsoup(page.text, 'lxml')
# outer containers
top_matter = soup.find_all("div", attrs={"class": "top-matter"})
# all further approachable urls
all_urls = [
    quote(base_url + outer_container.find("a", attrs={"class": "title may-blank"})["href"], safe=":/?&") \
        for outer_container in top_matter if outer_container.find("a", attrs={"class": "title may-blank"})]
# We get some extra urls using the strategy used above, collect all comments
comments = [url for url in all_urls if "https://old.reddit.com/r/datascience" in url]

# Make request for each url
soups = [bsoup(requests.get(url, headers=headers).text, 'lxml') for url in comments]