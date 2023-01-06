import requests
from bs4 import BeautifulSoup as bsoup

url = "https://old.reddit.com/r/datascience"
# Headers to mimic a browser visit
headers = {
    'User-Agent': r'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
        AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 \
            Safari/537.36'
}

# Returns a requests.models.Response object 
page = requests.get(url, headers=headers)

soup = bsoup(page.text, 'html.parser')

domains = soup.find_all("span", class_="domain")

soup.find_all("span", {"class": "domain", "height": "100px"})

for domain in domains:
    if domain != "(self.datascience)":
        continue
    print(domain.text)
