import queue
import threading

from urllib.parse import quote
import requests
from bs4 import BeautifulSoup as bsoup

# XXX: Takes a lot of time when run sequentially :
# - (1) Segregate networking (I/O) and html-parsing (CPU) as different tasks. ✅
# - (2) Utilize threads for networking (low resource consumption). ✅
# - (3) Utilize multi-core processes for html-parsing (high resource consumption) ❌
# - (4) Use Queues to pass on the fetched HTML pages over to parsing. ✅

def get_html(buffer: queue.Queue, barrier, all_urls, headers, id):
    print(f'Producer {id}: Running')
    for url in all_urls:
        html = requests.get(url, headers=headers).text
        buffer.put(html)
    barrier.wait()
    if id == 0:
        buffer.put(None)
    print(f'Producer {id}: Done')

def get_soup(buffer: queue.Queue, all_soups: queue.Queue):
    print('Parsing soups...')
    while True:
        html_text = buffer.get()
        if html_text is None:
            break
        soup = bsoup(html_text, 'html.parser')
        all_soups.put(soup)

def get_soups_async(all_urls):
    all_soups = queue.Queue(maxsize=100)
    num_producer_threads = 10
    buffer = queue.Queue(maxsize=100)
    barrier = threading.Barrier(num_producer_threads)
    consumer = threading.Thread(target=get_soup, args=(buffer, all_soups))
    consumer.start()

    producers = [threading.Thread(target=get_html, args=(buffer, barrier, all_urls, headers, idx)) for idx in range(num_producer_threads)]
    for producer in producers:
        producer.start()
    for producer in producers:
        producer.join()
    consumer.join()

    soups = []
    size = all_soups.qsize()
    for _ in range(size):
        soups.append(all_soups.get())
    print(len(soups))

    return soups

def get_soup_and_headers(base_url):
    target_url = base_url + "/r/datascience"
    # Headers to mimic a browser visit
    headers = {
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
            AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 \
                Safari/537.36').lstrip()
    }
    # sanitize urls
    target_url = quote(target_url, safe=":/?&")
    base_url = quote(base_url, safe=":/?&")
    # Returns a requests.models.Response object 
    page = requests.get(target_url, headers=headers)
    soup = bsoup(page.text, 'lxml')

    return soup, headers


if __name__ == '__main__':

    base_url = "https://old.reddit.com"
    soup, headers = get_soup_and_headers(base_url)

    # outer containers
    top_matter = soup.find_all("div", attrs={"class": "top-matter"})
    # all further approachable urls
    all_urls = [
        quote(base_url + outer_container.find("a", {"class": "title may-blank"})["href"], safe=":/?&") \
            for outer_container in top_matter if outer_container.find("a", attrs={"class": "title may-blank"})]
    
    # remove extraneous URLs
    all_urls = [url for url in all_urls if "https://old.reddit.com/r/datascience" in url]
    
    soups = get_soups_async(all_urls)

    # Collect topics
    topic_divs = [soup.find("div", {"id": "siteTable", "class": "sitetable linklisting"}) for soup in soups]
    topics = [topic_div.find("a", {"class": "title may-blank"}).text for topic_div in topic_divs]
    # now extract data
    comment_data = [soup.find("div", {"class": "commentarea"}).p.text for soup in soups]
    # zip the question/heading with the underlying comments and data
    zipped_data = list(zip(topics, comment_data))

    assert len(zipped_data) == len(topics)