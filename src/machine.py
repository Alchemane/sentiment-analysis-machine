from bs4 import BeautifulSoup
import requests, nltk

class Scraper:
    def __init__(self, url):
        self.url = url

    def scraper(self):
        response = requests.get(self.url)
        soup = BeautifulSoup(response.text, 'html.parser')

        text = ' '.join(soup.stripped_strings)

if __name__ == "__main__":
    scraper = Scraper(url='https://nasa.com')
    text = scraper.scraper()