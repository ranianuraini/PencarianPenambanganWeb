# Install streamlit jika belum terinstal
# pip install streamlit

import streamlit as st
import requests
from bs4 import BeautifulSoup

def scrape_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    first_page = soup.findAll('li', "dropdown mega-full menu-color1")

    save_categories = []
    for links in first_page:
        category = links.find('a').get('href')
        save_categories.append(category)

    category_search = [save_categories[5], save_categories[6], save_categories[9]]
    datas = []

    for ipages in range(1, 25):
        for category in category_search:
            response_news = requests.get(category + "/" + str(ipages))
            name_category = category.split("/")
            soup_news = BeautifulSoup(response_news.text, 'html.parser')
            pages_news = soup_news.findAll('article', {'class': 'simple-post simple-big clearfix'})

            for items in pages_news:
                get_link_in = items.find("a").get("href")
                response_article = requests.get(get_link_in)
                soup_article = BeautifulSoup(response_article.text, 'html.parser')

                check_title = soup_article.findAll("h1", "post-title")
                if check_title:
                    title = soup_article.find("h1", "post-title").text
                else:
                    title = ""

                label = name_category[-1]

                try:
                    date = soup_article.find("span", "article-date").text
                except AttributeError:
                    date = "Data tanggal tidak ditemukan"

                check_read_more = soup_article.findAll("span", "baca-juga")
                trash1 = ""
                if check_read_more:
                    for read_more in check_read_more:
                        text_trash = read_more.text
                        trash1 += text_trash + ' '
                else:
                    trash1 = ""

                articles = soup_article.find_all('div', {'class': 'post-content clearfix'})
                if articles:
                    article_content = soup_article.find('div', {'class': 'post-content clearfix'}).text
                    article = article_content.replace("\n", " ").replace("\t", " ").replace("\r", "") \
                        .replace(trash1, "").replace("\xa0", "")
                else:
                    article = ""

                check_author = soup_article.findAll("p", "text-muted small mt10")
                if check_author:
                    author = soup_article.find("p", "text-muted small mt10").text.replace("\t\t", " ")
                else:
                    author = ""

                datas.append({
                    'Tanggal': date,
                    'Penulis': author,
                    'Judul': title,
                    'Artikel': article,
                    'Label': label
                })

    return datas

def main():
    st.title("Aplikasi Berita Streamlit")

    url = "https://www.antaranews.com/"
    datas = scrape_data(url)

    st.write("## Data Berita")
    st.write(datas)

if __name__ == "__main__":
    main()
