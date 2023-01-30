import pandas as pd
import wikipediaapi
from bs4 import BeautifulSoup as bs
import math
from tqdm import tqdm
import requests
import os


class WikiDataLoader(object):
    
    def __init__(self, config):
        # setting up the url for openaccess csv file with all data information
        self.url = config["MainSettings"]["meturl"]
        self.output_filename = config["DatasetsSettings"]["artists_info_dataset_filename"]
        self.maximum_artists = int(config["DatasetsSettings"]["maximum_artists"])
        # init
        self.list_of_records = list()
    
    def create_dataset(self):
        # running tasks for creating dataset
        self.__convert_data_frame()
        self.__create_dataset()
    
    def __convert_data_frame(self):
        # loads csv file from url
        df = pd.read_csv(self.url)
        attrs = df.keys().tolist()
        # creating new data structure - list of dictionaries - easy access for me
        for record in tqdm(df.values, desc="Creating list of dictionaries from DataFrame"):
            tmp_record = dict()
            for idx, attr in enumerate(attrs):
                if isinstance(record[idx], float):
                    if math.isnan(record[idx]):
                        tmp_record[attr] = None
                else:
                    tmp_record[attr] = record[idx]
            if tmp_record['Artist Wikidata URL'] is not None:
                self.list_of_records.append(tmp_record)

    def __create_dataset(self):
        # creating dataset in form: rows (artist_name \t info about artist from wikipedia)
        authors_with_desc = list()
        os.makedirs(os.path.dirname(self.output_filename), exist_ok=True)
        with open(self.output_filename, "w+", encoding='utf-8') as fw:
            pbar = tqdm(total=self.maximum_artists,
                        desc="Processing the list of dictionaries and creating tsv file for training")
            for record in self.list_of_records:
                if record['Artist Wikidata URL'] is not None:
                    try:
                        # if there are multiple artist for one examplar - creating two different records in dataset
                        if "|" in record['Artist Wikidata URL']:
                            for artist_url in record['Artist Wikidata URL'].split("|"):
                                author_dict = self.__load_artist_info(artist_url)
                        else:
                            author_dict = self.__load_artist_info(record['Artist Wikidata URL'])
                        # creating dataset just with one record per artist
                        if author_dict not in authors_with_desc and author_dict is not None:
                            authors_with_desc.append(author_dict)
                            print(author_dict['artist_name'], author_dict['about'], file=fw, sep="\t")
                            fw.flush()
                            pbar.update(1)
                            if len(authors_with_desc) == self.maximum_artists:
                                fw.close()
                                return
                    except Exception as e:
                        pass

    def __load_artist_info(self, artist_url):
        # extracting artist info from WIKIPEDIA using API
        wiki_wiki = wikipediaapi.Wikipedia('en', extract_format=wikipediaapi.ExtractFormat.WIKI)
        # loading data from wikidata, not from WIKIPEDIA
        r = requests.get(artist_url)
        soup = bs(r.content, features="html.parser")
        # finding artist name
        artist_name = soup.find('span', attrs={"class": ['wikibase-title-label']}).get_text()
        # loading full page of wikipedia (if exists)
        page = wiki_wiki.page(artist_name)
        if page.exists():
            author_dict = dict()
            author_dict['artist_name'] = artist_name
            # if the summary is short -> load another section with text
            if (len(page.summary) < 200) and (len(page.text) > len(page.summary)):
                section = page.sections[0]
                about = page.summary.replace("\n", "") + " " + section.text.replace("\n", "")
            else:
                about = page.summary.replace("\n", "")
            author_dict['about'] = about
            return author_dict
        return None


