import os
import re
import requests
import spacy
import sys
import time
import pickle
import json
from bs4 import BeautifulSoup

from convinse.library.utils import get_config, get_logger
import convinse.library.wikipedia_library as wiki

from convinse.evidence_retrieval_scoring.wikipedia_retriever.text_parser import (
    extract_text_snippets,
)
from convinse.evidence_retrieval_scoring.wikipedia_retriever.table_parser import (
    extract_wikipedia_tables,
    json_tables_to_evidences,
)
from convinse.evidence_retrieval_scoring.wikipedia_retriever.infobox_parser import (
    InfoboxParser,
    infobox_to_evidences,
)
from convinse.evidence_retrieval_scoring.wikipedia_retriever.evidence_annotator import (
    EvidenceAnnotator,
)


API_URL = "http://en.wikipedia.org/w/api.php"
PARAMS = {
    "prop": "extracts|revisions",
    "format": "json",
    "action": "query",
    "explaintext": "",
    "rvprop": "content",
}

YEAR_PATTERN = re.compile("^[0-9][0-9][0-9][0-9]$")
WIKI_DATE_PATTERN = re.compile("[0-9]+ [A-Z][a-z]* [0-9][0-9][0-9][0-9]")


class WikipediaRetriever:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)

        # whether Wikipedia evidences are retrieved on the fly (i.e. from the Wikipedia API)
        self.on_the_fly = config["ers_on_the_fly"]

        # initialize dump
        self._init_wikipedia_dump()

        if self.on_the_fly:
            # open dicts
            with open(config["path_to_wikidata_mappings"], "r") as fp:
                self.wikidata_mappings = json.load(fp)
            with open(config["path_to_wikipedia_mappings"], "r") as fp:
                self.wikipedia_mappings = json.load(fp)

            # initialize evidence annotator (used for (text)->Wikipedia->Wikidata)
            self.annotator = EvidenceAnnotator(config, self.wikidata_mappings)

            # load nlp pipeline
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("sentencizer")
        self.logger.debug("WikipediaRetriever successfully initialized!")

    def retrieve_wp_evidences(self, question_entity_id):
        """
        Retrieve evidences from Wikipedia for the given Wikipedia title.
        Always returns the full set of evidences (text, table, infobox).
        Filtering is done via filter_evidences function.
        """
        if question_entity_id in self.wikipedia_dump:
            self.logger.debug(f"Found Wikipedia evidences in dump!")
            return self.wikipedia_dump.get(question_entity_id)

        if not self.on_the_fly:
            self.logger.debug(f"No Wikipedia evidences in dump, but on-the-fly retrieval not active!")
            return []

        # get Wikipedia title
        wiki_path = self.wikipedia_mappings.get(question_entity_id)
        if not wiki_path:
            self.logger.debug(f"No Wikipedia link found for this Wikidata ID: {question_entity_id}.")
            self.wikipedia_dump[question_entity_id] = [] # remember
            return []
        self.logger.debug(f"Retrieving Wikipedia evidences for: {wiki_path}.")

        # retrieve Wikipedia soup
        wiki_title = wiki._wiki_path_to_title(wiki_path)
        soup = self._retrieve_soup(wiki_title)
        if soup is None:
            self.wikipedia_dump[question_entity_id] = [] # remember
            return []

        # retrieve Wikipedia markdown
        wiki_md = self._retrieve_markdown(wiki_title)

        # extract anchors
        doc_anchor_dict = self._build_document_anchor_dict(soup)

        # retrieve evidences
        infobox_evidences = self._retrieve_infobox_entries(wiki_title, soup, doc_anchor_dict)
        table_records = self._retrieve_table_records(wiki_title, wiki_md)
        text_snippets = self._retrieve_text_snippets(wiki_title, wiki_md)

        # prune e.g. too long evidences
        evidences = infobox_evidences + table_records + text_snippets
        evidences = self.filter_and_clean_evidences(evidences)

        ## add wikidata entities (for table and text)
        # evidences with no wikidata entities (except for the wiki_path) are dropped
        self.annotator.annotate_wikidata_entities(wiki_path, evidences, doc_anchor_dict)

        # store result in dump
        self.wikipedia_dump[question_entity_id] = evidences

        self.logger.debug(f"Evidences successfully retrieved for {question_entity_id}.")
        return evidences

    def filter_and_clean_evidences(self, evidences):
        """
        Drop evidences which do not suffice specific
        criteria. E.g. such evidences could be too
        short, long, or contain too many symbols.
        """
        filtered_evidences = list()
        for evidence in evidences:
            evidence_text = evidence["evidence_text"]
            ## filter evidences
            # too short
            if len(evidence_text) < self.config["evr_min_evidence_length"]:
                continue
            # too long
            if len(evidence_text) > self.config["evr_max_evidence_length"]:
                continue
            # ratio of letters very low
            letters = sum(c.isalpha() for c in evidence_text)
            if letters < len(evidence_text) / 2:
                continue

            ## clean evidence
            evidence_text = self.clean_evidence(evidence_text)
            evidence["evidence_text"] = evidence_text
            filtered_evidences.append(evidence)
        return filtered_evidences

    def clean_evidence(self, evidence_text):
        """Clean the given evidence text."""
        evidence_text = re.sub(r"\[[0-9]*\]", "", evidence_text)
        return evidence_text

    def _retrieve_infobox_entries(self, wiki_title, soup, doc_anchor_dict):
        """
        Retrieve infobox entries for the given Wikipedia entity.
        """
        # get infobox (only one infobox possible)
        infoboxes = soup.find_all("table", {"class": "infobox"})
        if not infoboxes:
            return []
        infobox = infoboxes[0]

        # parse infobox content
        p = InfoboxParser(doc_anchor_dict)
        infobox_html = str(infobox)
        p.feed(infobox_html)

        # transform parsed infobox to evidences
        infobox_parsed = p.tables[0]
        evidences = infobox_to_evidences(infobox_parsed, wiki_title)
        return evidences

    def _retrieve_table_records(self, wiki_title, wiki_md):
        """
        Retrieve table records for the given Wikipedia entity.
        """
        # extract wikipedia tables
        tables = extract_wikipedia_tables(wiki_md)

        # extract evidences from tables
        evidences = json_tables_to_evidences(tables, wiki_title)
        return evidences

    def _retrieve_text_snippets(self, wiki_title, wiki_md):
        """
        Retrieve text snippets for the given Wikidata entity.
        """
        evidences = extract_text_snippets(wiki_md, wiki_title, self.nlp)
        return evidences

    def _build_document_anchor_dict(self, soup):
        """
        Establishes a dictionary that maps from Wikipedia text
        to the Wikipedia entity (=link). Is used to map to
        Wikidata entities (via Wikipedia) later.
        Format: text -> Wikidata entity.
        """
        # prune navigation bar
        for div in soup.find_all("div", {"class": "navbox"}):
            div.decompose()

        # go through links
        anchor_dict = dict()
        for tag in soup.find_all("a"):
            # anchor text
            text = tag.text.strip()
            if len(text) < 3:
                continue
            # duplicate anchor text (keep first)
            # -> later ones can be more specific/incorrect
            if anchor_dict.get(text):
                continue

            # wiki title (=entity)
            href = tag.attrs.get("href")
            if not wiki.is_wikipedia_path(href):
                continue
            wiki_path = wiki.format_wiki_path(href)

            anchor_dict[text] = wiki_path
        return anchor_dict

    def _retrieve_soup(self, wiki_title):
        """
        Retrieve Wikipedia html for the given Wikipedia Title.
        """
        wiki_path = wiki._wiki_title_to_path(wiki_title)
        link = f"https://en.wikipedia.org/wiki/{wiki_path}"
        try:
            html = requests.get(link).text
            soup = BeautifulSoup(html, features="html.parser")
        except:
            return None
        return soup

    def _retrieve_markdown(self, wiki_title):
        """
        Retrieve the content of the given wikipedia title.
        """
        params = PARAMS.copy()
        params["titles"] = wiki_title
        try:
            # make request
            r = requests.get(API_URL, params=params)
            res = r.json()
        except:
            return None
        pages = res["query"]["pages"]
        page = list(pages.values())[0]
        return page

    def _init_wikipedia_dump(self):
        """
        Initialize the Wikipedia dump. The consists of a mapping
        from Wikidata IDs to Wikipedia evidences in the expected format.
        """
        path_to_dump = self.config["ers_wikipedia_dump"]
        with open(path_to_dump, "rb") as fp:
            self.wikipedia_dump = pickle.load(fp)
        self.wikipedia_dump_version = len(self.wikipedia_dump)

    def store_dump(self):
        """Store the updated Wikipedia dump."""
        if len(self.wikipedia_dump) > self.wikipedia_dump_version:
            self.logger.info("Wikipedia dump extended! Storing data on disk.")
            path_to_dump = self.config["ers_wikipedia_dump"]
            with open(path_to_dump, "wb") as fp:
                pickle.dump(self.wikipedia_dump, fp)

#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    # RUN: python convinse.evidence_retrieval_scoring/wikipedia_retriever/wikipedia_retriever.py config/convmix/convinse.yml
    if len(sys.argv) != 2:
        raise Exception(
            "python convinse.evidence_retrieval_scoring/wikipedia_retriever/wikipedia_retriever.py <PATH_TO_CONFIG>"
        )

    # load config
    config_path = sys.argv[1]
    config = get_config(config_path)

    # create retriever
    retriever = WikipediaRetriever(config)

    # retrieve evidences
    start = time.time()
    question_entity = {"id": "Q23572", "label": "Game of Thrones"}
    evidences = retriever.retrieve_wp_evidences(question_entity["id"])
    print("Time consumed", time.time() - start)

    # show evidences
    for evidence in evidences:
        print(evidence)
        break
