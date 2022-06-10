import os
import re
import json
import pickle
import logging
import requests

from convinse.library.string_library import StringLibrary as string_lib
import convinse.library.wikipedia_library as wiki

YEAR_PATTERN = re.compile("[0-9][0-9][0-9][0-9]")
# dmy dates: https://en.wikipedia.org/wiki/Template:Use_dmy_dates
DMY_PATTERN = re.compile("[0-9]+ [A-Z][a-z]* [0-9][0-9][0-9][0-9]")
# mdy dates: https://en.wikipedia.org/wiki/Template:Use_mdy_dates
MDY_PATTERN = re.compile("[A-Z][a-z]* [0-9]+, [0-9][0-9][0-9][0-9]")

MAX_WIKI_PATHS_PER_REQ = 50

# supress warnings on parser errors
logging.getLogger("wikitables").setLevel("ERROR")


class EvidenceAnnotator:
    """
    Annotate evidences with entities, dates and potentially other constants.
    """

    def __init__(self, config, wikidata_mappings):
        self.config = config
        self.wikidata_mappings = wikidata_mappings

        # open Wikidata labels
        with open(config["path_to_labels"], "r") as fp:
            self.labels_dict = json.load(fp)

        # initialize cache
        self.path = f"cache/cache_wikipedia_to_wikidata.pickle"
        self._init_cache()

    def annotate_wikidata_entities(self, wiki_path, evidences, doc_anchor_dict):
        """
        Add Wikidata entities, dates and potentially other constants to evidences.
        """
        # sort anchor-texts by their length
        doc_anchor_tuples = [(key, value) for key, value in doc_anchor_dict.items()]
        doc_anchor_tuples = sorted(doc_anchor_tuples, key=lambda y: len(y[0]), reverse=True)

        # detect wikipedia entities and dates first
        new_evidences = list()

        for evidence in evidences:
            # detect wikipedia entities
            if not evidence.get("source") == "infobox":  # entities for infobox are already done
                wiki_paths, disambiguations = self._detect_wikipedia_entities(
                    wiki_path, evidence, doc_anchor_tuples
                )
                evidence["wikipedia_paths"] = wiki_paths
                evidence["wp_disambiguations"] = disambiguations

            # detect dates
            dates, disambiguations = self._extract_dates(evidence["evidence_text"])
            evidence["wikidata_entities"] = dates
            evidence["disambiguations"] = disambiguations

            # drop evidences with no entities (will only connect to Wikipedia page entity)
            if len(evidence["wikipedia_paths"]) + len(evidence["wikidata_entities"]) <= 1:
                continue
            new_evidences.append(evidence)

        # retrieve redirects
        all_wiki_paths = [
            wiki_path for evidence in evidences for wiki_path in evidence["wikipedia_paths"]
        ]
        redirects = self.extract_redirects(all_wiki_paths)

        # Wikipedia -> Wikidata
        for evidence in evidences:
            wiki_paths = evidence["wikipedia_paths"]
            disambiguations = evidence["wp_disambiguations"]
            # print(wiki_paths, disambiguations, evidence)
            # transformation
            wikidata_entities = list()
            for i, wiki_path in enumerate(wiki_paths):
                wikidata_id = self._wiki_path_to_wikidata(wiki_path, redirects)
                wikidata_entities.append(wikidata_id)
                dis_text, _ = disambiguations[i]
                disambiguations[i] = (dis_text, wikidata_id)

            # drop False values (wiki path could not be translated to wikidata)
            wikidata_entities = [entity for entity in wikidata_entities if entity]
            evidence["wikidata_entities"] += wikidata_entities
            evidence["disambiguations"] += disambiguations
            del evidence["wikipedia_paths"]
            del evidence["wp_disambiguations"]

            # drop duplicates
            evidence["wikidata_entities"] = list(set(evidence["wikidata_entities"]))

            # add labels to Wikidata entities
            evidence["wikidata_entities"] = [
                {"id": item_id, "label": string_lib.item_to_label(item_id, self.labels_dict)}
                for item_id in evidence["wikidata_entities"]
            ]

    def _detect_wikipedia_entities(self, wiki_path, evidence, doc_anchor_tuples):
        """
        Identify Wikipedia entities in the given evidence using
        the given anchor dict for the Wikipedia page.
        Longer matches would be checked first.
        """
        # remember all anchor texts for prunings
        finds = list()
        disambiguations = list()
        evidence_text = evidence["evidence_text"]

        wikipedia_paths = list()
        for anchor_text, anchor_path in doc_anchor_tuples:
            if anchor_text in evidence_text:
                ## do not consider wiki_paths with hashtags
                # hashtag indicates a paragraph on entity, rather than entity
                if "#" in anchor_path:
                    continue

                # search start and end points
                new_start = evidence_text.find(anchor_text)
                new_end = new_start + len(anchor_text)

                ## detect duplicate match for substring
                # positions must be inside range of [_start,_end]
                # since anchor texts are sorted by length
                duplicate = False
                for _start, _end in finds:
                    if new_start >= _start and new_start <= _end:
                        duplicate = True
                    elif new_end >= _start and new_end <= _end:
                        duplicate = True

                # if no duplicate match found -> new anchor
                if not duplicate:
                    finds.append((new_start, new_end))
                    wikipedia_paths.append(anchor_path)
                    disambiguations.append((anchor_text, anchor_path))
                    # print(f"{anchor_text} ||| {anchor_path} ||| {evidence_text}")
                    # print("\n")

        # add path of Wikipedia page entity
        if not wiki_path in wikipedia_paths:
            wikipedia_paths.append(wiki_path)
            wiki_title = wiki._wiki_path_to_title(wiki_path)
            disambiguations.append((wiki_title, wiki_path))

        return wikipedia_paths, disambiguations

    def _extract_dates(self, evidence_text):
        """
        Extract dates in text (added to entities).
        First, text is searched for text, then the dates
        are brought into a compatible format (timestamps).
        """
        # dates in standard Wikipedia formats
        dates = list()
        disambiguations = list()

        # dates in dmy format
        dmy_dates = re.findall(DMY_PATTERN, evidence_text)
        for date in dmy_dates:
            try:
                timestamp = string_lib.convert_date_to_timestamp(date, date_format="dmy")
            except:
                # catch false positive, e.g. "13 and 1997"
                continue
            dates.append(timestamp)
            disambiguations.append((date, timestamp))

        # dates in mdy format
        mdy_dates = re.findall(MDY_PATTERN, evidence_text)
        for date in mdy_dates:
            try:
                timestamp = string_lib.convert_date_to_timestamp(date, date_format="mdy")
            except:
                # catch false positive, e.g. "13 and 1997"
                continue
            dates.append(timestamp)
            disambiguations.append((date, timestamp))

        # years (duplicates with dates are welcome for different answer types)
        years = re.findall(YEAR_PATTERN, evidence_text)
        for year in years:
            timestamp = string_lib.convert_year_to_timestamp(year)
            dates.append(timestamp)
            disambiguations.append((year, timestamp))

        return dates, disambiguations

    def _wiki_path_to_wikidata(self, wiki_path, redirects):
        """
        Transform the given Wikipedia path to the Wikidata ID.
        """
        if self.cache.get(wiki_path):
            return self.cache[wiki_path]

        # try look-up
        if self.wikidata_mappings.get(wiki_path):
            wikidata_id = self.wikidata_mappings.get(wiki_path)
        elif self.wikidata_mappings.get(wiki_path.replace(".", "")):
            wikidata_id = self.wikidata_mappings.get(wiki_path.replace(".", ""))

        # try via redirect look-up
        elif redirects.get(wiki_path):
            wiki_title = redirects.get(wiki_path)
            wiki_path = wiki._wiki_title_to_path(wiki_title)
            wikidata_id = self.wikidata_mappings.get(wiki_path)
            # remember that wiki_path was already translated
            # -> different from None return value
            if wikidata_id is None:
                wikidata_id = False
        else:
            return None

        self.cache[wiki_path] = wikidata_id
        return wikidata_id

    def extract_redirects(self, wiki_paths):
        """
        Extract redirects for set of Wikipedia paths (one entity can have multiple paths).
        Makes use of _extract_redirects_for_50 function to decrease number
        of requests required.
        """

        wiki_paths = list(set(wiki_paths))

        # drop wiki_paths for which Wikidata mapping is already known (redirect not required)
        wiki_paths = [
            wiki_path
            for wiki_path in wiki_paths
            if wiki_path and self._wiki_path_to_wikidata(wiki_path, {}) is None
        ]

        # limit for wiki_paths per request is 50
        start_index = 0
        redirects = dict()
        while start_index < len(wiki_paths):
            end_index = min(start_index + MAX_WIKI_PATHS_PER_REQ, len(wiki_paths) - 1)
            wiki_paths_batch = wiki_paths[start_index:end_index]
            new_redirects = self._extract_redirects_for_50(wiki_paths_batch)
            redirects.update(new_redirects)
            start_index += MAX_WIKI_PATHS_PER_REQ
        return redirects

    def _extract_redirects_for_50(self, wiki_paths):
        """
        Extract redirects of given (max.) 50 Wikipedia paths (one entity can have multiple paths).
        Used in extract_redirects function for efficiency.
        """
        if not wiki_paths:
            return {}

        # initialize
        redirects = dict()

        try:
            # create request url
            wiki_paths_string = "|".join(wiki_paths)
            url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&titles={wiki_paths_string}&redirects"

            # retrieve result
            res = requests.get(url)
            res_dict = json.loads(res.content)

            ## result has mappings:
            #   normalized: wiki_path -> wiki_title
            #   redirects: wiki_title -> redirected wiki_title
            if res_dict["query"].get("normalized"):
                normalized = {
                    normalized["to"]: normalized["from"]
                    for normalized in res_dict["query"]["normalized"]
                }
            else:
                normalized = dict()

            # if redirects not set, no redirects required!
            if not res_dict["query"].get("redirects"):
                return redirects

            # create redirects dict
            for redirect in res_dict["query"]["redirects"]:
                # get key
                if normalized.get(redirect["from"]):
                    key = normalized[redirect["from"]]
                else:
                    key = redirect["from"]

                # add entry
                redirects[key] = redirect["to"]

        # catch exception and log problem
        except Exception as e:
            print(f"Error catched for url: {url}")
            print(e)
            traceback.print_tb(e.__traceback__)

        return redirects

    def store_cache(self):
        """Store the cache to disk."""
        with open(self.path, "wb") as fp:
            pickle.dump(self.cache, fp)

    def _init_cache(self):
        """Initialize the cache."""
        if os.path.isfile(self.path):
            with open(self.path, "rb") as fp:
                self.cache = pickle.load(fp)
        else:
            self.cache = dict()
