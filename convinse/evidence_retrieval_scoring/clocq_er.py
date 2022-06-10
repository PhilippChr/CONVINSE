import json
import os
import re
import time
import pickle
import logging
from pathlib import Path
from filelock import FileLock

from clocq.CLOCQ import CLOCQ
from clocq.interface.CLOCQInterfaceClient import CLOCQInterfaceClient

from convinse.library.utils import print_verbose, get_logger
from convinse.evidence_retrieval_scoring.wikipedia_retriever.wikipedia_retriever import (
	WikipediaRetriever,
)

ENT_PATTERN = re.compile("^Q[0-9]+$")
PRE_PATTERN = re.compile("^P[0-9]+$")

KB_ITEM_SEPARATOR = ", "


class ClocqRetriever:
	def __init__(self, config):
		self.config = config
		self.logger = get_logger(__name__, config)

		# load cache
		self.use_cache = config["ers_use_cache"]
		if self.use_cache:
			self.cache_path = config["ers_cache_path"]
			self._init_cache()
			self.cache_changed = False

		# initialize clocq for KB-facts and disambiguations
		if config["clocq_use_api"]:
			self.clocq = CLOCQInterfaceClient(host=config["clocq_host"], port=config["clocq_port"])
		else:
			self.clocq = CLOCQ()

		# initialize wikipedia-retriever
		self.wiki_retriever = WikipediaRetriever(config)
		if config["qu"] == "sr":
			self.sr_delimiter = config["sr_delimiter"].strip()
		else:
			self.sr_delimiter = " "

	def retrieve_evidences(self, structured_representation, sources):
		"""
		Retrieve evidences and question entities
		for the given SR (or other question/text).

		This function is used for initial evidence
		retrieval. These evidences are filtered in the
		next step.

		Can also be used from external modules to access
		all evidences for the given SR (if possible from cache).
		"""
		# KB-facts (always required for question entities)
		all_question_entities = list()
		all_question_entity_ids = set()
		all_evidences = list()

		# remove delimiter from SR
		structured_representation = structured_representation.replace(self.sr_delimiter, " ")

		evidences, question_entities = self.retrieve_KB_facts(structured_representation)
		all_evidences += evidences

		## TODO: might not be required any more, since only single SR considered
		# identify new entities (avoid duplicates)
		new_entities = [
			entity
			for entity in question_entities
			if not entity["item"]["id"] in all_question_entity_ids
		]
		all_question_entities += new_entities

		# update set of entity ids
		all_question_entity_ids.update([entity["item"]["id"] for entity in new_entities])

		# wikipedia evidences (only if required)
		if any(src in sources for src in ["text", "table", "info"]):
			for question_entity in all_question_entities:
				all_evidences += self.retrieve_wikipedia_evidences(question_entity)

		# config-based filtering
		all_evidences = self.filter_evidences(all_evidences, sources)
		return all_evidences, all_question_entities

	def retrieve_wikipedia_evidences(self, question_entity):
		"""
		Retrieve evidences from Wikipedia for the given question entity.
		"""
		question_entity_id = question_entity["item"]["id"]

		# look-up cache
		# if self.use_cache and question_entity_id in self.cache["wikipedia"]:
			# return self.cache["wikipedia"][question_entity_id]

		# used for debugging
		# self.logger.debug(f"No cache hit: Retrieving wikipedia evidences for: {question_entity_id}.")
		# cache_res = self.cache["wikipedia"].get(question_entity_id)
		# if self.use_cache:
		# 	cache_len = len(self.cache["wikipedia"])
		# 	# self.logger.debug(f"Cache result: {cache_res}.")
		# 	self.logger.debug(f"Cache length: {cache_len}.")
	 
		 # retrieve result
		evidences = self.wiki_retriever.retrieve_wp_evidences(question_entity_id)
		for evidence in evidences:
			evidence["retrieved_for_entity"] = question_entity["item"]
 
		assert not evidences is None # evidences should never be None
		# store result in cache
		# if self.use_cache:
			# self.cache_changed = True
			# self.cache["wikipedia"][question_entity_id] = evidences
		return evidences

	def retrieve_KB_facts(self, structured_representation):
		"""
		Retrieve KB facts for the given SR (or other question/text).
		Also returns the question entities, for usage in Wikipedia retriever.
		"""

		def _is_potential_answer(item_id):
			"""Return if item_id could be answer."""
			# keep all KB-items except for predicates
			if PRE_PATTERN.match(item_id):
				return False
			return True

		# look-up cache
		if self.use_cache and structured_representation in self.cache["kb"]:
			return self.cache["kb"][structured_representation]

		self.logger.debug(f"No cache hit: Retrieving search space for: {structured_representation}.")

		# apply CLOCQ
		clocq_result = self.clocq.get_search_space(
			structured_representation, parameters=self.config["clocq_params"], include_labels=True
		)

		# get question entities (predicates dropped)
		question_entities = [
			item
			for item in clocq_result["kb_item_tuple"]
			if not item["item"]["id"] is None and ENT_PATTERN.match(item["item"]["id"])
		]

		question_items_set = set([item["item"]["id"] for item in clocq_result["kb_item_tuple"]])

		# remember potential duplicate facts
		potential_duplicates = set()

		# transform facts to evidences
		evidences = list()
		for fact in clocq_result["search_space"]:
			# evidence text
			evidence_text = self._kb_fact_to_text(fact)

			# entities the fact was retrieved for from clocq
			retrieved_for = [item for item in fact if item["id"] in question_items_set][0]

			# potential duplicate
			if len(retrieved_for) > 1:
				# skip duplicate
				if evidence_text in potential_duplicates:
					continue
				# remember evidence
				potential_duplicates.add(evidence_text)

			evidence = {
				"evidence_text": evidence_text,
				"wikidata_entities": [item for item in fact if _is_potential_answer(item["id"])],
				"disambiguations": [
					(item["label"], item["id"]) for item in fact if ENT_PATTERN.match(item["id"])
				],
				"retrieved_for_entity": retrieved_for,
				"source": "kb",
			}
			evidences.append(evidence)

		# store result in cache
		if self.use_cache:
			self.cache_changed = True
			self.cache["kb"][structured_representation] = (evidences, question_entities)
		return evidences, question_entities

	def retrieve_kb_facts_for_item(self, item_id):
		"""
		Retrieve KB facts with the given KB item ID.
		Returns a list of KB facts, with each KB fact being a tuple of {"id", "label"} dictionaries.
		"""
		# retrieve KB facts via clocq (API)
		return self.clocq.get_neighborhood(item_id, p=self.config["clocq_p"], include_labels=True)

	def filter_evidences(self, evidences, sources):
		"""
		Filter the set of evidences according to their source.
		"""
		filtered_evidences = list()
		for evidence in evidences:
			if len(evidence["wikidata_entities"]) == 1:
				continue
			if len(evidence["wikidata_entities"]) > self.config["evr_max_entities"]:
				continue
			if evidence["source"] in sources:
				filtered_evidences.append(evidence)

		return filtered_evidences

	def _kb_fact_to_text(self, fact):
		"""Verbalize the KB-fact."""
		return KB_ITEM_SEPARATOR.join([item["label"] for item in fact])

	def store_cache(self):
		"""Store the cache to disk."""
		if not self.use_cache: # store only if cache in use
			return
		if not self.cache_changed: # store only if cache changed
			return
		# check if the cache was updated by other processes
		if self._read_cache_version() == self.cache_version:
			# no updates: store and update version
			self.logger.info(f"Writing ER cache at path {self.cache_path}.")
			with FileLock(f"{self.cache_path}.lock"):
				self._write_cache(self.cache)
				self._write_cache_version()
		else:
			# update! read updated version and merge the caches
			self.logger.info(f"Merging ER cache at path {self.cache_path}.")
			with FileLock(f"{self.cache_path}.lock"):
				# read updated version
				updated_cache = self._read_cache()
				# overwrite with changes in current process (most recent)
				updated_cache["kb"].update(self.cache["kb"])
				updated_cache["wikipedia"].update(self.cache["wikipedia"])
				# store
				self._write_cache(updated_cache)
				self._write_cache_version()
		# store extended wikipedia dump (if any changes occured)
		self.wiki_retriever.store_dump()

	def reset_cache(self):
		"""Reset the cache for new population."""
		self.logger.warn(f"Resetting ER cache at path {self.cache_path}.")
		with FileLock(f"{self.cache_path}.lock"):
			self.cache = {"kb": {}, "wikipedia": {}}
			self._write_cache(self.cache)
			self._write_cache_version()

	def _init_cache(self):
		"""Initialize the cache."""
		if os.path.isfile(self.cache_path):
			# remember version read initially
			self.logger.info(f"Loading ER cache from path {self.cache_path}.")
			with FileLock(f"{self.cache_path}.lock"):
				self.cache_version = self._read_cache_version()
				self.logger.debug(self.cache_version)
				self.cache = self._read_cache()
			self.logger.info(f"ER cache successfully loaded.")
		else:
			self.logger.info(f"Could not find an existing ER cache at path {self.cache_path}.")
			self.logger.info("Populating ER cache from scratch!")
			self.cache = {"kb": {}, "wikipedia": {}}
			self._write_cache(self.cache)
			self._write_cache_version()

	def _read_cache(self):
		"""
		Read the current version of the cache.
		This can be different from the version used in this file,
		given that multiple processes may access it simultaneously.
		"""
		# read file content from cache shared across QU methods
		with open(self.cache_path, "rb") as fp:
			cache = pickle.load(fp)
		return cache

	def _write_cache(self, cache):
		"""Write to the cache."""
		cache_dir = os.path.dirname(self.cache_path)
		Path(cache_dir).mkdir(parents=True, exist_ok=True)
		with open(self.cache_path, "wb") as fp:
			pickle.dump(cache, fp)
		return cache

	def _read_cache_version(self):
		"""Read the cache version (hashed timestamp of last update) from a dedicated file."""
		if not os.path.isfile(f"{self.cache_path}.version"):
			self._write_cache_version()
		with open(f"{self.cache_path}.version", "r") as fp:
			cache_version = fp.readline().strip()
		return cache_version

	def _write_cache_version(self):
		"""Write the current cache version (hashed timestamp of current update)."""
		with open(f"{self.cache_path}.version", "w") as fp:
			version = str(time.time())
			fp.write(version)
		self.cache_version = version
