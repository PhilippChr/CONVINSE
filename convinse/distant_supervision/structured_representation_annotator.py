import os
import json
import sys
import time

from convinse.library.utils import get_logger
from convinse.library.string_library import StringLibrary


class StructuredRepresentationAnnotator:
    def __init__(self, clocq, config):
        self.clocq = clocq
        self.config = config

        self.string_lib = StringLibrary(config)
        with open(config["path_to_labels"], "r") as fp:
            self.labels_dict = json.load(fp)

        self.type_relevance_cache = dict()

    def annotate_structured_representations(self, flow_graph, conversation):
        """
        Get the abstract representations for the questions in the flow_graph.
        """
        # initialize
        structured_representations = dict()
        explored_turns = set()
        leafs = flow_graph["leafs"]

        # search tree bottom-up
        while leafs:
            for node in leafs:
                turn_id = node["turn"]
                node_type = node["type"]
                node_str = str(turn_id) + node_type
                if node_str in explored_turns:
                    continue
                elif node["type"] == "question":
                    # extract SR
                    structured_representation = self._extract_structured_representation(
                        node, turn_id, conversation, structured_representations
                    )
                    structured_representations[turn_id] = structured_representation

                    # add SR to data
                    conversation["questions"][turn_id]["silver_SR"].append(
                        structured_representation
                    )

                explored_turns.add(node_str)
            leafs = [node for leaf in leafs for node in leaf["parents"]]

    def _extract_structured_representation(
        self, node, turn_id, conversation, structured_representations
    ):
        """
        Extract a structured representation for the question.
        """
        question = node["question"]
        if self.config["benchmark"] == "convquestions":
            answers = conversation["questions"][turn_id]["answer"]
            answers = self.string_lib.parse_answers_to_dicts(answers, self.labels_dict)
        else:
            answers = conversation["questions"][turn_id]["answers"]

        disambiguations_context = node["relevant_context"]
        disambiguations_current = node["relevant_disambiguations"]

        # extract common disambiguations
        # -> avoid that same disambiguation is in context and entity slot!
        res = self._extract_common_disambiguations(disambiguations_context, disambiguations_current)
        entities, common_disambiguations, context = res

        # remove entity surface forms from question
        question = self._remove_surface_forms(question, disambiguations_current)

        # derive SR context and entities
        sr_context = self._get_context(entities, common_disambiguations, context)
        sr_entities = self._get_entities(entities, common_disambiguations, context)

        ## derive SR relation
        # remove stopwords
        if self.config["sr_remove_stopwords"]:
            words = self.string_lib.get_question_words(question, ner=None)
            sr_relation = [" ".join(words)]
        else:
            # remove symbols
            sr_relation = [self._normalize_relation_str(question)]
        # shared relation
        if self.config["sr_relation_shared_active"]:
            # if the previous turn had the same predicate, append the previous turns relation here
            if node.get("relation_shared_with"):
                prev_turn = node["relation_shared_with"]["turn"]
                prev_sr = structured_representations[prev_turn]
                prev_relation = prev_sr[2][-1]
                sr_relation.append(prev_relation)

        # derive SR answer type
        sr_answer_type = self._get_answer_type(question, answers)

        # create SR
        structured_representation = (
            sr_context,
            sr_entities,
            sr_relation,
            sr_answer_type,
        )
        return structured_representation

    def _remove_surface_forms(self, question, relevant_disambiguations):
        """
        Remove disambiguated surface forms from question. Sort surface forms by
        length to avoid problems: e.g. removing 'unicorn' before removing 'last unicorn'
        leads to a problem.
        """
        # derive set of surface forms
        distinct_surface_forms = set()
        for (item_id, surface_forms, label) in relevant_disambiguations:
            distinct_surface_forms.update(surface_forms)
        # sort surface forms by string length
        distinct_surface_forms = sorted(distinct_surface_forms, key=lambda j: len(j), reverse=True)
        for surface_form in distinct_surface_forms:
            # mechanism to avoid lowering full question at this point
            start_index = question.lower().find(surface_form.lower())
            if not start_index == -1:
                end_index = start_index + len(surface_form)
                question = question[:start_index] + question[end_index:]
        return question

    def _extract_common_disambiguations(self, disambiguations_context, disambiguations_current):
        """
        Returns the common disambiguations. We care only about surface forms here,
        but compare common items.
        Parameters:
        - disambiguations_context: disambiguations in previous turns
        - disambiguations_current: disambiguations in current turns
        """
        entities = set()
        common_disambiguations = set()
        common_disambiguations_items = set()
        context = set()

        # disambiguated item-ids from current question
        question_item_ids = [item for item, surface_forms, label in disambiguations_current]

        # go through disambiguations in context and check if any common disambiguations exist
        for item, surface_forms, label in disambiguations_context:
            if item in question_item_ids:
                for surface_form in surface_forms: # same entity can occur with different surface forms
                    common_disambiguations.add(surface_form)
                common_disambiguations_items.add(item)
            else:
                for surface_form in surface_forms:
                    context.add(surface_form)

        # other disambiguations, that are not in common, are entities
        for item, surface_forms, label in disambiguations_current:
            # check if entity was already in context (common disambiguation)
            if not item in common_disambiguations_items:
                for surface_form in surface_forms:
                    entities.add(surface_form)
        return list(entities), list(common_disambiguations), list(context)

    def _get_context(self, entities, common_disambiguations, context):
        """
        Get the mentions that provide additional context for the information need,
        beyond the mentioned entity.
        """
        if entities:
            return common_disambiguations + context
        elif common_disambiguations:
            return context
        else:
            return []

    def _get_entities(self, entities, common_disambiguations, context):
        """ Get the mentions, that are most relevant to the information need. """
        if entities:
            return entities
        elif common_disambiguations:
            return common_disambiguations
        else:
            return context

    def _get_answer_type(self, question, answers):
        """
        Get the answer_type from the answer.
        In case the answer has multiple types, compute the most relevant type
        to the question using word2vec similarities
        """
        if self.string_lib.is_year(answers[0]["label"]):
            return "year"
        elif self.string_lib.is_timestamp(answers[0]["id"]):
            return "date"
        elif self.string_lib.is_number(answers[0]["id"]):
            return "number"
        elif self.string_lib.is_entity(answers[0]["id"]):
            type_ = self._get_most_relevant_type(answers)
            if type_ is None:
                return ""
            return type_["label"]
        else:
            return "string"

    def _type_relevance(self, type_id):
        """
        Score the relevance of the type.
        """
        if self.type_relevance_cache.get(type_id):
            return self.type_relevance_cache.get(type_id)
        freq1, freq2 = self.clocq.get_frequency(type_id)
        type_relevance = freq1 + freq2
        self.type_relevance_cache[type_id] = type_relevance
        return type_relevance

    def _get_most_relevant_type(self, answers):
        """
        Get the most relevant type for the item, as given by the type_relevance funtion.
        """
        # fetch types
        all_types = list()
        for item in answers:
            item_id = item["id"]
            types = self.clocq.get_types(item_id)
            if not types:
                continue
            for type_ in types:
                if type_ != "None":
                    all_types.append(type_)
        if not all_types:
            return None
        # sort types by relevance, and take top one
        most_relevant_type = sorted(
            all_types, key=lambda j: self._type_relevance(j["id"]), reverse=True
        )[0]
        return most_relevant_type

    def _get_relevant_types(self, item):
        """
        NOT IN USE: Get only the relevant types for the item.
        E.g. Christopher Nolan has 11 different occupations, but only 3-4 are important.
        Implemented by matching with description, if no exact match found return all types.
        """
        all_types = self.clocq.get_types(item)
        description = descriptions.get(item)
        if not all_types:
            return "unknown"
        elif not description:
            return all_types
        # extract types that have an exact match in the description
        relevant_types = list()
        for candidate in all_types:
            candidate_label = candidate["label"]
            if candidate_label in description:
                relevant_types.append(candidate)
        # if no such exact match found, return all
        if not relevant_types:
            return all_types
        return relevant_types

    def _normalize_relation_str(self, relation_str):
        """Remove punctuation, whitespaces and lower the string."""
        relation_str = (
            relation_str.replace(",", "")
            .replace("!", "")
            .replace("?", "")
            .replace(".", "")
            .replace("'", "")
            .replace('"', "")
            .replace(":", "")
            .replace("’", "")
            .replace("{", "")
            .replace("}", "")
            .replace(" s ", " ")
        )
        relation_str = relation_str.lower()
        relation_str = relation_str.strip()
        return relation_str
