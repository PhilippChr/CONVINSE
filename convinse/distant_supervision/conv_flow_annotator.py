import json
import random
import sys

from convinse.library.utils import get_logger
from convinse.library.string_library import StringLibrary


class ConvFlowAnnotator:
    def __init__(self, clocq, config):
        self.clocq = clocq
        self.config = config
        self.logger = get_logger(__name__, config)

        self.string_lib = StringLibrary(config)

    def get_conv_flow_graph(self, conversation):
        """
        Extract a noisy history graph for the given conversation.
        Questions asking for years, countries, and existential questions
        are dropped since the answering paths are usually noisy (spurious paths).
        """
        # load data
        turns = conversation["questions"]

        # parsing answers for ConvQuestions
        if self.config["benchmark"] == "convquestions":
            question_answer_pairs = [
                (turn["question"], self.string_lib.parse_answers_to_ids(turn["answer"]))
                for turn in turns[1:]
            ]
            first_question = turns[0]["question"]
            first_answers = self.string_lib.parse_answers_to_ids(turns[0]["answer"])
        # parsing answers for ConvMix
        else:
            question_answer_pairs = [
                (turn["question"], [answer["id"] for answer in turn["answers"]])
                for turn in turns[1:]
            ]
            first_question = turns[0]["question"]
            first_answers = [answer["id"] for answer in turns[0]["answers"]]

        # if self._prune_question_answer_pair(first_answers):
        # return None

        # log loaded data
        self.logger.debug(f"First question: {first_question}")
        self.logger.debug(f"First answers: {first_answers}")

        # process initial question to initialize flow graph
        (
            first_relevant_disambiguations,
            answering_facts,
        ) = self._get_relevant_entities_initial_turn(first_question, first_answers)
        if not first_relevant_disambiguations:
            return None

        # create node for question in flow graph
        question_node = {
            "question": first_question,
            "turn": 0,
            "relevant_disambiguations": first_relevant_disambiguations,
            "relevant_context": [],
            "answering_facts": answering_facts,
            "type": "question",
            "parents": [],
        }

        # create node for answer in flow graph
        answer_label, answer_disambiguations = self.transform_answers(first_answers)
        answer_node = {
            "question": answer_label,
            "turn": 0,
            "relevant_disambiguations": answer_disambiguations,
            "relevant_context": [],
            "type": "answer",
            "parents": [question_node],
        }

        # create initial flow graph
        conv_flow_graph = {"leafs": [answer_node], "not_answered": []}

        # process follow-up questions to populate flow-graph
        for i, (question, answers) in enumerate(question_answer_pairs):
            turn = i + 1
            # prune QA pairs with "Yes"/"No"/years or countries as answer (too noisy matches)
            if self._prune_question_answer_pair(answers):
                conv_flow_graph["not_answered"].append([turn, question, answers])
                continue
            self.logger.debug(f"Question: {question}")
            self.logger.debug(f"Answers: {answers}")
            conv_flow_graph = self._get_relevant_entities_followup(
                question, answers, turn, conv_flow_graph
            )

        # return populated flow graph
        return conv_flow_graph

    def _get_relevant_entities_initial_turn(self, question, answers):
        """
        Get relevant entities for the first turn.
        """
        clocq_result = self.clocq.get_search_space(
            question, parameters=self.config["clocq_params"], include_labels=True
        )
        facts = clocq_result["search_space"]
        kb_item_tuple = clocq_result["kb_item_tuple"]
        disambiguated_items = [item["item"]["id"] for item in kb_item_tuple]
        answering_facts = self._get_answering_facts(facts, answers, disambiguated_items)
        disambiguation_triples = self._get_answer_connecting_disambiguations(
            kb_item_tuple, answering_facts, answers
        )
        return disambiguation_triples, answering_facts

    def _get_relevant_entities_followup(self, question, answers, turn, conv_flow_graph):
        """
        Get relevant entities for a follow up question. Begin your graph traversal in the leaf nodes,
        and traverse up to the root, until the answer is found.
        """
        # remember whether answer was found
        answer_found = False

        # check whether question can be answered without context
        clocq_result = self.clocq.get_search_space(
            question, parameters=self.config["clocq_params"], include_labels=True
        )
        facts = clocq_result["search_space"]
        kb_item_tuple = clocq_result["kb_item_tuple"]
        disambiguated_items = [item["item"]["id"] for item in kb_item_tuple]
        all_entities = set(disambiguated_items)
        answering_facts = self._get_answering_facts(facts, answers, disambiguated_items)
        if answering_facts:
            answer_found = True
            # add disambiguation triples (kb_item_id, surface_forms, label) for answering facts
            disambiguation_triples = self._get_answer_connecting_disambiguations(
                kb_item_tuple, answering_facts, answers
            )
        else:
            disambiguation_triples = []

        # create a new node for the follow-up question
        new_node = {
            "question": question,
            "turn": turn,
            "relevant_disambiguations": disambiguation_triples,
            "relevant_context": [],
            "answering_facts": answering_facts,
            "type": "question",
            "parents": [],
            "relation_shared_with": None,
        }

        # initialize traversal of context
        explored_turns = set()
        leafs_to_remove = list()  # remember leafs that need to be removed
        prev_question_node = None

        # explore leafs first, then go one layer higher (to leaf's parents)
        leafs = conv_flow_graph["leafs"]
        while leafs:
            for node in leafs:
                turn_id = node["turn"]
                node_str = str(turn_id) + node["type"]
                if (node_str) in explored_turns:  # context question was already processed
                    continue

                # remember the context question
                if node["turn"] == (turn - 1) and node["type"] == "question":
                    prev_question_node = node
                explored_turns.add(node_str)

                # load disambiguations for the context question
                context_dis_triples = node["relevant_disambiguations"]

                # bring disambiguations into CLOCQ format
                kb_item_tuple = [
                    {"item": {"id": item, "label": label}, "question_word": surface_forms[0]}
                    for item, surface_forms, label in context_dis_triples
                ]

                # retrieve facts for this context question (~search space, but for incomplete using prev results)
                facts = list()
                for item, surface_forms, label in context_dis_triples:
                    # check whether item already processed
                    if item in all_entities:
                        continue
                    else:
                        all_entities.add(item)

                    # log fact retrieval
                    self.logger.debug(f"Retrieve facts via CLOCQ for: {item}")

                    # retrieve facts for context entity
                    facts += self.clocq.get_neighborhood(
                        item, p=self.config["clocq_p"], include_labels=True
                    )

                # get answer connecting facts from search space
                answering_facts = self._get_answering_facts(facts, answers, disambiguated_items)

                # bring disambiguations into CLOCQ output format
                if answering_facts:
                    answer_found = True
                    relevant_context_dis_triples = self._get_answer_connecting_disambiguations(
                        kb_item_tuple, answering_facts, answers
                    )
                    new_node["relevant_context"] += relevant_context_dis_triples
                    new_node["answering_facts"] += answering_facts
                    new_node["parents"].append(node)
                    # if this context node was a leaf, it won't be a leaf any more (will be parent node of current question)
                    if node in conv_flow_graph["leafs"]:
                        leafs_to_remove.append(node)

            # go through next layer
            leafs = [node for leaf in leafs for node in leaf["parents"]]
            in_leafs = False  # not in leafs any more, since moving up

            # log next layer
            self.logger.debug(f"Leafs: {[leaf['question'] for leaf in leafs]}")

        # check if the previous question node had the same predicate
        if prev_question_node and self._check_if_answering_paths_shared(
            new_node, prev_question_node
        ):
            new_node["relation_shared_with"] = prev_question_node
            if not prev_question_node in new_node["parents"]:
                new_node["parents"].append(prev_question_node)

        # remove leaf nodes; if this is done on the fly, nodes are potentially skipped
        for leaf in leafs_to_remove:
            if node in conv_flow_graph["leafs"]:
                conv_flow_graph["leafs"].remove(node)

        # only add turn to flow graph if answer was found
        if answer_found:
            answer_label, answer_disambiguations = self.transform_answers(answers)
            new_answer_node = {
                "question": answer_label,
                "turn": turn,
                "relevant_disambiguations": answer_disambiguations,
                "type": "answer",
                "relevant_context": [],
                "parents": [new_node],
            }
            conv_flow_graph["leafs"].append(new_answer_node)
        else:  # otherwise, append to list of unanswered questions
            conv_flow_graph["not_answered"].append([turn, question, answers])

        # return new flow graph
        return conv_flow_graph

    def _get_answering_facts(self, facts, answers, disambiguated_items):
        """
        Among the given facts, extract the subset that has the answers. If the
        disambiguated item is already an answer, no need to return full 1-hop of
        such an answer (full 1-hop has answer => is answering).
        """
        # check if answer is among disambiguated items
        for answer in answers:
            if answer in disambiguated_items:
                # data point is not meaningful in this case
                # => very likely a spurious path!
                return []
        # check whether answer is in facts
        answering_facts = list()
        for fact in facts:
            if self._fact_has_answer(answers, fact):
                fact = [{"id": item["id"], "label": item["label"]} for item in fact]
                answering_facts.append(fact)
        return answering_facts

    def _fact_has_answer(self, answers, fact):
        """
        Check if the given answer occurs in the fact.
        Follows evaluation-code of CLOCQ.
        """
        for item in fact:
            item = item["id"]
            if not item or len(item) < 2:
                continue
            item = item.replace('"', "").replace("+", "")
            if self.string_lib.is_timestamp(item):
                year = self.string_lib.get_year(item)
                if year in answers:
                    return True
                # fix for new year format in ConvMix
                elif self.string_lib.convert_year_to_timestamp(year) in answers:
                    return True
            if item in answers:
                return True

    def _get_answer_connecting_disambiguations(self, disambiguations, answering_facts, answers):
        """
        Extract the relevant disambiguations using the answering facts.
        Returns disambiguation triples, which have the following form:
        (kb_item_id, surface_forms, label).
        There are multiple surface forms, since the same KB item can potentially
        be disambiguated for several different question words.
        """
        # create dict from item_id to surface forms
        inverse_disambiguations = dict()
        local_labels = dict()
        for disambiguation in disambiguations:
            surface_form = disambiguation["question_word"]
            item_id = disambiguation["item"]["id"]
            label = disambiguation["item"]["label"]

            # skip disambiguations that are the answer
            if item_id in answers:
                continue

            local_labels[item_id] = label
            if item_id in inverse_disambiguations:
                inverse_disambiguations[item_id].append(surface_form)
            else:
                inverse_disambiguations[item_id] = [surface_form]
        # create disambiguation triples
        disambiguation_triples = list()
        for fact in answering_facts:
            # get items that led to the answering fact coming into the context
            for item in fact:
                item_id = item["id"]
                label = item["label"]
                if item_id in inverse_disambiguations and self._valid_item(item["id"]):
                    surface_forms = inverse_disambiguations[item_id]
                    label = local_labels[item_id]
                    if not (item_id, surface_forms, label) in disambiguation_triples:
                        disambiguation_triples.append((item_id, surface_forms, label))
        return disambiguation_triples

    def _valid_item(self, item):
        """
        Verify that the item is valid.
        1.  Checks whether the item is very frequent. For frequent items,
            the occurence in an extracted fact could be misleading.
        2.  Checks whether item is a predicate (predicates go into relation slot, not entity slot)
        """
        if item[0] == "P": # predicates are dropped 
            return False
        if self._item_is_country(item): # countries are always frequent
            return False
        freq1, freq2 = self.clocq.get_frequency(item)
        freq = freq1 + freq2
        return freq < 100000

    def _item_occurs_in_disambiguations(self, item, disambiguations):
        """
        Check whether the given item occurs in the disambiguations.
        """
        for disambiguation in disambiguations:
            if [item] in disambiguation.values():
                return True
        return False

    def _check_if_answering_paths_shared(self, question_node, previous_question_node):
        """
        Check if the two question nodes share an answering path
        if yes, they most likely share the predicate.
        """
        answering_facts = question_node["answering_facts"]
        prev_answering_facts = previous_question_node["answering_facts"]
        answering_paths = [
            answering_fact[1]["id"] for answering_fact in answering_facts if len(answering_fact) > 1
        ]
        prev_answering_paths = [
            answering_fact[1]["id"]
            for answering_fact in prev_answering_facts
            if len(answering_fact) > 1
        ]
        intersection = set(answering_paths) & set(prev_answering_paths)
        return len(intersection) > 0

    def transform_answers(self, answers):
        """
        Transform the answers into a string using the labels and the corresponding disambiguation dict.
        """
        answer_labels = []
        relevant_disambiguations = []
        for item in answers:
            if item[0] == "Q":
                label = self.clocq.get_label(item)
            else:
                label = item
            relevant_disambiguations.append((item, [label], label))
            answer_labels.append(label)
        return str(answer_labels), relevant_disambiguations

    def _item_is_country(self, item_id):
        """
        Check if the item is of type country.
        """
        if item_id[0] != "Q":
            return False
        types = self.clocq.get_types(item_id)

        if not types or types == ["None"]:
            return False
        type_ids = [type_["id"] for type_ in types]
        if "Q6256" in type_ids: # country type
            return True
        return False

    def _prune_question_answer_pair(self, answers):
        """
        Some answers trigger to many incorrect reasoning paths
        -> question-answer pairs are dropped.
        """
        if answers[0] in ["Yes", "No"]:
            return True
        elif any(
            (self.string_lib.is_year(answer) or self._item_is_country(answer)) for answer in answers
        ):
            return True
        return False

    def get_question_count(self, conv_flow_graph):
        """
        Count the number of questions in the history graph.
        """
        if not conv_flow_graph:
            return 0
        explored_turns = set()
        question_count = 0
        leafs = conv_flow_graph["leafs"]
        while leafs:
            for node in leafs:
                if node["type"] == "answer":
                    continue
                elif (str(node["turn"]) + node["type"]) in explored_turns:
                    continue
                explored_turns.add(str(node["turn"]) + node["type"])
                question_count += 1
            leafs = [node for leaf in leafs for node in leaf["parents"]]
        return question_count

    def _print_dict(self, python_dict):
        """
        Print python dict as json.
        """
        # if self.verbose:
        jsonobject = json.dumps(python_dict)
        self.logger.info(jsonobject)

