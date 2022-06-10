import json
import os
import random
import logging
import sys
import traceback

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from tqdm import tqdm
from pathlib import Path

from clocq.CLOCQ import CLOCQ
from clocq.interface.CLOCQInterfaceClient import CLOCQInterfaceClient

from structured_representation_annotator import StructuredRepresentationAnnotator
from conv_flow_annotator import ConvFlowAnnotator
from turn_relevance_annotator import TurnRelevanceAnnotator

from convinse.library.utils import get_config, get_logger
from convinse.library.string_library import StringLibrary as string_lib


class NoFlowGraphFoundException(Exception):
    pass

class SilverAnnotation:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)

        # initialize clocq
        if config["clocq_use_api"]:
            self.clocq = CLOCQInterfaceClient(host=config["clocq_host"], port=config["clocq_port"])
        else:
            self.clocq = CLOCQ(dev=True)

        # initialize annotators
        self.sr_annotator = StructuredRepresentationAnnotator(self.clocq, config)
        self.fg_annotator = ConvFlowAnnotator(self.clocq, config)
        self.tr_annotator = TurnRelevanceAnnotator(config)

        #  open labels
        labels_path = config["path_to_labels"]
        with open(labels_path, "r") as fp:
            self.labels_dict = json.load(fp)

    def process_dataset(self, dataset_path, output_path, tr_data_path):
        """
        Annotate the given dataset and store the output in the specified path.
        """
        with open(dataset_path, "r") as fp:
            dataset = json.load(fp)

        # initialize
        total_question_count = 0
        fail_count = 0
        tr_dataset = list()

        # process data
        for conversation in tqdm(dataset):
            # initialize
            flow_graph_found = True # TO BE REMOVED
            for turn in conversation["questions"]:
                turn["silver_SR"] = []
                turn["silver_relevant_turns"] = None

            # annotate data
            try:
                # create conversation flow graph
                flow_graph = self.fg_annotator.get_conv_flow_graph(conversation)
                if not flow_graph:
                    flow_graph_found = False # TO BE REMOVED
                    raise NoFlowGraphFoundException("No flow graph found for conversation!")

                # add annotations
                self.sr_annotator.annotate_structured_representations(flow_graph, conversation)
                tr_data = self.tr_annotator.annotate_turn_relevances(flow_graph, conversation)
                tr_dataset += tr_data

                # count questions
                question_count = self.fg_annotator.get_question_count(flow_graph)
                total_question_count += question_count

            except NoFlowGraphFoundException as e:
                # if no flow graph found, continue with next conversation
                pass
                # else: # TO BE REMOVED
                #     # log error
                #     self.logger.warning(
                #         f"Exception catched at l. 85: {e}, for conversation {conversation}."
                #     )
                #     fail_count += 1

            # parse answers
            if self.config["benchmark"] == "convquestions":
                for turn in conversation["questions"]:
                    answers = string_lib.parse_answers_to_dicts(turn["answer"], self.labels_dict)
                    del turn["answer"]
                    turn["answers"] = answers

        # log
        self.logger.info(f"Done with DS on: {dataset_path}")
        self.logger.info(f"\t#SRs extracted: {total_question_count}")
        self.logger.info(f"\t#Fails: {fail_count}")

        # store annotated dataset
        with open(output_path, "w") as fp:
            fp.write(json.dumps(dataset, indent=4))

        # store turn relevance dataset
        if self.config["tr_extract_dataset"]:
            with open(tr_data_path, "w") as fp:
                fp.write(json.dumps(tr_dataset, indent=4))

    def _plot_graph(self, flow_graph, metadata=False, structured_representations=None):
        """
        FOR DEV: Plot the flow graph using nx and plt.
        """
        if structured_representations:
            structured_representations = {
                turn["question"]: turn["structured_representation"]
                for turn in structured_representations
            }
        if self.config["log_level"] == "DEBUG":
            G = nx.DiGraph()
            leafs = flow_graph["leafs"]
            while leafs:
                for node in leafs:
                    turn = node["turn"]
                    question = node["question"]
                    label = str(turn) + ": " + question
                    if node["type"] == "question" and metadata:
                        if structured_representations:
                            label += "\n" + str(structured_representations[node["question"]])
                        else:
                            label += "\nDIS=" + self._transform_diambiguation_triples(
                                node["relevant_disambiguations"]
                            )
                            label += "\nCXT=" + self._transform_diambiguation_triples(
                                node["relevant_context"]
                            )
                    G.add_node(label)
                    parents = node["parents"]
                    for parent in parents:
                        parent_question = parent["question"]
                        parent_turn = parent["turn"]
                        parent_label = str(parent_turn) + ": " + parent_question
                        if parent["type"] == "question" and metadata:
                            if structured_representations:
                                parent_label += "\n" + str(
                                    structured_representations[parent["question"]]
                                )
                            else:
                                parent_label += "\nDIS=" + self._transform_diambiguation_triples(
                                    parent["relevant_disambiguations"]
                                )
                                parent_label += "\nCXT=" + self._transform_diambiguation_triples(
                                    parent["relevant_context"]
                                )
                        G.add_node(parent_label)
                        G.add_edge(parent_label, label)
                leafs = [node for leaf in leafs for node in leaf["parents"]]

            # add questions that could not be answered to graph
            if flow_graph.get("not_answered"):
                not_answered_label = ""
                for turn in flow_graph["not_answered"]:
                    not_answered_label += str(turn) + "\n"
                not_answered_label = not_answered_label.strip()
                G.add_node(not_answered_label)

            # add conversation id
            if flow_graph.get("conv_id"):
                conv_id = str(flow_graph["conv_id"])
                G.add_node(conv_id)

            nx.nx_agraph.write_dot(
                G, os.path.join(config["silver_annotation_path"], "examples", "test.dot")
            )

            # same layout using matplotlib with no labels
            pos = graphviz_layout(G, prog="dot")
            pos = pos
            plt.figure(figsize=(18, 20))
            nx.draw(G, pos, with_labels=True, arrows=True, node_size=100)
            # nx.draw(G, pos, with_labels=True, arrows=True, node_size=100, figsize=(20, 20), dpi=150)
            plt.xlim([-1, 800])
            plt.show()

    def _transform_diambiguation_triples(self, disambiguated_triples):
        """
        For debugging: transform a list of disambiguation triples into a string for printing in graph.
        """
        disambiguations = dict()
        for item, surface_forms, label in disambiguated_triples:
            for surface_form in surface_forms:
                if disambiguations.get(surface_form):
                    disambiguations[surface_form].append(label)
                else:
                    disambiguations[surface_form] = [label]
        string = ""
        for surface_form in disambiguations:
            string += surface_form + ": " + str(disambiguations[surface_form]) + "\n"
        return string.strip()

    def _random_example(self, dataset_path, index=None):
        """
        For debugging: Compute the flow graph + structured representations for a random example.
        """
        with open(dataset_path, "r") as fp:
            data = json.load(fp)

        # (pseudo) random index
        if index:
            random_index = index
        else:
            random_index = random.randint(0, len(data) - 1)
        self.logger.info(f"Random_index: {random_index}")

        # run example
        for conversation in data[random_index:]:
            # initialize
            for turn in conversation["questions"]:
                turn["silver_SR"] = []
                turn["silver_relevant_turns"] = None

            self.logger.info(conversation["questions"][0]["question"])

            # create conversation flow graph
            flow_graph = self.fg_annotator.get_conv_flow_graph(conversation)
            # self.fg_annotator._print_dict(flow_graph)

            if not flow_graph:
                continue

            # add annotations
            self.sr_annotator.annotate_structured_representations(flow_graph, conversation)
            self.tr_annotator.annotate_turn_relevances(flow_graph, conversation)

            self.fg_annotator._print_dict(conversation)
            # plot graph
            # self._plot_graph(flow_graph, extended=True, structured_representations=structured_representations)
            # break


#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    """
    MAIN
    """
    if len(sys.argv) != 3:
        raise Exception(
            "Usage: python convinse/distant_supervision/silver_annotation.py --FUNCTION <PATH_TO_CONFIG>"
        )

    # load options
    args = sys.argv[1:]
    function = args[0]
    config_path = args[1]
    config = get_config(config_path)
    benchmark_path = config["benchmark_path"]

    # create annotator
    annotator = SilverAnnotation(config)
    
    if function == "--example":
        input_path = os.path.join(benchmark_path, config["train_input_path"])
        annotator._random_example(input_path, index=1351)
    else:
        output_dir = config["path_to_annotated"]
        tr_output_dir = os.path.join(config["path_to_intermediate_results"], "tr")
        Path(tr_output_dir).mkdir(parents=True, exist_ok=True)

        # process data
        input_path = os.path.join(benchmark_path, config["train_input_path"])
        output_path = os.path.join(output_dir, "annotated_train.json")
        tr_data_path = os.path.join(tr_output_dir, "train.json")
        annotator.process_dataset(input_path, output_path, tr_data_path)

        input_path = os.path.join(benchmark_path, config["dev_input_path"])
        output_path = os.path.join(output_dir, "annotated_dev.json")
        tr_data_path = os.path.join(tr_output_dir, "dev.json")
        annotator.process_dataset(input_path, output_path, tr_data_path)

        input_path = os.path.join(benchmark_path, config["test_input_path"])
        output_path = os.path.join(output_dir, "annotated_test.json")
        tr_data_path = os.path.join(tr_output_dir, "test.json")
        annotator.process_dataset(input_path, output_path, tr_data_path)
