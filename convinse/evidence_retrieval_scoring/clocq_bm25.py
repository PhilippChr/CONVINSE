import os
import sys
import json
import time
import logging

from tqdm import tqdm
from pathlib import Path

from convinse.library.utils import get_config, get_logger
from convinse.evidence_retrieval_scoring.evidence_retrieval_scoring import EvidenceRetrievalScoring
from convinse.evidence_retrieval_scoring.clocq_er import ClocqRetriever
from convinse.evidence_retrieval_scoring.bm25_es import BM25Scoring


class ClocqBM25(EvidenceRetrievalScoring):
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.evr = ClocqRetriever(config)
        self.evs = BM25Scoring(config)

    def inference_on_turn(self, turn, sources=["kb", "text", "table", "info"]):
        """Retrieve best evidences for SR."""
        structured_representation = turn["structured_representation"]
        evidences, _ = self.evr.retrieve_evidences(structured_representation, sources)
        top_evidences = self.evs.get_top_evidences(structured_representation, evidences)
        turn["top_evidences"] = top_evidences
        return top_evidences

    def store_cache(self):
        """Store cache of evidence retriever."""
        self.evr.store_cache()


#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("python convinse/evidence_retrieval_scoring/clocq_bm25.py <PATH_TO_CONFIG>")

    # load config
    config_path = sys.argv[1]
    config = get_config(config_path)
    ers = ClocqBM25(config)

    # inference: add predictions to data
    input_dir = config["path_to_annotated"]
    output_dir = config["path_to_intermediate_results"]

    qu = config["qu"]
    source_combinations = config["source_combinations"]

    for sources in source_combinations:
        sources_string = "_".join(sources)

        input_path = os.path.join(input_dir, qu, "train_qu.json")
        if os.path.exists(input_path):
            output_path = os.path.join(
                output_dir, qu, "clocq_bm25", sources_string, "train_ers.jsonl"
            )
            ers.inference_on_data_split(input_path, output_path, sources)

        input_path = os.path.join(input_dir, qu, "dev_qu.json")
        if os.path.exists(input_path):
            output_path = os.path.join(
                output_dir, qu, "clocq_bm25", sources_string, "dev_ers.jsonl"
            )
            ers.inference_on_data_split(input_path, output_path, sources)

        input_path = os.path.join(input_dir, qu, "test_qu.json")
        output_path = os.path.join(output_dir, qu, "clocq_bm25", sources_string, "test_ers.jsonl")
        ers.inference_on_data_split(input_path, output_path, sources)

    # store results in cache
    ers.store_cache()
