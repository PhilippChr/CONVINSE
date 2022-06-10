import os
import json

from pathlib import Path
from tqdm import tqdm

from convinse.library.utils import get_config, get_logger
from convinse.evaluation import answer_presence


class EvidenceRetrievalScoring:
    """Abstract class for ERs phase."""

    def __init__(self, config):
        """Initialize ERS module."""
        self.config = config
        self.logger = get_logger(__name__, config)

    def train(self, sources=None):
        """Method used in case no training required for ERS phase."""
        self.logger.info("Module used does not require training.")

    def inference(self, sources=None):
        """Run ERS on data and add retrieve top-e evidences for each source combination."""
        input_dir = self.config["path_to_annotated"]
        output_dir = self.config["path_to_intermediate_results"]

        qu = self.config["qu"]
        ers = self.config["ers"]

        # either use given option, or from config
        if not sources is None:
            source_combinations = [sources]
        else:
            source_combinations = self.config["source_combinations"]

        # go through all combinations
        for sources in source_combinations:
            sources_string = "_".join(sources)

            input_path = os.path.join(input_dir, qu, "train_qu.json")
            output_path = os.path.join(output_dir, qu, ers, sources_string, "train_ers.jsonl")
            self.inference_on_data_split(input_path, output_path, sources)

            input_path = os.path.join(input_dir, qu, "dev_qu.json")
            output_path = os.path.join(output_dir, qu, ers, sources_string, "dev_ers.jsonl")
            self.inference_on_data_split(input_path, output_path, sources)

            input_path = os.path.join(input_dir, qu, "test_qu.json")
            output_path = os.path.join(output_dir, qu, ers, sources_string, "test_ers.jsonl")
            self.inference_on_data_split(input_path, output_path, sources)

        # store results in cache (if applicable)
        self.store_cache()

    def inference_on_data_split(self, input_path, output_path, sources):
        """
        Run ERS on the dataset to predict
        answering evidences for each SR in the dataset.
        """
        # open data
        with open(input_path, "r") as fp:
            data = json.load(fp)
        self.logger.info(f"Input data loaded from: {input_path}.")

        # score
        answer_presences = list()
        source_to_ans_pres = {"kb": 0, "text": 0, "table": 0, "info": 0, "all": 0}

        # create folder if not exists
        output_dir = os.path.dirname(output_path)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # process data
        with open(output_path, "w") as fp:
            for conversation in tqdm(data):
                for turn in conversation["questions"]:
                    top_evidences = self.inference_on_turn(turn, sources)
                    turn["top_evidences"] = top_evidences

                    # answer presence
                    hit, answering_evidences = answer_presence(top_evidences, turn["answers"])
                    turn["answer_presence"] = hit
                    turn["answer_presence_per_src"] = {
                        evidence["source"]: 1 for evidence in answering_evidences
                    }

                # write conversation to file
                fp.write(json.dumps(conversation))
                fp.write("\n")

                # accumulate results
                c_answer_presences = [turn["answer_presence"] for turn in conversation["questions"]]
                answer_presences += c_answer_presences
                for turn in conversation["questions"]:
                    answer_presence_per_src = turn["answer_presence_per_src"]
                    # add per source answer presence
                    for src, ans_presence in answer_presence_per_src.items():
                        source_to_ans_pres[src] += ans_presence
                    # aggregate overall answer presence for validation
                    if len(answer_presence_per_src.items()):
                        source_to_ans_pres["all"] += 1

        # print results
        res_path = output_path.replace(".jsonl", ".res")
        with open(res_path, "w") as fp:
            avg_answer_presence = sum(answer_presences) / len(answer_presences)
            fp.write(f"Avg. answer presence: {avg_answer_presence}\n")
            answer_presence_per_src = {
                src: (num / len(answer_presences)) for src, num in source_to_ans_pres.items()
            }
            fp.write(f"Answer presence per source: {answer_presence_per_src}")

        # log
        self.logger.info(f"Done with processing: {input_path}.")

    def inference_on_data(self, input_data, sources=["kb", "text", "table", "info"]):
        """Run ERS on given data."""
        input_turns = [turn for conv in input_data for turn in conv["questions"]]
        self.inference_on_turns(input_turns, sources)
        return input_data

    def inference_on_turns(self, input_turns, sources=["kb", "text", "table", "info"]):
        """Run ERS on given turns."""
        for turn in input_turns:
            top_evidences = self.inference_on_turn(turn, sources)
            turn["top_evidences"] = top_evidences

            # answer presence
            hit, answering_evidences = answer_presence(top_evidences, turn["answers"])
            turn["answer_presence"] = hit
            turn["answer_presence_per_src"] = {
                evidence["source"]: 1 for evidence in answering_evidences
            }
        return input_turns

    def inference_on_turn(self):
        raise Exception(
            "This is an abstract function which should be overwritten in a derived class!"
        )

    def store_cache(self):
        pass
