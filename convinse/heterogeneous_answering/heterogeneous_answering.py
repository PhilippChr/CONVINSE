import os
import json

from convinse.library.utils import store_json_with_mkdir, get_logger


class HeterogeneousAnswering:
    def __init__(self, config):
        """Initialize HA module."""
        self.config = config
        self.logger = get_logger(__name__, config)

    def train(self, sources=["kb", "text", "table", "info"]):
        """ Method used in case no training required for HA phase. """
        self.logger.info("Module used does not require training.")

    def inference(self):
        """Run HA on data and add answers for each source combination."""
        input_dir = self.config["path_to_annotated"]
        output_dir = self.config["path_to_intermediate_results"]

        qu = self.config["qu"]
        ers = self.config["ers"]
        ha = self.config["ha"]

        source_combinations = self.config["source_combinations"]
        for sources in source_combinations:
            sources_string = "_".join(sources)

            input_path = os.path.join(input_dir, qu, ers, sources_string, "test_ers.jsonl")
            output_path = os.path.join(input_dir, qu, ers, sources_string, ha, "test_ha.json")
            self.inference_on_data_split(input_path, output_path, sources)

    def inference_on_data_split(self, input_path, output_path):
        """Run HA on given data split."""
        # open data
        input_turns = list()
        data = list()
        with open(input_path, "r") as fp:
            line = fp.readline()
            while line:
                conversation = json.loads(line)
                input_turns += [turn for turn in conversation["questions"]]
                data.append(conversation)
                line = fp.readline()

        # inference
        self.inference_on_turns(input_turns)

        # store processed data
        store_json_with_mkdir(data, output_path)

    def inference_on_data(self, input_data):
        """Run HA on given data."""
        input_turns = [turn for conv in input_data for turn in conv["questions"]]
        self.inference_on_turns(input_turns)
        return input_data

    def inference_on_turns(self, input_turns):
        """Run HA on a set of turns."""
        for turn in turns:
            self.inference_on_turn(turn)

    def inference_on_turn(self, turn):
        raise Exception(
            "This is an abstract function which should be overwritten in a derived class!"
        )
