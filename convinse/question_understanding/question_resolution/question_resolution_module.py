import os
import sys
import json
import logging

from subprocess import Popen, PIPE

from convinse.library.utils import get_config, get_logger, store_json_with_mkdir
from convinse.question_understanding.question_understanding import QuestionUnderstanding
import convinse.question_understanding.question_resolution.question_resolution_utils as qres_utils


class QuestionResolutionModule(QuestionUnderstanding):
    def __init__(self, config, use_gold_answers):
        """Initialize SR module."""
        self.config = config
        self.logger = get_logger(__name__, config)
        self.use_gold_answers = use_gold_answers
        self.path_to_quretec = "convinse/question_understanding/question_resolution/quretec"
        self.model_id = config["qres_model_id"]
        self.model_dir = config["qres_model_dir"]

    def train(self):
        """Train the model on silver SR data."""
        # train model
        self.logger.info(f"Starting training...")
        input_dir = self.config["path_to_annotated"]
        train_path = os.path.join(input_dir, "annotated_train.json")
        dev_path = os.path.join(input_dir, "annotated_dev.json")
        qres_utils.prepare_data_for_training(self.config, train_path, dev_path)
        benchmark = self.config["benchmark"]
        data_dir = f"_intermediate_representations/{benchmark}/qres/data"

        # run training
        COMMAND = ["python", f"{self.path_to_quretec}/run_ner.py"]
        COMMAND += ["--task_name", "ner"]
        COMMAND += ["--bert_model", "bert-large-uncased"]
        COMMAND += ["--max_seq_length", "300"]
        COMMAND += ["--train_batch_size", "20"]
        COMMAND += ["--train_on", "train"]
        COMMAND += ["--hidden_dropout_prob", "0.4"]
        COMMAND += ["--dev_on", "dev"]
        COMMAND += ["--do_train"]
        COMMAND += ["--data_dir", data_dir]
        COMMAND += ["--base_dir", self.model_dir]
        COMMAND += ["--model_id", self.model_id]
        process = Popen(COMMAND, stdout=sys.stdout, stderr=sys.stderr)
        self.logger.info(f"Finished training.")

    def inference_on_data(self, input_data):
        """Run model on data and add predictions."""
        benchmark = self.config["benchmark"]
        data_dir = f"_intermediate_representations/{benchmark}/qres/data"
        output_path = os.path.join(data_dir, "data_for_inference.json")

        # model inference on given data
        qres_utils.prepare_data_for_inference(
            self.config, input_data, output_path, use_gold_answers=self.use_gold_answers
        )
        self._inference()

        # postprocess predictions
        quretec_pred_path = os.path.join(
            self.model_dir, self.model_id, "eval_results_data_for_inference_epoch0.json"
        )
        qres_utils.postprocess_data(input_data, quretec_pred_path)
        return input_data

    def inference_on_turn(self, turn, history_turns):
        """Run inference on a single turn (and history)."""
        if not history_turns:
            turn["structured_representation"] = turn["question"]
            return turn

        benchmark = self.config["benchmark"]
        data_dir = f"_intermediate_representations/{benchmark}/qres/data"
        output_path = os.path.join(data_dir, "data_for_inference.json")
        
        # model inference on given data
        qres_utils.prepare_turn_for_inference(self.config, turn, history_turns, output_path, self.use_gold_answers)
        self._inference()

        # postprocess predictions
        quretec_pred_path = os.path.join(
            self.model_dir, self.model_id, "eval_results_data_for_inference_epoch0.json"
        )
        qres_utils.postprocess_turn(turn, quretec_pred_path)
        return turn

    def _inference(self):
        """Run QuReTeC model on given input via separate script."""
        benchmark = self.config["benchmark"]
        data_dir = f"_intermediate_representations/{benchmark}/qres/data"

        # run inference
        COMMAND = ["python", f"{self.path_to_quretec}/run_ner.py"]
        COMMAND += ["--task_name", "ner"]
        COMMAND += ["--do_eval"]
        COMMAND += ["--do_lower_case"]
        COMMAND += ["--data_dir", data_dir]
        COMMAND += ["--base_dir", self.model_dir]
        COMMAND += ["--dev_on", "data_for_inference"]
        COMMAND += ["--model_id", self.model_id]
        COMMAND += ["--no_cuda"]
        process = Popen(COMMAND, stdout=sys.stdout, stderr=sys.stderr)
        process.communicate()


#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception(
            "Usage: python convinse/question_understanding/question_resolution/question_resolution_module.py --<FUNCTION> <PATH_TO_CONFIG>"
        )

    function = sys.argv[1]
    config_path = sys.argv[2]
    config = get_config(config_path)

    # train: train model
    if function == "--train":
        qrm = QuestionResolutionModule(config, use_gold_answers=True)
        qrm.train()

    # inference: add predictions to data
    elif function == "--inference":
        # load config
        qrm = QuestionResolutionModule(config, use_gold_answers=True)
        qrm.inference()
