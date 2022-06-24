import os
import sys
import logging

from convinse.library.utils import get_config, get_logger
from convinse.question_understanding.question_understanding import QuestionUnderstanding
from convinse.question_understanding.question_rewriting.question_rewriting_model import (
    QuestionRewritingModel,
)
import convinse.question_understanding.question_rewriting.dataset_question_rewriting as dataset


class QuestionRewritingModule(QuestionUnderstanding):
    def __init__(self, config, use_gold_answers):
        """Initialize QR module."""
        self.config = config
        self.logger = get_logger(__name__, config)
        self.use_gold_answers = use_gold_answers
        
        # create model
        self.qr_model = QuestionRewritingModel(config)
        self.model_loaded = False

        self.history_separator = config["history_separator"]

    def train(self):
        """Train the model on silver AR data."""
        # create paths
        self.logger.info(f"Starting training...")
        data_dir = self.config["path_to_annotated"]
        train_path = os.path.join(data_dir, "annotated_train.json")
        dev_path = os.path.join(data_dir, "annotated_dev.json")
        self.qr_model.train(train_path, dev_path)
        self.logger.info(f"Finished training.")

    def inference_on_conversation(self, conversation):
        """Run inference on a single conversation."""
        # load QR model (if required)
        self._load()

        # QR model inference
        history_turns = list()
        for i, turn in enumerate(conversation["questions"]):
            # append to history
            question = turn["question"]
            history_turns.append(question)

            # prepare input (omitt gold answer(s))
            rewrite_input = self.history_separator.join(history_turns)

            # run inference
            qrew = self.qr_model.inference(rewrite_input)
            turn["structured_representation"] = qrew

            # only append answer if there is a next question
            if i + 1 < len(conversation["questions"]):
                if self.use_gold_answers:
                    answer_text = " ".join([answer["label"] for answer in turn["answers"]])
                else:
                    # answer_text = ", ".join([answer["label"] for answer in turn["pred_answers"]])
                    answer_text = turn["pred_answers"][0]["label"]
                history_turns.append(answer_text)
        return conversation

    def inference_on_turn(self, turn, history_turns):
        """Run inference on a single turn (and history)."""
        # load QR model (if required)
        self._load()

        # SR model inference
        question = turn["question"]
        history_turns.append(question)

        # prepare input (omitt gold answer(s))
        rewrite_input = self.history_separator.join(history_turns)

        # run inference
        intent_explicit = self.qr_model.inference(rewrite_input)
        turn["structured_representation"] = intent_explicit
        return turn

    def _load(self):
        """Load the QRes model."""
        # only load if not already done so
        if not self.model_loaded:
            self.qr_model.load()
            self.qr_model.set_eval_mode()
            self.model_loaded = True


#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception(
            "Invalid number of options provided.\nUsage: python convinse/question_understanding/question_rewriting/question_rewriting_module.py <PATH_TO_CONFIG>"
        )

    function = sys.argv[1]
    config_path = sys.argv[2]
    config = get_config(config_path)

    # train: train model
    if function == "--train":
        qrm = QuestionRewritingModule(config, use_gold_answers=True)
        qrm.train()

    # inference: add predictions to data
    elif function == "--inference":
        # load config
        qrm = QuestionRewritingModule(config, use_gold_answers=True)
        qrm.inference()
