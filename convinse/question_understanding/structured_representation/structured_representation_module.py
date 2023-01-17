import os
import sys
import torch
import logging
import random

from convinse.library.utils import get_config, get_logger
from convinse.question_understanding.question_understanding import QuestionUnderstanding
from convinse.question_understanding.structured_representation.structured_representation_model import (
    StructuredRepresentationModel,
)
import convinse.question_understanding.structured_representation.dataset_structured_representation as dataset


class StructuredRepresentationModule(QuestionUnderstanding):
    def __init__(self, config, use_gold_answers):
        """Initialize SR module."""
        self.config = config
        self.logger = get_logger(__name__, config)
        self.use_gold_answers = use_gold_answers
        
        # create model
        self.sr_model = StructuredRepresentationModel(config)
        self.model_loaded = False

        self.history_separator = config["history_separator"]
        self.sr_delimiter = config["sr_delimiter"]

    def train(self):
        """Train the model on silver SR data."""
        # train model
        self.logger.info(f"Starting training...")
        data_dir = self.config["path_to_annotated"]
        train_path = os.path.join(data_dir, "annotated_train.json")
        dev_path = os.path.join(data_dir, "annotated_dev.json")
        self.sr_model.train(train_path, dev_path)
        self.logger.info(f"Finished training.")

    def inference_on_conversation(self, conversation):
        """Run inference on a single conversation."""
        # load SR model (if required)
        self._load()

        with torch.no_grad():
            # SR model inference
            history_turns = list()
            for i, turn in enumerate(conversation["questions"]):
                self.inference_on_turn(turn, history_turns)

                # only append answer if there is a next question
                if i + 1 < len(conversation["questions"]):
                    if self.use_gold_answers:
                        answer_text = ", ".join([answer["label"] for answer in turn["answers"]])
                    else:
                        # answer_text = ", ".join([answer["label"] for answer in turn["pred_answers"]])
                        answer_text = turn["pred_answers"][0]["label"]
                    history_turns.append(answer_text)
            return conversation

    def inference_on_turn(self, turn, history_turns):
        """Run inference on a single turn."""
        # load SR model (if required)
        self._load()

        with torch.no_grad():
            # SR model inference
            question = turn["question"]
            history_turns.append(question)

            # prepare input (omitt gold answer(s))
            rewrite_input = self.history_separator.join(history_turns)

            # run inference
            sr = self.sr_model.inference(rewrite_input)
            turn["structured_representation"] = sr
            return turn

    def _load(self):
        """Load the SR model."""
        # only load if not already done so
        if not self.model_loaded:
            self.sr_model.load()
            self.sr_model.set_eval_mode()
            self.model_loaded = True

    def adjust_sr_for_ablation(self, sr, ablation_type):
        """
        Adjust the given SR based on the specific ablation type.
        """
        slots = sr.split(self.sr_delimiter, 3)
        if len(slots) < 4 and not slots[0]:
            # type missing
            slots = slots + [""]
        elif len(slots) < 4:
            # context missing
            slots = [""] + slots
        if len(slots) < 4:
            # fix other (strange) cases
            slots = slots + (4 - len(slots)) * [""]
        context, entity, pred, ans_type = slots
        if ablation_type == "nocontext":
            sr = f"{entity.strip()} {self.sr_delimiter} {pred.strip()} {self.sr_delimiter} {ans_type.strip()}"
        elif ablation_type == "noentity":
            sr = f"{context.strip()} {self.sr_delimiter} {pred.strip()} {self.sr_delimiter} {ans_type.strip()}"
        elif ablation_type == "nopred":
            sr = f"{context.strip()} {self.sr_delimiter} {entity.strip()} {self.sr_delimiter} {ans_type.strip()}"
        elif ablation_type == "notype":
            sr = f"{context.strip()} {self.sr_delimiter} {entity.strip()} {self.sr_delimiter} {pred.strip()}"
        elif ablation_type == "nostructure":
            slots = [context, entity, pred, ans_type]
            random.shuffle(slots)
            sr = f"{slots[0].strip()} {self.sr_delimiter} {slots[1].strip()} {self.sr_delimiter} {slots[2].strip()} {self.sr_delimiter} {slots[3].strip()}"
        elif ablation_type == "full":
            sr = f"{context.strip()} {self.sr_delimiter} {entity.strip()} {self.sr_delimiter} {pred.strip()} {self.sr_delimiter} {ans_type.strip()}"
        else:
            raise Exception(f"Unknown ablation type: {ablation_type}")
        return sr


#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception(
            "Usage: python convinse/question_understanding/structured_representation/structured_representation_module.py --<FUNCTION> <PATH_TO_CONFIG>"
        )

    function = sys.argv[1]
    config_path = sys.argv[2]
    config = get_config(config_path)

    # train: train model
    if function == "--train":
        srm = StructuredRepresentationModule(config, use_gold_answers=True)
        srm.train()

    # inference: add predictions to data
    elif function == "--inference":
        # load config
        srm = StructuredRepresentationModule(config, use_gold_answers=True)
        srm.inference()
