import os
import sys
import json
import time
import torch
import random

from subprocess import Popen, PIPE

from convinse.library.utils import get_config, store_json_with_mkdir
import convinse.heterogeneous_answering.fid_module.fid_utils as fid_utils
from convinse.heterogeneous_answering.heterogeneous_answering import HeterogeneousAnswering
import convinse.evaluation as evaluation


class FiDModule(HeterogeneousAnswering):
    def __init__(self, config):
        """Initialize the FiD module."""
        self.config = config
        self.path_to_fid = "convinse/heterogeneous_answering/fid_module/FiD"
        self._initialize_conda_dir()

    def train(self, sources=["kb", "text", "table", "info"]):
        """ Train the FiD model on the dataset. """
        # set paths
        sources_string = "_".join(sources)
        input_dir = self.config["path_to_intermediate_results"]
        qu = self.config["qu"]
        ers = self.config["ers"]
        train_path = os.path.join(input_dir, qu, ers, sources_string, "train_ers.jsonl")
        dev_path = os.path.join(input_dir, qu, ers, sources_string, "dev_ers.jsonl")

        # load train data
        with open(train_path, "r") as fp:
            train_data = list()
            line = fp.readline()
            while line:
                train_data.append(json.loads(line))
                line = fp.readline()
            train_input_turns = [turn for conv in train_data for turn in conv["questions"]]
        # load dev data
        with open(dev_path, "r") as fp:
            dev_data = list()
            line = fp.readline()
            while line:
                dev_data.append(json.loads(line))
                line = fp.readline()
            dev_input_turns = [turn for conv in dev_data for turn in conv["questions"]]

        # prepare paths
        prepared_train_path, _ = self._prepare_paths()
        prepared_dev_path, _ = self._prepare_paths()

        # prepare data
        fid_utils.prepare_data(self.config, train_input_turns, prepared_train_path, train=True)
        fid_utils.prepare_data(self.config, dev_input_turns, prepared_dev_path, train=False)

        # free up memory
        del dev_input_turns
        del train_input_turns
        del dev_data
        del train_data

        # train
        self._train(prepared_train_path, prepared_dev_path, sources_string)

    def inference_on_turns(self, input_turns):
        """Run HA on given turns."""
        # paths
        prepared_input_path, res_name = self._prepare_paths()

        # prepare data
        fid_utils.prepare_data(self.config, input_turns, prepared_input_path)

        # inference
        self._inference(res_name, prepared_input_path)

        # parse result
        path_to_result = f"{self.path_to_fid}/tmp_output_data/{res_name}/final_output.txt"
        generated_answers = self._parse_result(path_to_result)

        # add predicted answers to turns
        for turn in input_turns:
            self._postprocess_turn(turn, generated_answers)
        return input_turns

    def inference_on_turn(self, turn):
        """Run HA on a single turn."""
        # paths
        prepared_input_path, res_name = self._prepare_paths()

        # prepare data
        fid_utils.prepare_turn(self.config, turn, prepared_input_path, train=False)

        # inference
        self._inference(res_name, prepared_input_path)

        # parse result
        path_to_result = f"{self.path_to_fid}/tmp_output_data/{res_name}/final_output.txt"
        generated_answers = self._parse_result(path_to_result)

        # add predicted answers to turns
        self._postprocess_turn(turn, generated_answers)
        return turn

    def _train(self, prepared_train_path, prepared_dev_path, sources_string):
        benchmark = self.config["benchmark"]
        method_name = self.config["name"]
        qu = self.config["qu"]
        name = f"{method_name}_{sources_string}"
        COMMAND = [self.path_to_fid_python_env, f"{self.path_to_fid}/train_reader.py"]
        COMMAND += ["--name", name]
        COMMAND += ["--checkpoint_dir", f"_data/{benchmark}/{method_name}/fid"]
        COMMAND += ["--train_data", prepared_train_path]
        COMMAND += ["--eval_data", prepared_dev_path]
        COMMAND += ["--model_size", "base"]
        COMMAND += ["--lr", str(self.config["fid_lr"])]
        COMMAND += ["--optim", str(self.config["fid_optim"])]
        COMMAND += ["--scheduler", str(self.config["fid_scheduler"])]
        COMMAND += ["--weight_decay", str(self.config["fid_weight_decay"])]
        COMMAND += ["--text_maxlength", str(self.config["fid_text_maxlength"])]
        COMMAND += ["--answer_maxlength", str(self.config["fid_answer_maxlength"])]
        COMMAND += ["--per_gpu_batch_size", str(self.config["fid_per_gpu_batch_size"])]
        COMMAND += ["--n_context", str(self.config["fid_max_evidences"])]
        COMMAND += ["--total_step", str(self.config["fid_total_step"])]
        COMMAND += ["--warmup_step", str(self.config["fid_warmup_step"])]
        process = Popen(COMMAND, stdout=sys.stdout, stderr=sys.stderr)
        process.communicate()

    def _inference(self, res_name, prepared_input_path):
        """Run inference on a given question (or SR), and a set of evidences."""
        COMMAND = [self.path_to_fid_python_env, f"{self.path_to_fid}/test_reader.py"]
        COMMAND += ["--name", res_name]
        COMMAND += ["--model_path", self.config["fid_model_path"]]
        COMMAND += ["--checkpoint_dir", f"{self.path_to_fid}/tmp_output_data"]
        COMMAND += ["--eval_data", prepared_input_path]
        COMMAND += ["--n_context", str(self.config["fid_max_evidences"])]
        COMMAND += ["--per_gpu_batch_size", str(self.config["fid_per_gpu_batch_size"])]
        COMMAND += ["--write_results"]
        process = Popen(COMMAND, stdout=sys.stdout, stderr=sys.stderr)
        process.communicate()

    def _postprocess_turn(self, turn, generated_answers):
        ques_id = turn["question_id"]
        generated_answer = generated_answers.get(ques_id)
        turn["generated_answer"] = generated_answer

        # get ranked answers
        ranked_answers = evaluation.get_ranked_answers(
            self.config, generated_answer, turn
        )
        try:
            turn["pred_answers"] = [
                {
                    "id": ans["answer"]["id"],
                    "label": ans["answer"]["label"],
                    "rank": ans["rank"],
                    "score": ans["score"],
                }
                for ans in ranked_answers
            ]
        except:
            print(f"Fail with: {ranked_answers}")
            turn["pred_answers"] = [
                {"id": ans["answer"]["id"], "label": ans["answer"]["label"], "rank": ans["rank"]}
                for ans in ranked_answers
            ]
        # eval
        p_at_1 = evaluation.precision_at_1(ranked_answers, turn["answers"])
        turn["p_at_1"] = p_at_1
        mrr = evaluation.mrr_score(ranked_answers, turn["answers"])
        turn["mrr"] = mrr
        h_at_5 = evaluation.hit_at_5(ranked_answers, turn["answers"])
        turn["h_at_5"] = h_at_5

        # delete noise
        if turn.get("top_evidences"):
            del turn["top_evidences"]
        if turn.get("question_entities"):
            del turn["question_entities"]
        if turn.get("silver_SR"):
            del turn["silver_SR"]
        if turn.get("silver_relevant_turns"):
            del turn["silver_relevant_turns"]
        if turn.get("silver_answering_evidences"):
            del turn["silver_answering_evidences"]

    def _parse_result(self, path_to_result):
        """
        Parse the output generated by FiD, and add predicted
        (and generated) answers to the data.
        """
        # get answers from output file
        generated_answers = dict()
        with open(path_to_result, "r") as fp:
            line = fp.readline()
            while line:
                try:
                    ques_id, answer = line.split(None, 1)
                except:
                    ques_id = line.strip()
                    answer = ""
                ques_id = ques_id.strip()
                answer = answer.strip()
                generated_answers[ques_id] = answer
                line = fp.readline()
        return generated_answers

    def _prepare_paths(self):
        """ Prepare random path for handling input/output with piped FiD process. """
        random_num = str(random.randint(0, 10000))
        prepared_input_path = f"{self.path_to_fid}/tmp_input_data/data_{random_num}.jsonl"
        res_name = f"output_{random_num}"
        return prepared_input_path, res_name

    def _initialize_conda_dir(self):
        """ Code to automatically detect and set the path to the FiD environment."""
        conda_dir = os.environ.get("CONDA_PREFIX", None)
        if not conda_dir:
            raise Exception("Something went wrong! Tried accessing the value of the CONDA_PREFIX variable, but failed. Please make sure that you have a valid conda installation.")
        
        # in case some environment is activated (which should be `convinse`), move one dir upwards
        if "envs" in conda_dir:
            conda_dir = os.path.dirname(conda_dir)
        else:
            conda_dir = os.path.join(conda_dir, "envs")
        self.path_to_fid_python_env = os.path.join(conda_dir, "fid", "bin", "python")


#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception(
            "python convinse/heterogeneous_answering/fid_module/fid_module.py --<FUNCTION> <PATH_TO_CONFIG>"
        )

    function = sys.argv[1]
    config_path = sys.argv[2]
    config = get_config(config_path)

    if function == "--train":
        # set paths
        qu = config["qu"]
        ers = config["ers"]
        input_dir = config["path_to_intermediate_results"]
        data_sources_str = config["fid_train_sources"]
        path = os.path.join(input_dir, qu, ers, data_sources_str)
        train_path = os.path.join(path, "train_ers.jsonl")
        dev_path = os.path.join(path, "dev_ers.jsonl")

        # train
        fid = FiDModule(config)
        fid.train(train_path, dev_path)

    elif function == "--example":
        # set paths
        qu = config["qu"]
        ers = config["ers"]
        input_dir = config["path_to_intermediate_results"]
        data_sources_str = "kb_text_table_info"
        path = os.path.join(input_dir, qu, ers, data_sources_str)
        input_path = os.path.join(path, "dev_ers.jsonl")

        with open(input_path, "r") as fp:
            line = fp.readline()
            conv = json.loads(line)
        turn = conv["questions"][0]

        # run inference on example
        fid = FiDModule(config)

        start = time.time()
        res = fid.inference_on_turn(turn)
        print(res)
        print(f"Spent {time.time()-start} seconds!")
