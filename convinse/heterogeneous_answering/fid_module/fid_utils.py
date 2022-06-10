import os
import sys
import json
import random
import logging

from pathlib import Path
from Levenshtein import distance as levenshtein_distance

from convinse.library.utils import get_config
from convinse.evaluation import evidence_has_answer, question_is_existential


def prepare_turn(config, input_turn, output_path, train=False):
    """
    Prepare the given turn for input into FiD.
    Input will be top-100 evidences per question
    as predicted by ERS stage.
    Writes the result in the given output path.
    """
    # create output dir
    output_dir = os.path.dirname(output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # prepare
    res = _prepare_turn(config, input_turn, train)
    if res is None:
        sr = input_turn["structured_representation"]
        raise Exception(f"No evidences found for this turn! SR: {sr}.")
    
    # store
    with open(output_path, "w") as fp:
        fp.write(json.dumps(res))
        fp.write("\n")


def prepare_data(config, input_turns, output_path, train=False):
    """
    Prepare the given data for input into FiD.
    Input will be top-100 evidences per question
    as predicted by ERS stage.
    """
    # create output dir
    output_dir = os.path.dirname(output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # process data
    with open(output_path, "w") as fp_out:
        # transform
        for turn in input_turns:
            # skip instances that are already processed
            if not turn.get("pred_answers") is None:
                continue

            res = _prepare_turn(config, turn, train)
            # skip turns for which no evidences were found
            if res is None:
                continue

            # write new
            fp_out.write(json.dumps(res))
            fp_out.write("\n")


def _prepare_turn(config, input_turn, train):
    """
    Prepare the given turn for input into FiD.
    Input will be top-100 evidences per question
    as predicted by ERS stage.
    Returns the object. For internal usage!
    """
    # construct set of answers that are present (from silver evidences)
    answer_ids = [answer["id"] for answer in input_turn["answers"]]

    # prepare target answers
    target_answers = set()
    # retrieve target answers from answering evidences -> preserve order!
    evidences = input_turn["top_evidences"]
    for evidence in evidences:
        if evidence_has_answer(evidence, input_turn["answers"]):
            for disambiguation in evidence["disambiguations"]:
                if disambiguation[1] in answer_ids:
                    target_answers.add(disambiguation[0])

    # if no answer can be found, skip instance during train/dev!
    if train and not target_answers:
        return None

    evidences = input_turn["top_evidences"]
    evidences = evidences[: config["fid_max_evidences"]]

    # create data
    answers = list(target_answers) + [answer["label"] for answer in input_turn["answers"]]
    target_answer = answers[0]  # always first element of target_answers
    evidences = [
        {"title": evidence["retrieved_for_entity"]["label"], "text": evidence["evidence_text"]}
        for evidence in input_turn["top_evidences"]
    ]

    # if there are no evidences, return None (=skip instance)
    if evidences == []:
        return None

    # return transformed instance
    return {
        "id": input_turn["question_id"],
        "question": input_turn["structured_representation"],
        "target": target_answer,
        "answers": answers,
        "ctxs": evidences,
    }


def get_ranked_answers(config, generated_answer, turn, convquestions=False):
    """
    Convert the predicted answer to a Wikidata ID (or Yes/No),
    and return the ranked answers.
    """
    # check if existential (special treatment)
    question = turn["question"]
    if question_is_existential(question):
        ranked_answers = [
            {"answer": {"id": "yes", "label": "yes"}, "score": 1.0, "rank": 1},
            {"answer": {"id": "no", "label": "no"}, "score": 0.5, "rank": 2},
        ]
    # no existential
    else:
        # return dummy answer in case None was found (if no evidences found)
        if generated_answer is None:
            return [{"answer": {"id": "None", "label": "None"}, "rank": 1, "score": 0.0}]
        smallest_diff = 100000
        ranked_answers = list()
        mentions = set()
        for evidence in turn["top_evidences"]:
            for disambiguation in evidence["disambiguations"]:
                mention = disambiguation[0]
                id = disambiguation[1]
                if id is None or id == False:
                    continue

                # skip duplicates
                ans = str(mention) + str(id)
                if ans in mentions:
                    continue
                mentions.add(ans)
                # exact match
                if generated_answer == mention:
                    diff = 0
                # otherwise compute edit distance
                else:
                    diff = levenshtein_distance(generated_answer, mention)

                # ranked answers different for convquestions
                if convquestions and diff <= smallest_diff:
                    # check if new single best answer
                    if diff < smallest_diff:
                        smallest_diff = diff
                        # add 1 to all ranks
                        if ranked_answers:
                            for answer in ranked_answers:
                                answer["rank"] += 1
                        # put new top ranked upfront
                        ranked_answers = [
                            {"answer": {"id": id, "label": mention}, "rank": 1, "score": diff}
                        ] + ranked_answers
                    # check if new best answer (among other) -> resolve ranking
                    elif diff == smallest_diff:
                        ranked_answers.append(
                            {"answer": {"id": id, "label": mention}, "rank": 1, "score": diff}
                        )
                        for answer in ranked_answers:
                            if answer["rank"] > 1:
                                answer["rank"] += 1
                # ranked answers for standard dataset
                elif diff <= smallest_diff:
                    # check if new single best answer
                    if diff < smallest_diff:
                        smallest_diff = diff
                        ranked_answers = [
                            {"answer": {"id": id, "label": mention}, "rank": 1, "score": diff}
                        ]
                    # check if new best answer (among other) -> resolve ranking
                    elif diff == smallest_diff:
                        ranked_answers.append(
                            {
                                "answer": {"id": id, "label": mention},
                                "rank": len(ranked_answers) + 1,
                                "score": diff,
                            }
                        )

    # don't return too many answers
    max_answers = config["fid_max_answers"]
    ranked_answers = ranked_answers[:max_answers]
    if not ranked_answers:
        ranked_answers = [{"answer": {"id": "None", "label": "None"}, "rank": 1, "score": 0.0}]
    return ranked_answers
