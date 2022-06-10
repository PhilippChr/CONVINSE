import json

from tqdm import tqdm

from convinse.library.string_library import StringLibrary


def evaluate_gnn(top_candidates, gold_answers, gold_answers_in_graph, question):
    """
    Evaluate the outputs by the GNN.
    """
    # get ranking
    if question_is_existential(question):
        answer_presence = True
        ranked_answers = [
            {"answer": {"id": "yes", "label": "yes"}, "score": 1.0, "rank": 1},
            {"answer": {"id": "no", "label": "no"}, "score": 0.5, "rank": 2},
        ]
    else:
        answer_presence = len(gold_answers_in_graph) > 0
        # filter out masked entities
        top_candidates = [
            candidate for candidate in top_candidates if type(candidate["entity"]) == dict
        ]
        # create answer dicts
        ranked_answers = [
            {
                "answer": answer["entity"],
                "answer_label": answer["entity"]["label"],
                "score": answer["score"],
                "rank": (i + 1),
            }
            for i, answer in enumerate(top_candidates)
        ]
    # compute scores
    p_at_1 = precision_at_1(ranked_answers, gold_answers)
    h_at_5 = hit_at_5(ranked_answers, gold_answers)
    mrr = mrr_score(ranked_answers, gold_answers)
    # create result
    res = {
        "precision_at_1": p_at_1,
        "hit_at_5": h_at_5,
        "mrr_score": mrr,
        "answer_presence": 1 if answer_presence else 0,
    }
    return res


def answer_presence(evidences, answers):
    """
    Compute the answer presence for a set of evidences
    and a parsed answer dict, and return a list of
    answering evidences.
    Return format: (boolean, [evidence-dict, ...])
    """
    # initialize
    answer_present = False
    answering_evidences = list()

    # go through evidences
    for evidence in evidences:
        if evidence_has_answer(evidence, answers):
            # remember evidence
            answer_present = True
            answering_evidences.append(evidence)
    # return results
    return (answer_present, answering_evidences)


def evidence_has_answer(evidence, gold_answers):
    """Check whether the given evidence has any of the answers."""
    for answer_candidate in evidence["wikidata_entities"]:
        # check if answering candidate
        if candidate_in_answers(answer_candidate, gold_answers):
            return True
    return False


def candidate_in_answers(answer_candidate, gold_answers):
    """Check if candidate is answer."""
    # get ids
    answer_candidate_id = answer_candidate["id"]
    gold_answer_ids = [answer["id"] for answer in gold_answers]

    # normalize
    answer_candidate_id = answer_candidate_id.lower().strip().replace('"', "").replace("+", "")
    gold_answer_ids = [answer.lower().strip().replace('"', "") for answer in gold_answer_ids]

    # perform check
    if answer_candidate_id in gold_answer_ids:
        return True

    # no match found
    return False


def mrr_score(answers, gold_answers):
    """Compute MRR score for given answers and gold answers."""
    # check if any answer was given
    if not answers:
        return 0.0
    # go through answer candidates
    for answer in answers:
        if candidate_in_answers(answer["answer"], gold_answers):
            return 1.0 / float(answer["rank"])
    return 0.0


def precision_at_1(answers, gold_answers):
    """Compute P@1 score for given answers and gold answers."""
    # check if any answer was given
    if not answers:
        return 0.0
    # go through answer candidates
    for answer in answers:
        if float(answer["rank"]) > float(1.0):
            break
        elif candidate_in_answers(answer["answer"], gold_answers):
            return 1.0
    return 0.0


def hit_at_5(answers, gold_answers):
    """Compute Hit@5 score for given answers and gold answers."""
    # check if any answer was given
    if not answers:
        return 0.0
    # go through answer candidates
    for answer in answers:
        if float(answer["rank"]) > float(5.0):
            break
        elif candidate_in_answers(answer["answer"], gold_answers):
            return 1.0
    return 0.0


def question_is_existential(question):
    existential_keywords = [
        "is",
        "are",
        "was",
        "were",
        "am",
        "be",
        "being",
        "been",
        "did",
        "do",
        "does",
        "done",
        "doing",
        "has",
        "have",
        "had",
        "having",
    ]
    lowercase_question = question.lower()
    lowercase_question = lowercase_question.strip()
    for keyword in existential_keywords:
        if lowercase_question.startswith(keyword):
            return True
    return False
