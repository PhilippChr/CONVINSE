CONVINSE: Conversational Question Answering on Heterogeneous Sources
============

- [Description](#description)
- [Code](#code)
    - [System requirements](#system-requirements)
	- [Installation](#installation)
	- [Reproduce paper results](#reproduce-paper-results)
	- [Training the pipeline](#training-the-pipeline)
	- [Testing the pipeline](#testing-the-pipeline)
	- [Using the pipeline](#using-the-pipeline)
	- [Comparing on ConvMix](#comparing-on-convmix)
- [Data](#data)
	- [Wikidata](#wikidata)
	- [Wikipedia](#wikipedia)
- [Feedback](#feedback)
- [License](#license)

# Description
This repository contains the code and data for our [SIGIR 2022 paper](https://arxiv.org/abs/2204.11677) on conversational question answering on heterogeneous sources. In this work, we present a general pipeline for the task:
1) Question Understanding (QU) -- creating an explicit representation of an incoming question and its conversational context
2) Evidence Retrieval and Scoring (ERS) -- harnessing this frame-like representation to uniformly capture the most relevant evidences from different sources
3) Heterogeneous Answering (HA) -- deriving the answer from these evidences.

We propose **CONVINSE** as an instantiation of this 3-staged pipeline, and devise structured representations (SR) to capture the user information need explicitly during Question Understanding in this work, as an instantiation of the QU phase.
Experiments are conducted on the newly collected dataset **ConvMix**, which is released on the [CONVINSE website](https://convinse.mpi-inf.mpg.de) (or as download when instantiating this repo).

If you use this code, please cite:
```bibtex
@article{christmann2022conversational,
  title={Conversational Question Answering on Heterogeneous Sources},
  author={Christmann, Philipp and Roy, Rishiraj Saha and Weikum, Gerhard},
  journal={arXiv preprint arXiv:2204.11677},
  year={2022}
}
```

# Code

## System requirements
- Conda (https://docs.conda.io/projects/conda/en/latest/index.html)
- GPU (suggested; never tried training/inference without a GPU)

## Installation
Clone the repo via:
```bash
    git clone https://github.com/PhilippChr/CONVINSE.git
    cd CONVINSE/
    CONVINSE_ROOT=$(pwd)
    conda create --name convinse python=3.8
    conda activate convinse
    pip install -e .
```


### Install dependencies
CONVINSE makes use of [CLOCQ](https://github.com/PhilippChr/CLOCQ) for retrieving relevant evidences, and [FiD](https://github.com/facebookresearch/FiD) for answering.
CLOCQ can be conveniently integrated via the [publicly available API](https://clocq.mpi-inf.mpg.de), using the client from [the repo](https://github.com/PhilippChr/CLOCQ). If efficiency is a primary concern, it is recommended to directly run the CLOCQ code on the local machine (details are given in the repo).  
In either case, install it via:
```bash
    git clone https://github.com/PhilippChr/CLOCQ.git
    cd CLOCQ/
    pip install -e .
```

FiD was built for PyTorch version 1.6.0, and the native code is therefore not compatible with more recent versions of the [Transformers](http://huggingface.co/transformers/) library. Therefore, we provide a wrapper to integrate FiD in the CONVINSE pipeline using [subprocess](https://docs.python.org/3/library/subprocess.html).
You can install it from the repo:
```bash
    cd $CONVINSE_ROOT 
    git clone https://github.com/PhilippChr/FiD.git convinse/heterogeneous_answering/fid_module/FiD
    cd convinse/heterogeneous_answering/fid_module/FiD/
    conda create --name fid python=3.6
    conda activate fid
    pip install -e .
    conda activate convinse
```

Optional: If you want to use or compare with [QuReTeC](https://github.com/nickvosk/sigir2020-query-resolution) for question resolution, you can run the following:
```bash
    cd convinse/question_understanding/question_resolution/
    git clone https://github.com/nickvosk/sigir2020-query-resolution.git quretec
```

Initialize the repo (download data, benchmark, models):
```bash
    cd $CONVINSE_ROOT
    bash scripts/initialize.sh
```

**Compatibility Notes**   
ImportError (version GLIBC_2.29 not found) occured when using latest/default version of tokenizers.  
=> Install tokenizers==0.10.1 via pip


## Reproduce paper results
You can either reproduce all results in Table 6, or only the results for a specific source (combination).

For reproducing all results, run:
``` bash
    bash scripts/pipeline.sh --main-results
```

If you would like to reproduce only the results for a specific source combination, run:
``` bash
    bash scripts/pipeline.sh --gold-answers config/convmix/convinse.yml kb_text_table_info
```
the last parameter (`kb_text_table_info`) specifies the sources to be used.  
Note, that CONVINSE retrieves evidences on-the-fly by default.
Given that the evidences in the information sources can change quickly (e.g. Wikipedia has many updates every day),
results can easily change.
A cache was implemented to improve the reproducability, and we provide a benchmark-related subset of Wikipedia (see details [below](#wikipedia)).

## Training the pipeline

To train a pipeline, just choose the config that represents the pipeline you would like to train, and run:
``` bash
    bash scripts/pipeline.sh --train [<PATH_TO_CONFIG>] [<SOURCES_STR>]
```

Example:
``` bash
    bash scripts/pipeline.sh --train config/convmix/convinse.yml kb_text_table_info
```
The HA part of the pipeline will be trained with the set of information sources you give as parameter.


## Testing the pipeline

If you create your own pipeline, it is recommended to test it once on an example, to verify that everythings runs smoothly.  
You can do that via:
``` bash
    bash scripts/pipeline.sh --example [<PATH_TO_CONFIG>]
```
and see the output file in `out/<benchmark>` for potential errors.

For standard evaluation, you can simply run:
``` bash
    bash scripts/pipeline.sh --gold-answers [<PATH_TO_CONFIG>] [<SOURCES_STR>]
```
Example:
``` bash
    bash scripts/pipeline.sh --gold-answers config/convmix/convinse.yml kb_text_table_info
```
<br/>

For evaluating with all source combinations, e.g. for comparing with CONVINSE, run:
``` bash
    bash scripts/pipeline.sh --main-results [<PATH_TO_CONFIG>] [<SOURCES_STR>]
```

Example:
``` bash
    bash scripts/pipeline.sh --main-results config/convmix/convinse.yml kb_text_table_info
```
<br/>

If you want to evaluate using the predicted answers of previous turns, you can run:
``` bash
    bash scripts/pipeline.sh --pred-answers [<PATH_TO_CONFIG>] [<SOURCES_STR>]
```

Example:
``` bash
    bash scripts/pipeline.sh --pred-answers config/convmix/convinse.yml kb_text_table_info
```
By default, the convinse config and all sources will be used.

## Using the pipeline
For using the pipeline, e.g. for improving individual parts of the pipeline, you can simply implement your own method that inherits from the respective part of the pipeline, create a corresponding config file, and add the module to the pipeline.py file. You can then use the commands outlined above to train and test the pipeline. 
Please see the documentation of the individual modules for further details:
- [Distant Supervision](convinse/distant_supervision/README.md)
- [Question Understanding (QU)](convinse/question_understanding/README.md)
- [Evidence Retrieval and Scoring (ERS)](convinse/evidence_retrieval_scoring/README.md)
- [Heterogeneous Answering (HA)](convinse/heterogeneous_answering/README.md)


## Comparing on ConvMix
Please make sure that...
- ...you use our dedicated Wikipedia dump, to have a comparable Wikipedia version (see further details below).
- ...you use the same Wikidata dump (2022-01-31), which can be conveniently accessed using the CLOCQ API available at https://clocq.mpi-inf.mpg.de (see further details below).
- ...you use the same evaluation method as CONVINSE (as defined in convinse/evaluation.py).


# Data
## Wikidata
For the CONVINSE paper, the Wikidata dump with the timestamp 2022-01-31 was used, which is currently also accessible via the [CLOCQ API](https://clocq.mpi-inf.mpg.de).
Further information on how to retrieve evidences from Wikidata can be found in the [ERS documentation](convinse/evidence_retrieval_scoring/README.md#wikidata-access).


## Wikipedia
Wikipedia evidences can be retrieved on-the-fly using the [`WikipediaRetriever`]() package. However, we provide a ConvMix-related subset, that can be downloaded via (already downloaded in default [initialize script](scripts/initialize.sh)):
``` bash
    bash scripts/download.sh wikipedia
```
The dump is provided as a .pickle file, and provides a mapping from Wikidata item IDs (e.g. Q38111) to Wikipedia evidences.

This ConvMix-related subset has been created as follows. We added evidences retrieved from Wikipedia in 2022-03/04 for the following Wikidata items:
1) all answer entities in the ConvMix benchmark,
2) all question entities in the ConvMix benchmark (as specified by crowdworkers),
3) the top-20 disambiguations for each entity mention detected by CLOCQ, with the input strings being the intent-explicit forms generated for the ConvMix dataset by the CONVINSE pipeline, or the baseline built upon the 'Prepend all' QU method,
4) and whenever new Wikidata entities occured (e.g. for the dynamic setup running the pipeline with predicted answers), we added the corresponding evidences to the dump.

We aim to maximize the number of entities (and thus of evidences) here, to allow for fair (as far as possible) comparison with dense retrieval methods. Crawling the whole Wikipedia dump was out of scope (also Wikimedia strongly discourages this). In total we collected \~230,000 entities, for which we tried retrieving Wikipedia evidences. Note that for some of these, the result was empty.

Further information on how to retrieve evidences from Wikipedia can be found in the [ERS documentation](convinse/evidence_retrieval_scoring/README.md#wikipedia-access)

# Feedback
We tried our best to document the code of this project, and for making it accessible for easy usage, and for testing your custom implementations of the individual pipeline-components. However, our strengths are not in software engineering, and there will very likely be suboptimal parts in the documentation and code.
If you feel that some part of the documentation/code could be improved, or have other feedback,
please do not hesitate and let use know! You can e.g. contact us via mail: pchristm@mpi-inf.mpg.de.
Any feedback (also positive ;) ) is much appreciated!

# License
The CONVINSE project by [Philipp Christmann](https://people.mpi-inf.mpg.de/~pchristm/), [Rishiraj Saha Roy](https://people.mpi-inf.mpg.de/~rsaharo/) and [Gerhard Weikum](https://people.mpi-inf.mpg.de/~weikum/) is licensed under [MIT license](LICENSE).

