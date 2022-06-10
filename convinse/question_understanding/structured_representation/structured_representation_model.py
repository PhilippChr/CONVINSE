import os
import torch
import transformers

from pathlib import Path
import convinse.question_understanding.structured_representation.dataset_structured_representation as dataset


class StructuredRepresentationModel(torch.nn.Module):
    def __init__(self, config):
        super(StructuredRepresentationModel, self).__init__()
        self.config = config
        self.sr_delimiter = self.config["sr_delimiter"]
        
        # select model architecture
        if config["sr_architecture"] == "BART":
            self.model = transformers.BartForConditionalGeneration.from_pretrained(
                "facebook/bart-base"
            )
            self.tokenizer = transformers.BartTokenizerFast.from_pretrained("facebook/bart-base")
        elif config["sr_architecture"] == "T5":
            self.model = transformers.T5ForConditionalGeneration.from_pretrained("t5-base")
            self.tokenizer = transformers.T5TokenizerFast.from_pretrained("t5-base")
        else:
            raise Exception(
                "Unknown architecture for SR module specified in config: currently, only T5-base (=T5) and BART-base (=BART) are supported."
            )

    def set_eval_mode(self):
        """Set model to eval mode."""
        self.model.eval()

    def save(self):
        """Save model."""
        model_path = self.config["sr_model_path"]
        # create dir if not exists
        model_dir = os.path.dirname(model_path)
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

    def load(self):
        """Load model."""
        if torch.cuda.is_available():
            state_dict = torch.load(self.config["sr_model_path"])
        else:
            state_dict = torch.load(self.config["sr_model_path"], torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def train(self, train_path, dev_path):
        """Train model."""
        # load datasets
        train_dataset = dataset.DatasetStructuredRepresentation(
            self.config, self.tokenizer, train_path
        )
        dev_dataset = dataset.DatasetStructuredRepresentation(self.config, self.tokenizer, dev_path)
        # arguments for training
        training_args = transformers.Seq2SeqTrainingArguments(
            output_dir="convinse/question_understanding/structured_representation/results",  # output directory
            num_train_epochs=self.config["sr_num_train_epochs"],  # total number of training epochs
            per_device_train_batch_size=self.config[
                "sr_per_device_train_batch_size"
            ],  # batch size per device during training
            per_device_eval_batch_size=self.config[
                "sr_per_device_eval_batch_size"
            ],  # batch size for evaluation
            warmup_steps=self.config[
                "sr_warmup_steps"
            ],  # number of warmup steps for learning rate scheduler
            weight_decay=self.config["sr_weight_decay"],  # strength of weight decay
            logging_dir="convinse/question_understanding/structured_representation/logs",  # directory for storing logs
            logging_steps=1000,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end="True"
            # predict_with_generate=True
        )
        # create the object for training
        trainer = transformers.Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
        )
        # training progress
        trainer.train()
        # store model
        self.save()

    def inference(self, input):
        """Run the model on the given input."""
        # encode
        input_encodings = self.tokenizer(
            input,
            padding=True,
            truncation=True,
            max_length=self.config["sr_max_input_length"],
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            input_encodings = input_encodings.to(torch.device("cuda"))
        # generate
        output = self.model.generate(
            input_ids=input_encodings["input_ids"],
            attention_mask=input_encodings["attention_mask"],
            no_repeat_ngram_size=self.config["sr_no_repeat_ngram_size"],
            num_beams=self.config["sr_num_beams"],
            early_stopping=self.config["sr_early_stopping"],
        )
        # decoding
        sr = self.tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        # format SR properly
        sr = self._format_sr(sr)
        return sr

    def _format_sr(self, sr):
        """Make sure the SR has 3 delimiters."""
        slots = sr.split(self.sr_delimiter.strip(), 3)
        if len(slots) < 4 and not slots[0]:
            # type missing
            slots = slots + [""]
        elif len(slots) < 4:
            # topic missing
            slots = [""] + slots
        if len(slots) < 4:
            # in case there are still less than 4 slots
            slots = slots + (4 - len(slots)) * [""]
        sr = self.sr_delimiter.join(slots)
        return sr

    def inference_on_batch(self, inputs):
        """Run the model on the given inputs (batch)."""
        # encode inputs
        input_encodings = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.config["sr_max_input_length"],
            return_tensors="pt",
        )
        # generation
        summary_ids = self.model.generate(
            input_ids=input_encodings["input_ids"],
            attention_mask=input_encodings["attention_mask"],
            no_repeat_ngram_size=self.config["sr_no_repeat_ngram_size"],
            num_beams=self.config["sr_num_beams"],
            early_stopping=self.config["sr_early_stopping"],
        )
        # decoding
        output = [
            self.tokenizer.decode(
                g,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for g in summary_ids
        ]
        return output
