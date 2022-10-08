import pytorch_lightning as pl
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.text_classification import (
    TextClassificationDataModule,
    TextClassificationTransformer,
)


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="bert-base-uncased"
    )
    dm = TextClassificationDataModule(
        batch_size=1,
        dataset_name="wiki_qa",
        max_length=512,
        input_feature_fields=["question", "answer"],
        tokenizer=tokenizer,
        truncation='only_second',
    )
    model = TextClassificationTransformer(
        pretrained_model_name_or_path="bert-base-uncased",
        num_labels=dm.num_classes
    )
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1)

    trainer.fit(model, dm)
