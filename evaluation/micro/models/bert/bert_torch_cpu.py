import torch
import numpy as np
import time

# pip install transformers
import transformers
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.configuration_auto import AutoConfig


def run_benchmark(batch_size=8, seq_len=512):
    # The bare Bert Model without any specific head on top.
    # config = AutoConfig.from_pretrained("bert-base-uncased")
    # at beginning of the script
    device = torch.device("cpu")

    # then whenever you get a new Tensor or Module
    # this won't copy if they are already on the desired device
    
    # config = transformers.BertConfig(
    #     vocab_size=30522,
    #     hidden_size=512,
    #     num_hidden_layers=4,
    #     num_attention_heads=8,
    #     intermediate_size=512,
    #     hidden_act="gelu",
    #     hidden_dropout_prob=0.1,
    #     attention_probs_dropout_prob=0.1,
    #     max_position_embeddings=512,
    #     type_vocab_size=2,
    #     initializer_range=0.02,
    #     layer_norm_eps=1e-12,
    #     pad_token_id=0,
    #     position_embedding_type="absolute",
    #     use_cache=True,
    #     classifier_dropout=None,
    # )
    # config = transformers.BertConfig(
    #     vocab_size=30522,
    #     hidden_size=768,
    #     num_hidden_layers=12,
    #     num_attention_heads=12,
    #     intermediate_size=3072,
    #     hidden_act="gelu",
    #     hidden_dropout_prob=0.1,
    #     attention_probs_dropout_prob=0.1,
    #     max_position_embeddings=512,
    #     type_vocab_size=2,
    #     initializer_range=0.02,
    #     layer_norm_eps=1e-12,
    #     pad_token_id=0,
    #     position_embedding_type="absolute",
    #     use_cache=True,
    #     classifier_dropout=None,
    # )
    config = transformers.BertConfig(
        vocab_size=30522,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
    )
    model = AutoModel.from_config(config).to(device)

    # Indices of input sequence tokens in the vocabulary. Indices can be obtained using BertTokenizer
    input_ids = torch.zeros([batch_size, seq_len], dtype=torch.long)
    input_ids = input_ids.to(device)

    # output: BaseModelOutputWithPoolingAndCrossAttentions, [bs, seq-len, hid-dim]
    with torch.no_grad():
        output = model(input_ids)

    number = 10
    repeats = 10

    for i in range(repeats):
        time_record = []
        for j in range(number):
            start = time.time()

            with torch.no_grad():
                model(input_ids)

            end = time.time()
            time_record.append((end - start) * 1e3)

        if i == repeats - 1:
            print("Best bert latency", min(time_record), "ms")
            print("Average bert latency", np.mean(time_record), "ms")
            print("Median  bert latency", np.median(time_record), "ms")


bert_config = [
    # seq-len
    (512,)
]


if __name__ == "__main__":
    batches = [2 ** i for i in range(1)]

    for batch in batches:
        for i, config in enumerate(bert_config):
            run_benchmark(batch, *config)

    # from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments

    # args = PyTorchBenchmarkArguments(models=["bert-base-uncased"], batch_sizes=[1], sequence_lengths=[512])
    # benchmark = PyTorchBenchmark(args)

    # print(benchmark.run())
