import torch
import numpy as np

# pip install transformers
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.configuration_auto import AutoConfig


def run_benchmark(batch_size=8, seq_len=512):
    # The bare Bert Model without any specific head on top.
    config = AutoConfig.from_pretrained("bert-base-uncased")
    model = AutoModel.from_config(config)
    model = model.cuda().eval()

    # Indices of input sequence tokens in the vocabulary. Indices can be obtained using BertTokenizer
    input_ids = torch.zeros([batch_size, seq_len], dtype=torch.long)
    input_ids = input_ids.cuda()

    # output: BaseModelOutputWithPoolingAndCrossAttentions, [bs, seq-len, hid-dim]
    with torch.no_grad():
        model(input_ids)

    number = 10
    repeats = 10

    for i in range(repeats):
        time_record = []
        for j in range(number):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            with torch.no_grad():
                model(input_ids)

            end.record()
            torch.cuda.synchronize()
            total = start.elapsed_time(end)
            time_record.append(total)
        if i == repeats - 1:
            print("Average bert latency", np.mean(time_record), "ms")
            print("Median  bert latency", np.median(time_record), "ms")


bert_config = [
    # seq-len
    (512,)
]


if __name__ == "__main__":
    assert torch.backends.cudnn.is_available()
    torch.backends.cudnn.enabled = True
    batches = [2 ** i for i in range(1)]

    for batch in batches:
        for i, config in enumerate(bert_config):
            run_benchmark(batch, *config)

    # from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments

    # args = PyTorchBenchmarkArguments(models=["bert-base-uncased"], batch_sizes=[1], sequence_lengths=[512])
    # benchmark = PyTorchBenchmark(args)

    # print(benchmark.run())
