import time

import tensorflow as tf
import numpy as np

# pip install transformers
from transformers.models.auto.modeling_tf_auto import TFAutoModel
from transformers.models.auto.configuration_auto import AutoConfig


def run_benchmark(batch_size=8, seq_len=512):
    # The bare Bert Model without any specific head on top.
    config = AutoConfig.from_pretrained("bert-base-uncased")
    model = TFAutoModel.from_config(config)

    @tf.function
    def test_step(inp):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        outp = model(inp, training=False)
        return outp

    # Indices of input sequence tokens in the vocabulary. Indices can be obtained using BertTokenizer
    input_ids = np.zeros([batch_size, seq_len], dtype=np.int64)

    number = 10
    repeats = 10

    for i in range(number):
        records = []
        for j in range(repeats):
            start_time = time.time()
            test_step(input_ids)
            end_time = time.time()
            records.append(end_time - start_time)
        print("Average inference latency {} ms".format(1000.0 * np.mean(records)))
        print("Median inference latency {} ms".format(1000.0 * np.median(records)))


bert_configs = [
    # seq-len
    (512,)
]


if __name__ == "__main__":
    batches = [2 ** i for i in range(1)]
    for batch in batches:
        for i, config in enumerate(bert_configs):
            run_benchmark(batch, *config)

    # from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments

    # args = TensorFlowBenchmarkArguments(models=["bert-base-uncased"], batch_sizes=[1], sequence_lengths=[512])
    # benchmark = TensorFlowBenchmark(args)

    # print(benchmark.run())
