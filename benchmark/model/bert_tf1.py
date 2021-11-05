import time

import numpy as np
import tensorflow as tf

# pip install bert-tensorflow
from bert import modeling


def build_bert_model(batch_size, seq_len):
    config = modeling.BertConfig(vocab_size=30522)
    input_ids = tf.ones([batch_size, seq_len], dtype=tf.int64)
    model = modeling.BertModel(config=config, is_training=False, input_ids=input_ids)
    output_layer = model.get_sequence_output()
    return output_layer


def run_benchmark(batch_size, seq_len):
    output_layer = build_bert_model(batch_size, seq_len)
    init_op = tf.global_variables_initializer()

    session = tf.Session()
    session.run(init_op)

    number = 10
    repeats = 10

    for i in range(number):
        records = []
        for j in range(repeats):
            start_time = time.time()
            session.run(output_layer)
            end_time = time.time()
            records.append(end_time - start_time)
        print("Average inference latency {} ms".format(1000. * np.mean(records)))
        print("Median inference latency {} ms".format(1000. * np.median(records)))


bert_configs = [
    # seq-len
    (512, )
]


if __name__ == "__main__":
    batches = [2**i for i in range(1)]
    for batch in batches:
        for i, config in enumerate(bert_configs):
            run_benchmark(batch, *config)
