def ceil(x, y):
    return (x + y - 1) // y


def uround(x, y):
    return int(ceil(x, y) * y)


shapes = [
    # (batch, M, N, K, L)
    (8, 512, 512 // 8, 512 // 8, 512),  # Bert-Small
    (12, 512, 768 // 12, 768 // 12, 512),  # Bert-Base
    (16, 512, 1024 // 16, 1024 // 16, 512),  # Bert-Large
    (12, 256, 768 // 12, 768 // 12, 256),  # ViT-Base/14
    (16, 256, 1024 // 16, 1024 // 16, 256),  # ViT-Large/14
    (16, 256, 1280 // 16, 1280 // 16, 256),  # ViT-Huge/14
    (12, uround(196, 16), 768 // 12, 768 // 12, uround(196, 16)),  # ViT-Base/16
    (16, uround(196, 16), 1024 // 16, 1024 // 16, uround(196, 16)),  # ViT-Large/16
    (16, uround(196, 16), 1280 // 16, 1280 // 16, uround(196, 16)),  # ViT-Huge/16
    (1, uround(49, 16), 512, 512, 2048),  # Mixer-Small/32-C
    (1, 512, uround(49, 16), uround(49, 16), 256),  # Mixer-Small/32-S
    (1, uround(196, 16), 512, 512, 2048),  # Mixer-Small/16-C # compute-bound
    (1, 512, uround(196, 16), uround(196, 16), 256),  # Mixer-Small/16-S # compute-bound
    (1, uround(49, 16), 768, 768, 3072),  # Mixer-Base/32-C
    (1, 768, uround(49, 16), uround(49, 16), 384),  # Mixer-Base/32-S
    (1, uround(196, 16), 768, 768, 3072),  # Mixer-Base/16-C # compute-bound
    (1, 768, uround(196, 16), uround(196, 16), 384),  # Mixer-Base/16-S # compute-bound
    (1, uround(49, 16), 1024, 1024, 4096),  # Mixer-Large/32-C
    (1, 1024, uround(49, 16), uround(49, 16), 512),  # Mixer-Large/32-S
    (1, uround(196, 16), 1024, 1024, 4096),  # Mixer-Large/16-C # compute-bound
    (1, 1024, uround(196, 16), uround(196, 16), 512),  # Mixer-Large/16-S # compute-bound
    (1, 256, 1280, 1280, 5120),  # Mixer-Huge/14-C # compute-bound
    (1, 1280, 256, 256, 640),  # Mixer-Huge/14-S # compute-bound
]
