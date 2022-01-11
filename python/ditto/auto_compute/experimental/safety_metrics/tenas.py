import torch

from safety_evaluator import BaseSafetyEvaluator


class TENASSafetyEvaluator(BaseSafetyEvaluator):

    """
    Neural Architecture Search on ImageNet in Four GPU Hours
    https://github.com/idstcv/ZenNAS/blob/main/ZeroShotProxy/compute_te_nas_score.py
    """

    def __init__(self):
        super(TENASSafetyEvaluator, self).__init__()

    def evaluate(self, module, model, input_shape, **kwargs):
        pass


if __name__ == '__main__':
    import torchvision.models as models

    model = models.resnet18(pretrained=False).cuda()
    module = dict(model.named_modules())['layer4.0']
    print(f"Selected module: {module}")

    evaluator = TENASSafetyEvaluator()
    score = evaluator.evaluate(module, model, [32, 3, 224, 224])
    print(f"score: {score:.4f}")