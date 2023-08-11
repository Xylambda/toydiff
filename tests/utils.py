import torch
import toydiff as tdf


class GradientTester:
    def __init__(self, operation):
        self.operation = operation

    def test_unary(self):
        pass

    def test_binary(self):
        # test (1) vs (1)
        # (3) vs (1)
        # (3,3) vs (3)
        # (3,3) vs (3,3)
        # (3,3,3) vs (3)
        # (3,3,3) vs (3,3)
        pass