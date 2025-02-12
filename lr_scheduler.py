import math
from typing import Literal


class LearningRateScheduler:
    def __init__(
        self,
        lr=1.0,
        type: Literal["step", "exponential", "cosine"] = "step",
        decay_rate=0.1,
        step_size=10,
    ):
        self.lr = lr
        self.type = type
        self.decay_rate = decay_rate
        self.step_size = step_size

    def step_decay(self, epoch):
        self.lr = self.lr * self.decay_rate ** (math.floor(1 + epoch) / self.step_size)
        return self.lr

    def exponential_decay(self, epoch):
        self.lr = self.lr * math.exp(-self.decay_rate * epoch)
        return self.lr

    def cosine_decay(self, epoch, total_epochs):
        self.lr = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs)) * self.lr
        return self.lr

    def get_lr(self, epoch, total_epochs):
        print(self.lr)
        if self.type == "step":
            return self.step_decay(epoch)
        elif self.type == "exponential":
            return self.exponential_decay(epoch)
        elif self.type == "cosine":
            return self.cosine_decay(epoch, total_epochs)
