from copy import deepcopy
from typing import Dict

import jax
import jax.numpy as jnp
import optax


class Metric:
    def __init__(self, compute_on_step=True):
        self.compute_on_step = compute_on_step

    def update(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        instanteneous_value = self.update(*args, **kwargs)
        if self.compute_on_step:
            return self.compute()

    def compute(self):
        pass

    def reset(self):
        pass


class MeanMetric(Metric):
    def __init__(self, compute_on_step=True):
        super().__init__(compute_on_step)
        self.value = 0
        self.num_samples = 0

    def add_prefix_postfix(self, metrics):
        if self.postfix is None and self.prefix is None:
            return metrics

        new_metrics = {}
        for k, v in metrics.values():
            new_metrics[self.prefix + k + self.postfix] = v

        return new_metrics

    def update(self, value, num_samples=1):
        if not isinstance(value, (int, float)):
            value = value.item()
        self.value += value * num_samples
        self.num_samples += num_samples

        # return instantenuous
        return value

    def reset(self):
        self.value = 0
        self.num_samples = 0

    def compute(self):
        assert self.num_samples > 0
        return self.value / self.num_samples


class AccuracyMetric(MeanMetric):
    def update(self, logits, labels):
        value = jnp.mean(jnp.argmax(logits, -1) == labels)
        return super().update(value=value, num_samples=len(labels))


class CrossEntropyLossMetric(MeanMetric):
    def update(self, logits, labels):
        labels_onehot = jax.nn.one_hot(labels, num_classes=logits.shape[1])
        value = optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()
        return super().update(value=value, num_samples=len(labels))


class ForgettingMetric(Metric):
    def __init__(self, compute_on_step=True):
        super().__init__(compute_on_step)
        self.max_accuracy = -10
        self.current_accuracy = 0

    def update(self, accuracy):
        self.current_accuracy = accuracy
        if self.current_accuracy > self.max_accuracy:
            self.max_accuracy = accuracy

    def compute(self):
        return self.current_accuracy - self.max_accuracy


class MetricCollection:
    def __init__(self, metrics: Dict[str, Metric], prefix=None, postfix=None, compute_on_step=True):
        self.metrics = metrics
        self.metric_names = list(metrics.keys())
        self.prefix = prefix
        self.postfix = postfix
        self.compute_on_step = True

    def __iter__(self):
        return iter(self.metric_names)

    def keys(self):
        return self.metric_names

    def __call__(self, *args, **kwargs):
        instanteneous_value = self.update(*args, **kwargs)
        if self.compute_on_step:
            return self.compute()

    def add_prefix_postfix(self, metrics):
        postfix = "" if self.postfix is None else self.postfix
        prefix = "" if self.prefix is None else self.prefix

        new_metrics = {}
        for k, v in metrics.items():
            new_metrics[prefix + k + postfix] = v

        return new_metrics

    def update(self, logits, labels):
        i_values = {}
        for metric_name, metric in self.metrics.items():
            i_values[metric_name] = metric.update(logits, labels)

        # return instantenuous
        return self.add_prefix_postfix(metrics=i_values)

    def reset(self):
        for _, v in self.metrics.items():
            v.reset()

    def compute(self):
        results = {k: v.compute() for k, v in self.metrics.items()}
        return self.add_prefix_postfix(results)

    def clone(self, prefix=None, postfix=None):
        if prefix is not None:
            if self.prefix is None:
                _prefix = ""
            prefix = self.prefix if prefix is None else prefix + _prefix
        else:
            prefix = self.prefix

        if postfix is not None:
            if self.postfix is None:
                _postfix = ""
            postfix = self.postfix if postfix is None else _postfix + postfix
        else:
            postfix = self.postfix

        _clone = deepcopy(self)
        _clone.prefix = prefix
        _clone.postfix = postfix

        _clone.reset()
        return _clone
