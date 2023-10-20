from abc import ABC, abstractmethod


class ExperimentMonitor(ABC):

    @abstractmethod
    def log_loss(self, train_loss, validate_loss):
        """ Log loss in experiment monitor"""

    @abstractmethod
    def watch_model(self, model, criterion, log="all", log_freq=10):
        """ Log informations about model in monitor"""
