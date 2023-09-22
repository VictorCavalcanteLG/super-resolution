from abc import ABC, abstractmethod


class ExperimentMonitor(ABC):

    @abstractmethod
    def log_loss_and_accuracy(self, loss, acc):
        """ Log loss and accuracy in experiment monitor"""

    @abstractmethod
    def watch_model(self, model, criterion, log="all", log_freq=10):
        """ Log informations about model in monitor"""
