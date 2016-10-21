from abc import ABCMeta, abstractmethod

class OCRModel(object):
    """This is the base model for other OCRModels"""

    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, tests, ans): pass

    @abstractmethod
    def run(self, test): pass

