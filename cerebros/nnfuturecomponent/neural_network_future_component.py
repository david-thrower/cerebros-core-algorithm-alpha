class NeuralNetworkFutureComponent:
    """docstring for ."""

    def __init__(self,
                 trial_number: int,
                 level_number: int,
                 *args,
                 **kwargs):
        self.trial_number = trial_number
        self.level_number = level_number
        self.name = f"{str(type(self)).split('.')[-1]}_{str(self.level_number).zfill(16)}_tr_{self.trial_number}"\
            .replace("<class '", '')\
            .replace("'>", '')

    def __str__(self):
        return str(self.__dict__)
