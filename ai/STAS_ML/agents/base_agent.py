# Базовый класс STAS_ML-агента

class BaseAgent:
    def __init__(self, config):
        self.config = config

    def act(self, state):
        # TODO: выбрать действие
        pass

    def learn(self, experience):
        # TODO: обновить параметры агента
        pass 