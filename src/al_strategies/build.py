from .al_gtd import ALGTDStrategy

def get_strategy(config):
    return ALGTDStrategy(config)
