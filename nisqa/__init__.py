__all__ = ["nisqaModel", "NISQAPredictor"]


def __getattr__(name):
    if name == "nisqaModel":
        from .NISQA_model import nisqaModel

        return nisqaModel
    if name == "NISQAPredictor":
        from .inference import NISQAPredictor

        return NISQAPredictor
    raise AttributeError("module 'nisqa' has no attribute '{}'".format(name))
