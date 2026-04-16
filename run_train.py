import argparse

import yaml

from nisqa.NISQA_model import nisqaModel
from nisqa._resources import resolve_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml",
        required=True,
        type=str,
        help="YAML file with config; supports local files and packaged configs",
    )
    args = vars(parser.parse_args())
    yaml_path = resolve_path(args["yaml"], "config")

    with open(yaml_path, "r") as ymlfile:
        args_yaml = yaml.load(ymlfile, Loader=yaml.FullLoader)

    args = {**args_yaml, **args}
    args["yaml"] = yaml_path

    nisqa = nisqaModel(args)
    nisqa.train()






























