from importlib.resources import files
from pathlib import Path


def _split_parts(path_value):
    return [part for part in str(path_value).replace("\\", "/").split("/") if part and part != "."]


def _packaged_resource_path(resource_dir, relative_path):
    resource = files("nisqa.resources").joinpath(resource_dir).joinpath(relative_path)
    if resource.is_file():
        return Path(str(resource))
    return None


def resolve_path(path_value, resource_dir):
    if not path_value:
        return path_value

    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError("File not found: {}".format(path_value))

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)

    parts = _split_parts(path_value)
    packaged_candidates = []

    if parts[:2] == ["resources", resource_dir]:
        packaged_candidates.append("/".join(parts[2:]))
    if parts[:1] == [resource_dir]:
        packaged_candidates.append("/".join(parts[1:]))
    packaged_candidates.append("/".join(parts))
    packaged_candidates.append(candidate.name)

    seen = set()
    for packaged_candidate in packaged_candidates:
        if not packaged_candidate or packaged_candidate in seen:
            continue
        seen.add(packaged_candidate)
        packaged_path = _packaged_resource_path(resource_dir, packaged_candidate)
        if packaged_path is not None:
            return str(packaged_path)

    raise FileNotFoundError(
        "Could not resolve '{}' from the current working directory or packaged '{}' resources.".format(
            path_value,
            resource_dir,
        )
    )
