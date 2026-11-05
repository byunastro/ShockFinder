from pathlib import Path
import re, numpy as np


def info(folder):
    folder = Path(folder)
    paths = list(folder.glob("info_*.txt"))
    if len(paths) != 1:
        raise ValueError(f"Expected one info file: {folder}")
    result = {}
    for line in paths[0].read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep:
            try:
                result[key.strip()] = float(value.strip().replace("D", "e"))
            except ValueError:
                result[key.strip()] = value.strip()
    return result


def nml_value(text, name, count=1, default=None):
    text = "\n".join(line.split("!")[0] for line in text.splitlines())
    match = re.search(r"\b" + re.escape(name) + r"\s*=\s*([^\n/]+)", text, re.I)
    if not match:
        if default is not None:
            return default
        raise ValueError(f"Missing namelist setting: {name}")
    tokens = re.split(r"[,\s]+", match.group(1).strip())
    numbers = []
    for token in tokens:
        if not token:
            continue
        # Stop if another assignment appears on the same line.
        if "=" in token:
            break
        try:
            repeat, value = token.split("*") if "*" in token else ("1", token)
            numbers.extend([float(value.lower().replace("d", "e"))] * int(repeat))
        except ValueError:
            break
    if len(numbers) < count:
        raise ValueError(f"Cannot parse {count} values for {name}")
    return numbers[0] if count == 1 else np.array(numbers[:count])
