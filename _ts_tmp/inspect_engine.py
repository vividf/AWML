"""Dump TRT engine layer information (JSON) and summarize trainStation/Myelin regions.

Usage: python inspect_engine.py <engine_file>
"""

import ctypes
import json
import re
import sys
from collections import Counter

import tensorrt as trt

ENG = sys.argv[1]
ctypes.CDLL("/opt/plugins/libautoware_tensorrt_plugins.so", mode=ctypes.RTLD_GLOBAL)


class L(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)

    def log(self, sev, msg):
        if sev <= trt.ILogger.Severity.ERROR:
            print(f"[TRT {sev}] {msg}", file=sys.stderr)


logger = L()
trt.init_libnvinfer_plugins(logger, "")
rt = trt.Runtime(logger)
with open(ENG, "rb") as f:
    engine = rt.deserialize_cuda_engine(f.read())

insp = engine.create_engine_inspector()
info = insp.get_engine_information(trt.LayerInformationFormat.JSON)
data = json.loads(info)
layers = data.get("Layers", data if isinstance(data, list) else [])


def lname(layer):
    return layer if isinstance(layer, str) else layer.get("Name", "")


names = [lname(l) for l in layers]
print(f"engine: {ENG}")
print(f"total layers: {len(names)}")

# Count trainStation / Myelin / foreign-node markers
ts = [n for n in names if "trainStation" in n]
myelin = [n for n in names if re.search(r"[Mm]yelin|ForeignNode|foreign", n)]
print(f"trainStation-named layers: {len(ts)}")
for n in sorted(set(ts)):
    print("   ", n)
print(f"Myelin/ForeignNode-named layers: {len(myelin)}")
for n in sorted(set(myelin))[:20]:
    print("   ", n)

# Tally layer "LayerType" if present
types = Counter()
for l in layers:
    if isinstance(l, dict):
        types[l.get("LayerType", "?")] += 1
if types:
    print("LayerType tally:")
    for k, v in types.most_common():
        print(f"   {v:4d}  {k}")
