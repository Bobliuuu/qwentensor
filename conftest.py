import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
# qwentensor imports qwentpu, which `pip install -e .` installs from its public repository.
# To test against a local qwenTPU checkout instead, set QWENTPU_SRC=/path/to/qwentpu.
paths = [HERE]
if os.environ.get("QWENTPU_SRC"):
    paths += [os.environ["QWENTPU_SRC"], os.path.join(os.environ["QWENTPU_SRC"], "tests")]
for p in paths:
    if p not in sys.path:
        sys.path.insert(0, p)
