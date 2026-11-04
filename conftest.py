import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.expanduser("~/qwentpu_src/qwentpu"), os.path.join(os.path.expanduser("~/qwentpu_src/qwentpu"), "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)
