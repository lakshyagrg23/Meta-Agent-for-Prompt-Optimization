"""
src/core/constants.py
----------------------
Central label constants for the phishing detection framework.

**All** code that produces or consumes classification labels must import
from here.  String literals ``"PHISHING"`` and ``"SAFE"`` must not appear
anywhere else in the codebase (outside of docstrings and comments).

Label semantics
---------------
* :data:`LABEL_PHISHING` — the **positive** class.  Precision, recall, and
  F1 are always computed w.r.t. this label.
* :data:`LABEL_SAFE` — the **negative** class.  Legitimate emails that the
  system should not flag.

Usage
-----
::

    from src.core.constants import LABEL_PHISHING, LABEL_SAFE

    if prediction == LABEL_PHISHING:
        ...
"""

#: Positive class label — phishing / malicious email.
LABEL_PHISHING: str = "PHISHING"

#: Negative class label — safe / legitimate email.
LABEL_SAFE: str = "SAFE"

#: Ordered pair used wherever sklearn requires a fixed label list.
#: Order is [NEGATIVE, POSITIVE] so confusion-matrix indices are consistent:
#:   cm[0,0] = TN,  cm[0,1] = FP,  cm[1,0] = FN,  cm[1,1] = TP
LABEL_LIST: list = [LABEL_SAFE, LABEL_PHISHING]
