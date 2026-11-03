"""Shared Earth Engine initialisation.

Workaround: earthengine-api 1.7.x downloads a deprecation catalogue from
storage.googleapis.com on every Initialize() call. That request has no
timeout, so it can hang on slow or restricted networks. The catalogue only
produces optional deprecation warnings, so we skip it.
"""

from __future__ import annotations

import ee
from ee import deprecation

PROJECT = "karachi-uhi"


def init(project: str = PROJECT) -> None:
    deprecation.InitializeDeprecatedAssets = lambda: None  # type: ignore[assignment]
    ee.Initialize(project=project)
