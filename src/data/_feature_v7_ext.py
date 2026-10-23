# -*- coding: utf-8 -*-
"""Stub fallback for spatial_channels_v7 (used when C++ extension unavailable).

This is a COPY of the core assembly logic from feature_v7.py — kept in sync
so the parity test has a reference implementation that doesn't import feature_v7.
If feature_v7.py changes, this file must be updated too.
"""
import numpy as np
import warnings

from src.data.feature_v7 import (
    spatial_channels_v7 as _reference,
)

# Re-export the real implementation as fallback
spatial_channels_v7 = _reference
_HAVE_CPP = False
