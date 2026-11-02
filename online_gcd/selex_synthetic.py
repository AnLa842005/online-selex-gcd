"""Known-base-supported synthetic geometry, embedded in 768D by the fixture CLI.

This is a data builder, not a learner. The first two ID-sorted known examples
are central anchors; remaining base features alternate symmetric known noise.
Stream/probe samples still come from the original deterministic generator.
"""
from dataclasses import replace
import numpy as np

from online_gcd.prototype_bank import normalize
from online_gcd.synthetic import generate_synthetic


def generate_selex_synthetic(config):
    data = generate_synthetic(config)
    replacements = {}
    for c in data.known_classes:
        rows = sorted((p for p in data.base if p.class_id == c), key=lambda p:p.sample_id)
        center = normalize(np.mean([p.feature for p in rows],axis=0),config.feature_dim)
        axis = np.zeros(config.feature_dim)
        axis[-1] = 1.
        tangent = normalize(axis - np.dot(axis,center)*center,config.feature_dim)
        for i,p in enumerate(rows):
            feature = center if i < 2 else normalize(center + (-1 if i%2 else 1)*.25*tangent,config.feature_dim)
            replacements[p.sample_id]=replace(p,feature=tuple(feature))
    return replace(data,base=tuple(replacements[p.sample_id] for p in data.base))
