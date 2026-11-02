"""Synthetic CPU kernel benchmark only; no real CUB or research quality claim."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
import numpy as np
from threadpoolctl import threadpool_limits
from online_gcd.prototype_bank import Prototype, normalize
from online_gcd.selex_euclidean import SelexEuclideanBank, SelexOptions
from reference_selex_euclidean import ReferenceBank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prototypes', type=int, default=128)
    parser.add_argument('--observations', type=int, default=48)
    parser.add_argument('--dimension', type=int, default=768)
    parser.add_argument('--block-size', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if min(vars(args).values()) < 1:
        parser.error('sizes/repetitions must be positive')
    rng = np.random.default_rng(123)
    seeds = rng.normal(size=(args.prototypes,args.dimension))
    seeds[:min(9,args.prototypes)] = seeds[0]
    # Distinct observation IDs with identical generated features exercise a
    # supported sparse relation, rather than benchmarking an empty graph only.
    stream = np.repeat(seeds[:1], args.observations, axis=0)
    times = {'optimized': [], 'reference': []}
    with threadpool_limits(limits=1,user_api='blas'):
        for _ in range(args.repeats):
            base = SelexEuclideanBank(args.dimension,-1.,structure_threshold=.75,
                                      options=SelexOptions(block_size=args.block_size))
            base.initialize_known({0: [np.eye(1,args.dimension)[0]]})
            for i, vector in enumerate(seeds,1):
                cid=f'novel-{i:06d}'
                base._prototypes[cid]=Prototype(cid,tuple(normalize(vector,args.dimension)),1,False,1,1)
            base._next_novel_id=args.prototypes+1
            base._last_session=1
            base.observation_steps=args.prototypes
            base._rebuild_cache()
            reference=ReferenceBank(args.dimension,-1.,structure_threshold=.75)
            reference.__dict__.update(deepcopy(base.__dict__))
            outputs=[]
            for name, bank in [('optimized',base),('reference',reference)]:
                start=time.perf_counter()
                predictions=[bank.observe(str(i),2,x) for i,x in enumerate(stream)]
                assigned=time.perf_counter()
                structure = bank.rebuild()
                done=time.perf_counter()
                times[name].append({'assignment_evidence_seconds':assigned-start,'hierarchy_seconds':done-assigned})
                outputs.append(predictions)
            assert outputs[0]==outputs[1] and base.prototypes==reference.prototypes and base.nodes==reference.nodes
            assert base.edges.keys()==reference.edges.keys()
            for pair in base.edges:
                assert base.edges[pair][1]==reference.edges[pair][1]
                assert abs(base.edges[pair][0]-reference.edges[pair][0])<=2e-13
    medians={name:{key:statistics.median(row[key] for row in runs) for key in runs[0]}
             for name,runs in times.items()}
    print(json.dumps({'fixture':'Gaussian leaves with nine duplicated centroids and distinct observation IDs; not CUB evidence','python':platform.python_version(),
                      'numpy':np.__version__,'parameters':vars(args),'blas_threads':1,'median_seconds':medians,
                      'speedup':{key:medians['reference'][key]/medians['optimized'][key] for key in medians['optimized']},
                      'active_nodes':len(base.nodes),'eligible_edges':structure['eligible_structural_edge_count'],
                      'exact_ids_leaf_state_and_hierarchy':True,'evidence_sum_absolute_tolerance':2e-13},indent=2))


if __name__ == '__main__':
    main()
