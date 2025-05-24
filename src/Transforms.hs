{-# OPTIONS_GHC -Wno-missing-export-lists #-}
{-# LANGUAGE OverloadedRecordDot #-}
module Transforms where

import qualified Data.Sequence as Seq

import IExpr
import TExpr2
import Util


-- lowerReduce (Reduce attr e) = undefined
--     where
--         -- f(parallel, sequential) = reduceIndex
--         -- e(reduceIndex) -> T[N]
--         nLevels = ilog2 attr.reduceParallel
--         seqInd = Seq.index attr.reduceMap.ifuncIndices 1
--         pInd = Seq.index attr.reduceMap.ifuncIndices 0
        -- inner compute
        -- T0[N * P] = compute [P, 1] |i, _| { e(f(i, s)) }
        -- T1[N * P / 2] = compute [N * P / 2, 1] |i, _| { T0[i] + T0[i + N*P/2] }
        -- ...



{-
TE1 -> TE1 optimizations
- tile + factor
- fuse
- generally expose parallelism, minimize memory traffic through global reordering of indexing, nesting
  and use of faster memory in a memory hierarchy (subject to hardware cost model)
- aim to be generic and hardware agnostic

TE1 -> TE2 lowering
- assign parallel and sequential workers, normalize nesting structure to fit onto regular SIMT parallelism

TE2 -> TE2 optimizations
1. canonicalize reduce - split reduce to compute and outer reduce with parallel=1
2. naive reduce - assume parallel=1, then lower to for loop

1. canonicalize + optimize compute
     optimization
     - assign/reorder indexing to obtain optimal index pattern given hardware cost model (more specialized now that we have concrete number of parallel elements)
        - coalesced memory
        - vectorized memory
        - async + pipeline
     - assign optimal parallel workers given hardware cost model (utilization and occupancy)
     canonicalization
     - "normalize" parallel dimensions within computes of the same level
     - merge inner computes to outer ones if parallelism is desired, otherwise all inner computes higher
        than hardware supported nesting level will be flattened to sequential

2. naive lower compute - first assume that all computes at a given level all have the same
     parallel dimension, and assume that every compute definition with nesting level higher
     than a pre-specified (hardware specific) nesting level has parallel=1
     This maps directly onto SIMT hardware specifications. For example, cuda has 3 nested levels
     (4 in sm90) grid (blockIdx), block (warpIdx), warp (threadIdx)

TE2 -> TE3 lowering
- plan memory and insert necessary synchronization

TE3 -> CuKernel

CuKernel Codegen
- generate C++ first
- maybe generate PTX/llvm ir directly


-}

