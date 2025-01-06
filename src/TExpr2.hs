module TExpr2 (module TExpr2) where

import IExpr

-- Let R = computeRepeats  (the sequential loop)
-- Let P = computeParallel (the parallel loop)
-- Let N = P * R           (the size of output)
data ComputeAttr = ComputeAttr 
    { computeWriteBackMap::IFunc -- [N] -> [N]
    , computeMap::IFunc          -- [P, R] -> [N]
    , computeRepeats::Int 
    , computeParallel::Int
    , computeIndex::ID
    }

-- Suppose we are reducing over N items, each with size [M]
-- we can factor the reduction N = P * S, P parallel workers reducing
-- S steps sequentially, and then doing a parallel reduction amongst the parallel workers.
data ReduceAttr = ReduceAttr
    { reduceMap::IFunc -- [P, S] -> [N]
    , reduceParallel::Int
    , reduceSequential::Int
    , reduceIndex::ID
    }

data MemoryLevel = Reg | MemLevel Int
data MemoryAttr = MemoryAttr
    { }

data TExpr2 e = 
    Compute ComputeAttr e
    | Reduce ReduceAttr e

