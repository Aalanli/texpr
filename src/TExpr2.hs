{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE FlexibleInstances #-}

module TExpr2 (module TExpr2) where

import IExpr
import Util
import Data.Sequence (Seq)

import Prettyprinter
import Prettyprinter.Render.String
import Data.Text (Text)


data TVar = TVar 
    { tVarId::ID
    , tVarMem::MemoryLevel
    , tVarSize::Int }

pprintTVar :: TVar -> Doc ann
pprintTVar (TVar tid mem size) = "t" <> pretty (idInt tid) <> "[" <> 
    pretty (show mem) <> "][" <> pretty size <> "]"

instance Show TVar where
    show = showPPrint . pprintTVar

tVar :: IDEnv m => MemoryLevel -> Int -> m TVar
tVar mem size = do
    tid <- newID
    return $ TVar tid mem size

-- Let R = computeRepeats  (the sequential loop)
-- Let P = computeParallel (the parallel loop)
-- Let N = P * R           (the size of output)
data ComputeAttr = ComputeAttr 
    { computeWriteBackMap::IFunc -- [N] -> [N]
    , computeMap::IFunc          -- [P, R] -> [N]
    , computeRepeats::Int 
    , computeParallel::Int
    , computeIndex::Index
    } deriving Show

-- Suppose we are reducing over N items, each with size [M]
-- we can factor the reduction N = P * S, P parallel workers reducing
-- S steps sequentially, and then doing a parallel reduction amongst the parallel workers.
data ReduceAttr = ReduceAttr
    { reduceMap::IFunc -- [P, S] -> [N]
    , reduceParallel::Int
    , reduceSequential::Int
    , reduceIndex::Index
    } deriving (Show)

data MemoryLevel = Reg | MemAddressable Int

instance Show MemoryLevel where
    show = renderString . layoutPretty defaultLayoutOptions . \case
        Reg -> "r"
        MemAddressable i -> "L" <+> pretty i

type OpCode = Text

data TExpr2 e = 
    Compute ComputeAttr e
    | Reduce ReduceAttr e
    | Let TVar e e
    | Index TVar ITerm
    | ElemWise Text (Seq e) -- TODO: add actual elemwise ops (not important for the optimizations we care about)
    deriving (Functor, Foldable, Traversable, Show)

type TTerm2 = Fix TExpr2

instance Show TTerm2 where
    show (Fix t) = show t

-- TODO (no optimizations for now) lowering to cuda
-- 1. naive transformation, flatten everything after 0, 1 levels
-- 2. plan memory

data ParallelAttr = ParallelAttr 
    { parallelWorkers::Int
    , parallelIdx::Index }

data TExpr3 e = 
    Parallel ParallelAttr (Seq e)
    | ForI Int Index (Seq e)
    | Read TVar TVar ITerm -- a[r][1] = t[L i][n][e]
    | Write TVar ITerm TVar -- t[L i][n][e] = a[r][1]
    | Alloc TVar -- no need to alloc Read results (those can be left as virtual regs)
    | Synchronize
    | ElemOp OpCode TVar (Seq TVar)


data CuKernel = CuKernel 
    { gridSize::Int
    , blockSize::Int
    , sMemVar::TVar
    , cuKernelArgs::Seq TVar
    , gridIdx::Index
    , blockIdx::Index
    , kernelStmts::Seq CuStmts }


data CuStmts = 
    ForStmt Int Index (Seq CuStmts) 
    | LoadStmt TVar TVar ITerm
    | StoreStmt TVar ITerm TVar
    | SyncStmt
    | ElemwiseStmt OpCode TVar (Seq TVar)


