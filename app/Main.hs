{-# OPTIONS_GHC -Wno-incomplete-uni-patterns #-}
{-# LANGUAGE OverloadedLists #-}
import IExpr

import Util
import TExpr2
import Data.Text (unpack)
import Data.Sequence (fromList)
import Text.Pretty.Simple

data Tensor = Tensor {
    tensorShape::[Int],
    tensorStride::[Int],
    indexOffsets::[ITerm],
    tensorDim::Int,
    tensorVar::TVar
}

strides = tail . scanr (*) 1

mkTensor :: TVar -> [Int] -> Tensor
mkTensor tvar shape = Tensor shape (strides shape) (map (const (iConst 0)) shape) (length shape) tvar

newTensor :: IDEnv m => MemoryLevel -> [Int] -> m Tensor
newTensor m shape = let size = product shape in
    do 
        t <- tVar m size
        return $ mkTensor t shape

(@!) :: Tensor -> [ITerm] -> TTerm2
tvar @! indices
    | length (tensorStride tvar) /= length indices = error "Indexing error"
    | otherwise = let 
        i = foldl1 (.+) $ zipWith (.*) (map iConst $ tensorStride tvar) (zipWith (.+) indices (indexOffsets tvar)) in
        Fix $ Index (tensorVar tvar) i

-- dim offset, dim size
dynSlice :: Tensor -> [(ITerm, Int)] -> Tensor
dynSlice tvar slices
    | not isSizeWellDefined || (length (tensorShape tvar) /= length slices) = error "invalid slice"
    | otherwise = Tensor newShape (strides newShape) (map fst slices) (tensorDim tvar) (tensorVar tvar)
    where 
        newShape = map snd slices
        isSizeWellDefined = and $ zipWith (<=) newShape (tensorShape tvar)

factorIdx :: ITerm -> [Int] -> [ITerm]
factorIdx i shapes = let 
    shTerms = map iConst shapes
    stTerms = map iConst (strides shapes) in
    init $ scanr (\(sh, st) i' -> (i' ./ st) .% sh) i (zip shTerms stTerms)


mmaTile :: IDEnv m => Tensor -> Tensor -> m TTerm2
mmaTile at bt
    | tensorDim at /= 2 || tensorDim bt /= 2 = error "incorrect dim"
    | k /= k' = error "incorrect shape"
    | otherwise = do
        idx <- newIndex sT
        rIdx <- newIndex k
        wBM <- makeIFunc [sT] head
        cM <- makeIFunc [1, sT] $ \[_, r] -> r
        rMap <- makeIFunc [1, k] $ \[_, s] -> s
        let cattr = ComputeAttr 
                { computeRepeats = m * n
                , computeParallel = 1
                , computeIndex = idx
                , computeWriteBackMap = wBM
                , computeMap = cM
                }
        let rattr = ReduceAttr {
            reduceIndex = rIdx,
            reduceParallel = 1,
            reduceSequential = k,
            reduceMap = rMap
        }
        let [mi, ni] = factorIdx (Fix (I idx)) [m, n]
        let ki = Fix (I rIdx)

        let ai = at @! [mi, ki]
        let bi = bt @! [ki, ni]
        let ci = Fix $ ElemWise "add" $ fromList [ai, bi]
        return $ Fix $ Compute cattr $ Fix 
            $ Reduce rattr ci
    where 
        [m, k] = tensorShape at
        [k', n] = tensorShape bt

        sT = m * n

data MMATileConfig = MMATileConfig 
    { mTile::Int
    , nTile::Int
    , kTile::Int
    , mWorker::Int
    , nWorker::Int
    }

mmaOuterTile :: IDEnv m => 
    MMATileConfig 
        -> Tensor 
        -> Tensor 
        -> (Tensor -> Tensor -> m TTerm2) -> m TTerm2
mmaOuterTile tile at bt f = do
    cIdx <- newIndex (tN * tM)
    rIdx <- newIndex (k `div` kTile tile)
    cWBMap <- makeIFunc [tN * tM] $ \[i] -> let 
        cthTile = i ./ iConst cTileSize
        ic = i .% iConst cTileSize
        icm = ic ./ iConst (nTile tile)
        icn = ic .% iConst (nTile tile)
        jcm = cthTile ./ iConst tN
        jcn = cthTile .% iConst tN
        in (icm .+ jcm) .* iConst n .+ icn .+ jcn
    cMap <- makeIFunc [mWorker tile * nWorker tile, mSeq * nSeq] $ \[i, j] -> let
        iM = i ./ iConst (nWorker tile)
        iN = i .% iConst (nWorker tile)
        jM = j ./ iConst nSeq
        jN = j .% iConst nSeq
        in (iM .+ jM) .* iConst tN .+ iN .+ jN
    let cAttr = ComputeAttr {
            computeWriteBackMap = cWBMap,
            computeMap = cMap,
            computeRepeats = mSeq * nSeq,
            computeParallel = mWorker tile * nWorker tile,
            computeIndex = cIdx
        }
    rMap <- makeIFunc [1, reduceK] $ \[_, ik] -> ik
    let rAttr = ReduceAttr
            { reduceMap = rMap
            , reduceParallel = 1
            , reduceSequential = reduceK
            , reduceIndex = rIdx }
    let cI = Fix (I cIdx)
    let indM = cI ./ iConst (nTile tile)
    let indN = cI .% iConst (nTile tile)
    let indK = Fix (I rIdx)
    let atSlice = at `dynSlice` [(indM, mTile tile), (indK, kTile tile)]
    let btSlice = bt `dynSlice` [(indK, kTile tile), (indN, nTile tile)]
    inner <- f atSlice btSlice
    return $ Fix $ Compute cAttr $ Fix $ Reduce rAttr inner
    where
        [m, k] = tensorShape at
        [k', n] = tensorShape bt
        cTileSize = mTile tile * nTile tile
        tN = (n `div` nTile tile)
        tM = (m `div` mTile tile)
        mSeq = m `div` (mTile tile * mWorker tile)
        nSeq = n `div` (nTile tile * nWorker tile)
        reduceK = k `div` kTile tile

testMMATile :: (IDEnv m) => m TTerm2
testMMATile = do
    at <- newTensor Reg [64, 32]
    bt <- newTensor Reg [32, 16]
    mmaTile at bt        

main = do
    let mma = runSimpleIDEnv testMMATile
    let opt = defaultOutputOptionsDarkBg 
            { outputOptionsCompact = True
            , outputOptionsIndentAmount = 2 }
    pPrintOpt NoCheckColorTty opt mma

