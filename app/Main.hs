{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE DeriveFunctor #-}


import Control.Monad.Except

import Data.Text (unpack, pack)
import qualified Data.Vector as Vec
import Data.Map (Map)
import qualified Data.Map as Map
import qualified Data.Set as Set

import TExpr
import IExpr


matmul :: TTerm -> TTerm -> TEBuilder TTerm
matmul a b = do
    sa <- calcShape a
    sb <- calcShape b
    when (sa /= sb) $ do
        throwError "matmul shape mismatch"
    tcompute 1024 (\i ->
        tcompute 1024 (\j ->
            treduce 1024 (\k -> do
                ai <- tindex a ((i `imul` 1024) `iadd` k)
                bi <- tindex b ((k `imul` 1024) `iadd` j)
                telemwise "add" [ai, bi]
            )))

data Conv2DConfig = Conv2DConfig {
    batch::Int,
    channelIn::Int,
    imHeight::Int,
    imWidth::Int,
    channelOut::Int,
    filterX::Int,
    filterY::Int,
    dilX::Int,
    dilY::Int,
    strideX::Int,
    strideY::Int
}

indexRowMajor :: [Int] -> [ITerm] -> ITerm
indexRowMajor dims coords =
    let strides = tail $ scanr (*) 1 dims
    in foldl iadd (iconst 0) $ zipWith imul coords strides

conv2d :: Conv2DConfig -> TTerm -> TTerm -> TEBuilder TTerm
conv2d c im f = do
    sIm <- calcShape im
    sF <- calcShape f
    let imDims = [c.batch, c.channelIn, c.imHeight, c.imWidth]
    let fDims = [c.channelOut, c.channelIn, c.filterY, c.filterX]
    when (sIm /= product imDims) $ do
        throwError $ pack $ "im shape mismatch, im dims" ++ show sIm ++ "!=" ++ show (product imDims)
    when (sF /= product fDims) $ do
        throwError "filter shape mismatch"
    let p = (c.imHeight - c.dilX * (c.strideX - 1) - 1) `div` c.strideX + 1
    let q = (c.imWidth - c.dilY * (c.strideY - 1) - 1) `div` c.strideY + 1
    tcompute c.batch (\ib ->
        tcompute c.channelOut (\ic ->
            tcompute p (\ip ->
                tcompute q (\iq ->
                    treduce c.channelIn (\jc ->
                        treduce c.filterX (\jx ->
                            treduce c.filterY (\jy -> do
                                imx <- tindex im (indexRowMajor imDims
                                    [ib, jc,
                                     (ip `imul` c.strideX) `iadd` (jx `imul` c.dilX),
                                     (iq `imul` c.strideY) `iadd` (jy `imul` c.dilY)])
                                fx <- tindex f (indexRowMajor fDims [ic, jc, jx, jy])
                                telemwise "mul" [imx, fx]
                            )))))))

progMatmul = do
    a <- tvar (1024 * 1024)
    b <- tvar (1024 * 1024)
    matmul a b

progConv2d = do
    im <- tvar (4 * 64 * 512 * 512)
    f <- tvar (128 * 64 * 3 * 3)
    let conv = Conv2DConfig {
        batch=4,
        channelIn=64,
        imHeight=512,
        imWidth=512,
        channelOut=128,
        filterX=3,
        filterY=3,
        dilX=1,
        dilY=1,
        strideX=1,
        strideY=1
    }
    conv2d conv im f


pprintProg :: TEBuilder TTerm -> String
pprintProg prog = unpack $ case runTEBuilder prog of
    Right e -> showTTerm e
    Left e -> e


main :: IO ()
main = do
    print ""
    putStrLn $ pprintProg progMatmul
    putStrLn $ pprintProg progConv2d
    -- let exprSTree = map (pruneSTree 3) (genSearch 5 (ID 0) 32)
    -- print $ map (estimateTerms) exprSTree
    -- let bijections = Map.toList $ mergeT $ map (Map.filterWithKey (\k _ -> Vec.length k > 1 && isBijection k) . reifySTree) exprSTree
    -- mapM_ (\(v, e) -> do
    --     print $ showITerm e
    --     print v) bijections


isTerminal (I _ _) = True
isTerminal (IConst _) = True
isTerminal _ = False

itermSize = cata (\case
    I _ _ -> 1
    IConst _ -> 1
    a -> sum a)

newtype IExprChoices i = IChoices {unChoice::IExpr [i]} deriving (Show, Functor)
type IExprTree = Fix IExprChoices

isBijection v = let
    n = Vec.length v
    in n == Set.size (Set.fromList $ filter (\i -> 0 <= i && i < n) (Vec.toList v))

genSearch n i d = searchTree
    where
        constants = [IConst t | t <- [1..n]]
        terminals = I i d : constants
        termsTree = map (Fix . IChoices) (tail terminals)
        searchTerms = terminals ++ [
                Add searchTree searchTree
                , Sub searchTree searchTree
                , Mul searchTree termsTree
                , Div searchTree termsTree
                , Mod searchTree termsTree
            ]
        searchTree = map (Fix . IChoices) searchTerms


pruneSTree :: Int -> IExprTree -> IExprTree
pruneSTree depth (Fix (IChoices e)) = Fix $ IChoices $ lower e
    where
        lower = fmap (\xs -> if depth == 1 then
            filter (isTerminal . unChoice . unFix) xs
            else map (pruneSTree (depth - 1)) xs)

minTerm a b
    | itermSize a < itermSize b = a
    | otherwise = b
mergeM a b = foldr (\(x, e) m -> case Map.lookup x m of
    Just e' -> Map.insert x (minTerm e e') m
    Nothing -> Map.insert x e m) a (Map.toList b)
mergeT = foldr mergeM Map.empty

estimateTerms :: IExprTree -> Integer
estimateTerms = cata (\(IChoices xs) -> foldr ((*) . sum) 1 xs)

reifySTree :: IExprTree -> Map (Vec.Vector Int) ITerm
reifySTree = cata (\(IChoices xs) -> let
   merged = fmap mergeT xs
   in reifySTree' merged)
    where
        broadcast f a b
            | Vec.length a == 1 = Vec.map (f (a Vec.! 0)) b
            | Vec.length b == 1 = broadcast (flip f) b a
            | Vec.length a == Vec.length b = Vec.zipWith f a b
            | otherwise = error "broadcast error: vecs different shape"
        

        interpT f c a b = let
            prod = (,) <$> Map.toList a <*> Map.toList b
            val = map (\((v1, e1), (v2, e2)) -> (broadcast f v1 v2, Fix $ c e1 e2)) prod
            in Map.fromList val
        reifySTree' :: IExpr (Map (Vec.Vector Int) ITerm) -> Map (Vec.Vector Int) ITerm
        reifySTree' (I i n) = Map.singleton (Vec.generate n id) (Fix $ I i n)
        reifySTree' (IConst i) = Map.singleton (Vec.fromList [i]) (Fix $ IConst i)
        reifySTree' (Add a b) = interpT (+) Add a b
        reifySTree' (Sub a b) = interpT (-) Sub a b
        reifySTree' (Mul a b) = interpT (*) Mul a b
        reifySTree' (Div a b) = interpT div Div a b
        reifySTree' (Mod a b) = interpT mod Mod a b


