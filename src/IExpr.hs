{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE GeneralisedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module IExpr (
    module IExpr,
    module IExpr.Internal,
    module IExpr.Parse,
    module IExpr.Random
) where

import Control.Monad 

import Data.Map (Map)
import qualified Data.Map as Map
import Data.Set (Set)
import qualified Data.Set as Set
import qualified Data.IntMap as IMap
import Data.Maybe (fromMaybe, mapMaybe)

import Data.Equality.Analysis
import Data.Equality.Graph
import Data.Equality.Graph.Lens
import Data.Equality.Matching
import DynRewrite

import IExpr.Internal
import IExpr.Parse
import IExpr.Random

import qualified Data.Vector.Unboxed as Vec
import Data.Vector.Unboxed ((!))

import DynESat
import Data.List (sort)
import Control.Monad.Identity (Identity(runIdentity))

data IFunc = IFunc {
    ifuncExpr::ITerm,
    ifuncBoundIDs::[ID]
}

makeIFunc :: Int -> ([ID] -> ITerm) -> IFunc
makeIFunc nIDs f = runIdentity $ evalTIEnv $ do
    xs <- replicateM nIDs newID
    return $ IFunc (f xs) xs

type BoundMap = Map ID (Int, Int)
data IExprBounds = IExprBounds {
    iexprBound::(Int, Int),
    iexprEvalMap::Vec.Vector Int
} deriving (Show, Eq, Ord)


instance Analysis IExprBounds IExpr where
    makeA = \case
        I _ n -> IExprBounds (0, n - 1) (Vec.generate n id)
        Add a b -> computeBounds (+) a b
        Sub a b -> computeBounds (-) a b
        Mul a b -> computeBounds (*) a b
        Div a b -> computeBounds div a b
        Mod x@(IExprBounds (a0, a1) ia) y@(IExprBounds (b0, b1) ib) ->
            let ia' = Vec.map (`mod` b1) ia
                iy = Vec.zipWith mod ia ib
            in if b0 == b1 then (IExprBounds (a0, min a1 b1) ia') else (IExprBounds (Vec.minimum iy, Vec.maximum iy) iy)
        IConst i -> IExprBounds (i, i) Vec.empty
        where
            computeBounds f x@(IExprBounds (a0, a1) ia) y@(IExprBounds (b0, b1) ib)
                | a0 == a1 =  -- const
                    IExprBounds (spair f1 f2) $ if f1 == f2 then Vec.empty else Vec.map (f a0) ib
                | b0 == b1 = computeBounds (flip f) y x
                | otherwise = IExprBounds (spair lc hc) c
                where
                    c = Vec.zipWith f ia ib
                    lc = Vec.minimum c
                    hc = Vec.maximum c
                    f1 = f a0 b0
                    f2 = f a1 b1
            spair a b = (min a b, max a b)
    joinA a b = if a == b then a else error ("invalid analysis, left " ++ show a ++ " right " ++ show b)
    modifyA c eg
        | l == h = let (c', eg') = represent (iconst l) eg
            in snd $ merge c c' eg'
        | not (null nconsts) = head nconsts
        | otherwise = eg
        where
            IExprBounds (l, h) _ = eg ^. _class c ._data
            nconsts = mapMaybe (mergeNode . unNode) $ Set.toList $ eg ^. _class c . _nodes
            mergeNode = \case
                Add a b -> mergeConst (+) a b
                Sub a b -> mergeConst (-) a b
                Mul a b -> mergeConst (*) a b
                Div a b -> mergeConst div a b
                Mod a b -> mergeConst mod a b
                _ -> Nothing
            mergeConst f a b = case (isConstIExpr a eg, isConstIExpr b eg) of
                (Just ca, Just cb) -> let (c', eg') = represent (iconst (f ca cb)) eg
                    in Just $ snd $ merge c c' eg'
                _ -> Nothing

isConstIExpr :: Int -> EGraph IExprBounds l -> Maybe Int
isConstIExpr c eg
    | l == h = Just l
    | otherwise = Nothing
    where IExprBounds (l, h) _ = eg ^. _class c ._data

-- testSimplify t = do
--     let e = makeIExpr t
--     let inv = invertTerm e
--     print (fmap pprintITerm inv)
--     print (makeAnalysis e)
--     print (fmap makeAnalysis inv)

iexprcost :: IExpr Int -> Int
iexprcost = \case
    IConst _ -> 1
    I _ _ -> 1
    Add a b -> a + b + 1
    Sub a b -> a + b + 1
    Mul a b -> a + b + 1
    Div a b -> a + b + 1
    Mod a b -> a + b + 1


iexprSimpleRewrites :: [Rewrite IExprBounds IExpr]
iexprSimpleRewrites = concatMap (\(commute, a, b) -> let pa = parsePat a; pb = parsePat b in
    if commute then [pa := pb, pb := pa] else [pa := pb])
    [ (True, "(a + (b + c))", "((a + b) + c)")
    , (True, "(a - (b - c))", "((a - b) + c)")
    , (True, "(a - (b + c))", "((a - b) - c)")
    , (False, "(a + b)", "(b + a)")
    , (False, "(a * b)", "(b * a)")
    , (True, "(a * (b + c))", "((a * b) + (a * c))")
    , (True, "(a * (b + c))", "((a * b) + (a * c))")
    , (False, "(a * (b / c))", "((a * b) / c)")
    , (False, "(a * 1)", "a")
    , (False, "(a * 0)", "0")
    , (False, "(a + 0)", "a")
    , (False, "(a % 1)", "0")
    , (False, "(a / 1)", "a")
    , (False, "(a / a)", "1")
    , (True, "((a / b) / c)", "(a / (b * c))")
    ]

simplifyIExpr :: Fix IExpr -> Fix IExpr
simplifyIExpr e = fst (equalitySaturation e iexprSimpleRewrites iexprcost)

data AffineExpr = AffineExpr {
    rbounds::Map ID Int, -- input ids -> [0, size)
    affineTerm::ITerm
}

makeAnalysis :: ITerm -> IExprBounds
makeAnalysis = cata makeA

invertTerm :: ITerm -> Maybe ITerm
invertTerm e = if sort (Vec.toList v) /= [0..h] then Nothing else
        Just $ applyF substSelf e cycleLCM
        -- Just cycleLCM
    where
        IExprBounds (l, h) v = cata makeA e
        graph = IMap.fromList (zip [0..] $ Vec.toList v)
        computeCycles g i
            | IMap.null g = []
            | units == g = computeCycles g' (i + 1)
            | otherwise = i : computeCycles g' (i + 1)
            where
                units = IMap.filterWithKey (/=) g
                g' = IMap.map (v!) units
        cycleLCM = foldr lcm 1 (computeCycles graph 1) :: Int

applyF :: Integral a => (t -> t -> t) -> t -> a -> t
applyF f x n
    | n <= 1 = x
    | even n = f half half
    | otherwise = f (f half half) x
    where half = applyF f x (n `div` 2)

substSelf :: Fix IExpr -> Fix IExpr -> Fix IExpr
substSelf x (Fix (I _ _)) = x
substSelf x (Fix e') = Fix (fmap (substSelf x) e')


collectIDs :: ITerm -> Set ID
collectIDs (Fix iterm) = foldr (Set.union . collectIDs) Set.empty iterm

isValidAffineExpr :: AffineExpr -> Bool
isValidAffineExpr (AffineExpr { rbounds=rb, affineTerm=at }) = let
    validBound = foldr ((&&) . (>0)) True rb
    ids = collectIDs at
    in validBound && Map.keysSet rb == ids

subst :: Map ID ITerm -> ITerm -> ITerm
subst mp = subst'
    where
        subst' e@(Fix (I i _)) = fromMaybe e (Map.lookup i mp)
        subst' (Fix e) = Fix (fmap subst' e)

