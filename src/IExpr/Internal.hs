{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
module IExpr.Internal (
    Index, indexBound, indexID,
    IDom (..),
    IExpr (..), ITerm,
    newIndex,
    pprintID, pprintIDSimple,
    pprintIExpr, pprintITerm, pprintITermSimple,
    showITerm, showITermSimple
) where

import Util

import Prettyprinter
import Prettyprinter.Render.Text
import Data.Text (Text, unpack)

class IDom i where
    index :: IDEnv f => Int -> f i
    iConst :: Int -> i
    iAdd :: i -> i -> i
    iSub :: i -> i -> i
    iMul :: i -> i -> i
    iDiv :: i -> i -> i
    iMod :: i -> i -> i
    iMod a b = a `iSub` ((a `iDiv` b) `iMul` b)



instance IDom ITerm where
    index b = Fix . I <$> newIndex b
    iConst = Fix . IConst
    iAdd a b = Fix (Add a b)
    iSub a b = Fix (Sub a b)
    iMul a b = Fix (Mul a b)
    iDiv a b = Fix (Div a b)

    
data Index = Index 
    { indexBound::Int
    , indexID::ID 
    } deriving (Eq, Ord)

instance Show Index where
    show = unpack . renderStrict . layoutPretty defaultLayoutOptions . pprintID

newIndex :: IDEnv f => Int -> f Index
newIndex bound = Index bound <$> newID

data IExpr i =
    I Index
    | Add i i
    | Sub i i
    | Mul i i
    | Div i i
    | IConst Int deriving (Show, Eq, Ord, Functor, Foldable, Traversable)

type ITerm = Fix IExpr

instance Show ITerm where
    show = unpack . showITerm

pprintID :: Index -> Doc ann
pprintID i = pprintIDSimple i <> "<" <> (pretty . indexBound) i <> ">"

pprintIDSimple :: Index -> Doc ann
pprintIDSimple i = "i" <> (pretty . idInt . indexID) i

pprintIExpr :: IExpr (Doc ann) -> Doc ann
pprintIExpr (I i) = pprintID i
pprintIExpr (Add i1 i2) = hcat ["(", i1, "+", i2, ")"]
pprintIExpr (Sub i1 i2) = hcat ["(", i1, "-", i2, ")"]
pprintIExpr (Mul i1 i2) = hcat ["(", i1, "*", i2, ")"]
pprintIExpr (Div i1 i2) = hcat ["(", i1, "/", i2, ")"]
-- pprintIExpr (Mod i1 i2) = hcat ["(", i1, "%", i2, ")"]
pprintIExpr (IConst i) = pretty i


pprintITerm :: ITerm -> Doc ann
pprintITerm = cata pprintIExpr

pprintITermSimple :: ITerm -> Doc ann
pprintITermSimple = cata $ \case
    (I i) -> pprintIDSimple i
    o -> pprintIExpr o

showITerm :: ITerm -> Text
showITerm t = renderStrict . layoutPretty defaultLayoutOptions $ pprintITerm t

showITermSimple :: ITerm -> Text
showITermSimple t = renderStrict . layoutPretty defaultLayoutOptions $ pprintITermSimple t
