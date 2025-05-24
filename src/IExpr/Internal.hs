{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
module IExpr.Internal (
    Index, indexBound, indexID,
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

type ITerm = TFix ID IExpr

instance Show ITerm where
    show = unpack . showITerm

pprintID :: Index -> Doc ann
pprintID i = pprintIDSimple i <> "<" <> (pretty . indexBound) i <> ">"

pprintIDSimple :: Index -> Doc ann
pprintIDSimple i = "i" <> (pretty . unwrapID . indexID) i

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
