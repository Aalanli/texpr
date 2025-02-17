{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeFamilies #-}
module IExpr (
    module IExpr,
    module IExpr.Internal,
    parseIExpr
) where

import Prettyprinter
import Prettyprinter.Render.Text

import IExpr.Internal
import IExpr.Parse
import Data.Sequence (Seq, fromList, (<|))
import Data.Foldable
import Util
import Data.Text (unpack)

data IFunc = IFunc {
    ifuncExpr::ITerm,
    ifuncIndices::Seq Index
}

makeIFunc :: IDEnv m => [Int] -> ([ITerm] -> ITerm) -> m IFunc
makeIFunc shape f = do
    inds <- mapM newIndex shape
    return $ IFunc (f (map (Fix . I) inds)) (fromList inds)

pprintIfunc :: IFunc -> Doc ann
pprintIfunc (IFunc it ic) = let 
    args = encloseSep "|" "|" comma (map pprintID (toList ic))
    in args <+> pprintITermSimple it

instance Show IFunc where
    show = unpack . renderStrict . layoutPretty defaultLayoutOptions . pprintIfunc

(.+) :: IDom i => i -> i -> i
a .+ b = iAdd a b
(.-) :: IDom i => i -> i -> i
a .- b = iSub a b
(.*) :: IDom i => i -> i -> i
a .* b = iMul a b
(./) :: IDom i => i -> i -> i
a ./ b = iDiv a b
(.%) :: IDom i => i -> i -> i
a .% b = iMod a b
