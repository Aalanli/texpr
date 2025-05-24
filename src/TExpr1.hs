{-# LANGUAGE DeriveTraversable #-}
module TExpr1 where

import Data.Sequence (Seq)
import Util
import IExpr
import Data.Text (Text)

data TVal = TVal 
    { tensorId::ID
    , tensorShape:: Seq Int
    , tensorStride::Seq Int }

data TExpr1 e = 
    Compute e (Seq IFunc)  -- write back function [N] -> [N]
    | Reduce e (Seq IFunc) -- write back function
    | Let TVal e e
    | Index TVal (Seq ITerm)
    | Elemwise Text (Seq ITerm)
    | Constant Float
    | Tuple (Seq e)
    deriving (Functor, Foldable, Traversable)


