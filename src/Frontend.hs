module Frontend where

import Data.Text (Text)
import Data.Map (Map)
import IExpr

data AttrItem = AttrDict (Map Text AttrItem) | AttrList [AttrItem] | AttrInt Int | AttrBool Bool | AttrStr Text
    deriving (Eq, Ord)

newtype Attr = Attr (Map Text AttrItem) deriving (Eq, Ord)

data Tensor = Tensor {
    tensorId::Int,
    shape::[Int],
    tensorAttrs::Attr
}

data TEAst = Compute [Int] [ITerm] Attr TEAst
    | Reduce [Int] [ITerm] Attr TEAst
    | Let Tensor TEAst TEAst
    | Index Tensor 
