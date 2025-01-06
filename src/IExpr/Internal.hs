{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE GeneralisedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module IExpr.Internal (
    Fix (..), cata,
    ID (..), IExpr (..), ITerm,
    TIEnv (..), newID, evalTIEnv,
    pprintID,
    pprintIExpr, pprintITerm,
    showITerm,
    iconst, iadd, isub, imul, idiv, imod, ivar
) where

import Control.Monad.State
import Data.Equality.Utils ( Fix(..), cata )
import Prettyprinter
import Prettyprinter.Render.Text
import Data.Text (Text)

newtype TIEnv m a = TIEnv (StateT Int m a)
    deriving (Functor, Applicative, Monad, MonadTrans)

evalTIEnv :: Monad m => TIEnv m a -> m a
evalTIEnv (TIEnv env) = evalStateT env 0

newID :: (Monad m) => TIEnv m ID
newID = TIEnv $ do
    env <- get
    put (env + 1)
    return (ID env)

newtype ID = ID Int deriving (Show, Eq, Ord)

data IExpr i =
    I ID Int -- in [0, n)
    | Add i i
    | Sub i i
    | Mul i i
    | Div i i
    | Mod i i
    | IConst Int deriving (Show, Eq, Ord, Functor, Foldable, Traversable)

type ITerm = Fix IExpr

iexprInterp (I _ _) = \_ _ -> 0
iexprInterp (IConst i) = \_ _ -> i
iexprInterp (Add _ _) = (+)
iexprInterp (Sub _ _) = (-)
iexprInterp (Mul _ _) = (*)
iexprInterp (Div _ _) = div
iexprInterp (Mod _ _) = mod


pprintID :: ID -> Doc ann
pprintID (ID i) = "i" <> pretty i

pprintIExpr :: IExpr (Doc ann) -> Doc ann
pprintIExpr (I i _) = pprintID i
pprintIExpr (Add i1 i2) = hcat ["(", i1, "+", i2, ")"]
pprintIExpr (Sub i1 i2) = hcat ["(", i1, "-", i2, ")"]
pprintIExpr (Mul i1 i2) = hcat ["(", i1, "*", i2, ")"]
pprintIExpr (Div i1 i2) = hcat ["(", i1, "/", i2, ")"]
pprintIExpr (Mod i1 i2) = hcat ["(", i1, "%", i2, ")"]
pprintIExpr (IConst i) = pretty i

pprintITerm :: Fix IExpr -> Doc ann
pprintITerm = cata pprintIExpr

showITerm :: ITerm -> Text
showITerm t = renderStrict . layoutPretty defaultLayoutOptions $ pprintITerm t

iconst :: Int -> ITerm
iconst i = Fix (IConst i)
iadd :: ITerm -> ITerm -> ITerm
iadd a b = Fix (Add a b)
isub :: ITerm -> ITerm -> ITerm
isub a b = Fix (Sub a b)
imul :: ITerm -> Int -> ITerm
imul a b = Fix (Mul a (iconst b))
idiv :: ITerm -> Int -> ITerm
idiv a b = Fix (Div a (iconst b))
imod :: ITerm -> Int -> ITerm
imod a b = Fix (Mod a (iconst b))
ivar :: (Monad m) => Int -> TIEnv m ITerm
ivar b = do
    i <- newID 
    return $ Fix (I i b)

