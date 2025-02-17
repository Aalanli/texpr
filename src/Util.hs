{-# LANGUAGE GeneralisedNewtypeDeriving #-}
module Util (
    IDEnv (..), ID, idInt, Fix (..), cata, 
    SimpleIDEnv, runSimpleIDEnv,
    showPPrint) where

import Control.Monad.State
import Prettyprinter
import Prettyprinter.Render.String


showPPrint :: Doc ann -> String
showPPrint = renderString . layoutPretty defaultLayoutOptions

newtype ID = ID Int deriving (Show, Eq, Ord)

class (Monad m) => IDEnv m where
    newID :: m ID

idInt :: ID -> Int
idInt (ID i) = i

newtype Fix f = Fix { unFix::f (Fix f) }

cata :: Functor f => (f b -> b) -> Fix f -> b
cata f (Fix g) = f (fmap (cata f) g)


newtype SimpleIDEnv a = SimpleIDEnv (State Int a)
    deriving (Functor, Applicative, Monad)

instance IDEnv SimpleIDEnv where
    newID = SimpleIDEnv $ do
        i <- get
        put (i + 1)
        return (ID i)

runSimpleIDEnv :: SimpleIDEnv a -> a
runSimpleIDEnv (SimpleIDEnv s) = evalState s 0
