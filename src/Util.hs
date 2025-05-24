{-# LANGUAGE GeneralisedNewtypeDeriving #-}
{-# LANGUAGE FlexibleContexts #-}
module Util (
    IDEnv (..), ID, unwrapID, cata, 
    SimpleIDEnv, runSimpleIDEnv,
    showPPrint, 
    
    -- HFix(..), newHashFix, fixToHFix,
    TFix(..),
    idFix,

    ilog2
) where

import Control.Monad.State
-- import Data.Hashable
import Prettyprinter
import Prettyprinter.Render.String

-- pretty printing
showPPrint :: Doc ann -> String
showPPrint = renderString . layoutPretty defaultLayoutOptions

---------
-- id generation
---------

newtype ID = ID Int deriving (Show, Eq, Ord)

class (Monad m) => IDEnv m where
    newID :: m ID

unwrapID :: ID -> Int
unwrapID (ID i) = i

newtype SimpleIDEnv a = SimpleIDEnv (State Int a)
    deriving (Functor, Applicative, Monad)

instance IDEnv SimpleIDEnv where
    newID = SimpleIDEnv $ do
        i <- get
        put (i + 1)
        return (ID i)

runSimpleIDEnv :: SimpleIDEnv a -> a
runSimpleIDEnv (SimpleIDEnv s) = evalState s 0

---------
-- fix
---------
data TFix t f = TFix { unTFix::f (TFix t f), fixAttr::t }

idFix :: IDEnv m => m (f (TFix ID f) -> TFix ID f)
idFix = flip TFix <$> newID

-- data HFix f = HFix { unHFix::f (HFix f), hashOfFix::Int }

-- newHashFix :: (Hashable (f (HFix f))) => f (HFix f) -> HFix f
-- newHashFix x = HFix x (hash x)

-- fixToHFix :: (Functor f, Hashable (f (HFix f))) => Fix f -> HFix f
-- fixToHFix = cata newHashFix


cata :: Functor f => (f b -> b) -> TFix t f -> b
cata f (TFix g _) = f (fmap (cata f) g)


-- misc


-- the minimum n = ilog2 k such that k / 2^n == 0
ilog2 :: Int -> Int
ilog2 i
    | i <= 0 = 0
    | otherwise = ilog2' 0 i 
    where 
        ilog2' n 0 = n
        ilog2' n j = ilog2' (n + 1) (j `div` 2)
