{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use tuple-section" #-}
module IExpr.Random where

import IExpr.Internal

import Control.Monad.State
import System.Random

type Depth = Int
type TerminalGen s f = s -> (Fix f, s)
data RecGen s f = Tree (f (RecGen s f)) 
    | Terminal (TerminalGen s f)
    | Choice [(Double, RecGen s f)]

liftState :: MonadState s m => (s -> (b, s)) -> m b
liftState f = do
    g <- get
    let (y, g') = f g
    put g'
    return y

runRand :: Int -> State StdGen a -> a
runRand seed m = evalState m (mkStdGen seed)

genTree :: (Traversable f, RandomGen s, MonadState s m) => TerminalGen s f 
    -> Int 
    -> RecGen s f -> m (Fix f)
genTree fallback maxD = genTree' 0
    where
        genTree' d (Tree r) 
            | d >= maxD = liftState fallback
            | otherwise = Fix <$> traverse (genTree' (d + 1)) r
        genTree' _ (Terminal f) = liftState f
        genTree' d (Choice xs) = let 
            probs = map fst xs
            in do
                i <- liftState (categorical probs)
                genTree' d (snd $ xs !! i)

categorical :: RandomGen b => [Double] -> b -> (Int, b)
categorical ps g = let
    cprob = scanl (+) 0.0 ps
    (i, g1) = uniformR (0.0::Double, last cprob) g
    in (length (takeWhile (<i) cprob) - 1, g1)

constTerm :: Fix f -> RecGen s f
constTerm x = Terminal (\s -> (x, s))

pickTerm :: [Fix f] -> RecGen s f
pickTerm xs = Choice (map (\x -> (1.0, constTerm x)) xs)

rconstRandu :: RandomGen s => (Int, Int) -> RecGen s IExpr
rconstRandu (a, b) = Terminal $ \s -> let 
    (i, s') = uniformR (a, b) s
    in (Fix $ IConst i, s')

randITerm :: RandomGen s => RecGen s IExpr -> RecGen s IExpr
randITerm i = Choice [
     (1.0, i)
    ,(1.0, Tree $ Add (randITerms i) (randITerms i)) 
    ,(1.0, Tree $ Sub (randITerms i) (randITerms i)) 
    ,(1.0, Tree $ Mul (randITerms i) (rconstRandu (0, 1024))) 
    ,(1.0, Tree $ Div (randITerms i) (rconstRandu (1, 1024))) 
    ,(1.0, Tree $ Mod (randITerms i) (rconstRandu (1, 1024))) 
    ]

randITerms :: RandomGen s => RecGen s IExpr -> RecGen s IExpr
randITerms i = randITerm i

fallbackTerm :: RandomGen s => TerminalGen s IExpr
fallbackTerm s = let 
    (i, s') = uniformR (1, 1024) s
    in (Fix $ IConst i, s')

