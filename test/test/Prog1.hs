{-# OPTIONS_GHC -Wno-unused-imports #-}
{-# OPTIONS_GHC -Wno-incomplete-uni-patterns #-}
module Prog1 (module Prog1) where

import Test.Tasty
import Test.Tasty.Discover
import Test.Tasty.HUnit
import Test.Tasty.QuickCheck
import Prettyprinter
import Prettyprinter.Render.Text


import TExpr2
import IExpr
import Util (runSimpleIDEnv)


unit_foo :: IO ()
unit_foo = do
    print ("foo unit test"::String)
    1 @?= (1::Int)

unit_PprintIFunc :: IO ()
unit_PprintIFunc = do
    let ifunc = runSimpleIDEnv $ makeIFunc [1, 2] $ \[a, b] ->
            (a .+ b) ./ iConst 3
    putDoc (pprintIfunc ifunc)


prop_additionCommutative :: Int -> Int -> Bool
prop_additionCommutative a b = a + b == b + a
