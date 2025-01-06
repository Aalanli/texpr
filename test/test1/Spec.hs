{-# LANGUAGE OverloadedRecordDot #-}
import TExpr
import TExpr (ITerm, TTerm)
import Data.Text (unpack, pack)

import Control.Monad.Except

matmul :: TTerm -> TTerm -> TEBuilder TTerm
matmul a b = do
    sa <- calcShape a
    sb <- calcShape b
    when (sa /= sb) $ do
        throwError "matmul shape mismatch"
    tcompute 1024 (\i ->
        tcompute 1024 (\j ->
            treduce 1024 (\k -> do
                ai <- tindex a ((i `imul` 1024) `iadd` k)
                bi <- tindex b ((k `imul` 1024) `iadd` j)
                telemwise "add" [ai, bi]
            )))

data Conv2DConfig = Conv2DConfig {
    batch::Int,
    channelIn::Int,
    imHeight::Int,
    imWidth::Int,
    channelOut::Int,
    filterX::Int,
    filterY::Int,
    dilX::Int,
    dilY::Int,
    strideX::Int,
    strideY::Int
}

indexRowMajor :: [Int] -> [ITerm] -> ITerm
indexRowMajor dims coords =
    let strides = tail $ scanr (*) 1 dims
    in foldl iadd (iconst 0) $ zipWith imul coords strides

conv2d :: Conv2DConfig -> TTerm -> TTerm -> TEBuilder TTerm
conv2d c im f = do
    sIm <- calcShape im
    sF <- calcShape f
    let imDims = [c.batch, c.channelIn, c.imHeight, c.imWidth]
    let fDims = [c.channelOut, c.channelIn, c.filterY, c.filterX]
    when (sIm /= product imDims) $ do
        throwError $ pack $ "im shape mismatch, im dims" ++ show sIm ++ "!=" ++ show (product imDims)
    when (sF /= product fDims) $ do
        throwError "filter shape mismatch"
    let p = (c.imHeight - c.dilX * (c.strideX - 1) - 1) `div` c.strideX + 1
    let q = (c.imWidth - c.dilY * (c.strideY - 1) - 1) `div` c.strideY + 1
    tcompute c.batch (\ib ->
        tcompute c.channelOut (\ic ->
            tcompute p (\ip ->
                tcompute q (\iq ->
                    treduce c.channelIn (\jc ->
                        treduce c.filterX (\jx ->
                            treduce c.filterY (\jy -> do
                                imx <- tindex im (indexRowMajor imDims
                                    [ib, jc,
                                     (ip `imul` c.strideX) `iadd` (jx `imul` c.dilX),
                                     (iq `imul` c.strideY) `iadd` (jy `imul` c.dilY)])
                                fx <- tindex f (indexRowMajor fDims [ic, jc, jx, jy])
                                telemwise "mul" [imx, fx]
                            )))))))

progMatmul = do
    a <- tvar (1024 * 1024)
    b <- tvar (1024 * 1024)
    matmul a b

progConv2d = do
    im <- tvar (4 * 64 * 512 * 512)
    f <- tvar (128 * 64 * 3 * 3)
    let conv = Conv2DConfig { 
        batch=4,
        channelIn=64,
        imHeight=512,
        imWidth=512,
        channelOut=128,
        filterX=3,
        filterY=3,
        dilX=1,
        dilY=1,
        strideX=1,
        strideY=1
    }
    conv2d conv im f


pprintProg :: TEBuilder TTerm -> String
pprintProg prog = unpack $ case runTEBuilder prog of
    Right e -> showTTerm e
    Left e -> e


main :: IO ()
main = do
    print ""
    putStrLn $ pprintProg progMatmul
    putStrLn $ pprintProg progConv2d
