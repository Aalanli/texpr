{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE GeneralisedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module TExpr (module TExpr) where

import Control.Monad.Except

import Prettyprinter
import Data.Text (Text)
import Data.Map (Map)
import qualified Data.Map as Map


import IExpr
import IExpr.Random
import Control.Monad.Identity (Identity (runIdentity))
import qualified Data.Text as Text
import Prettyprinter.Render.Text (renderStrict)

-- The first argument of Compute must be (TInd (I ID Int))
-- and so on with reduce, let and index.
-- This is for ematching, so we don't need to add a pattern for each ID value
-- (basically any Op that we want to ematch against have to have all arguments be 'e')
data TExpr e =
    Compute e e -- (TInd (I ID N)) expr
    | Reduce e e -- (TInd (I ID N)) expr
    | Let e e e -- let (TVar ID N) = e1 in e2
    | Index e e -- (TVar ID N) (TInd ind)
    | TConst Double Int
    | Elemwise Text [e]
    | TVar ID Int -- TID, Shape
    | TInd ITerm
    deriving (Show, Eq, Functor)

data TExprNorm e = 
    Compute' ID Int e
    | Reduce' ID Int e
    | Let' ID Int e e 
    | Index' ID Int ITerm
    | TConst' Double Int
    | TVar' ID Int
    | Elemwise' Text [e] 
    deriving (Show, Eq, Functor)

type TTerm = Fix TExpr
type TTermNorm = Fix TExprNorm
type TEBuilder = ExceptT Text (TIEnv Identity)

normalizeTExpr :: TTerm -> Maybe TTermNorm
normalizeTExpr (Fix t) = Fix <$> normalizeTExpr' t

normalizeTExpr' :: TExpr (Fix TExpr) -> Maybe (TExprNorm TTermNorm)
normalizeTExpr' (Compute (Fix (TInd (Fix (I i n)))) e) = Compute' i n <$> normalizeTExpr e
normalizeTExpr' (Reduce (Fix (TInd (Fix (I i n)))) e) = Reduce' i n <$> normalizeTExpr e
normalizeTExpr' (Let (Fix (TVar i n)) e1 e2) = Let' i n <$> normalizeTExpr e1 <*> normalizeTExpr e2
normalizeTExpr' (Index (Fix (TVar i n)) (Fix (TInd t))) = Just $ Index' i n t
normalizeTExpr' (TConst d i) = Just $ TConst' d i
normalizeTExpr' (Elemwise t es) = Elemwise' t <$> mapM normalizeTExpr es
normalizeTExpr' (TVar i n) = Just $ TVar' i n
normalizeTExpr' _ = Nothing

runTEBuilder :: TEBuilder t -> Either Text t
runTEBuilder t = runIdentity $ evalTIEnv (runExceptT t)


tvar :: Int -> TEBuilder TTerm
tvar n = Fix . (`TVar` n)  <$> lift newID

tind :: Int -> TEBuilder ITerm
tind n = lift $ ivar n


tcompute :: Int -> (ITerm -> TEBuilder TTerm) -> TEBuilder TTerm
tcompute n f = do
    ind <- tind n
    Fix . Compute (Fix . TInd $ ind) <$> f ind

treduce :: Int -> (ITerm -> TEBuilder TTerm) -> TEBuilder TTerm
treduce n f = do
    ind <- tind n
    Fix . Reduce (Fix . TInd $ ind) <$> f ind

tlet :: TTerm -> (TTerm -> TEBuilder TTerm) -> TEBuilder TTerm
tlet e1 f = do
    i <- calcShape e1
    tv <- tvar i
    Fix . Let tv e1 <$> f tv


tindex :: TTerm -> ITerm -> TEBuilder TTerm
tindex v idx = return $ Fix (Index v (Fix $ TInd idx))

tconst :: Double -> Int -> TEBuilder TTerm
tconst v shape = return $ Fix (TConst v shape)

telemwise :: Text -> [TTerm] -> TEBuilder TTerm
telemwise op es = do
    es' <- mapM calcShape es
    when (null es || any (/=1) es') $ do
        throwError "elemwise shapes should be 1"
    return $ Fix (Elemwise op es)


calcShape :: MonadError Text m => TTerm -> m Int
calcShape t = calcShape' $ unFix t

calcShape' :: MonadError Text m => TExpr TTerm -> m Int
calcShape' (Compute (Fix e1) e) = case e1 of
    TInd (Fix (I _ s)) -> (*s) <$> calcShape e
    _ -> throwError $ Text.pack $ "expected constant compute shape (I ID N), found " ++ show e1
calcShape' (Reduce _ e) = calcShape e
calcShape' (Index _ _) = return 1
calcShape' (Let t e1 e2) = do 
    sh <- calcShape e2
    se1 <- calcShape e1
    st <- calcShape t
    when (sh /= se1 || sh /= st) $ do
        throwError "let shape mismatch"
    return sh
calcShape' (TConst _ s) = return s
calcShape' (Elemwise _ es) = do
    s <- mapM calcShape es
    when (null s || any (/=head s) s) $ do
        throwError "elemwise shape mismatch"
    return (head s)
calcShape' (TVar _ s) = return s
calcShape' (TInd _) = return 1

pprintTExpr :: TExpr (Doc ann) -> Doc ann
pprintTExpr (Compute i e) = align $ sep ["compute", "|" <> i <> "|"] <> line <> indent 2 e
pprintTExpr (Reduce i e)  = align $ sep ["reduce",  "|" <> i <> "|"] <> line <> indent 2 e
pprintTExpr (Let i e1 e2) = hang 2 $ sep ["let", i, "=", e1, "in", e2]
pprintTExpr (Index i it) = hcat [i, "[", it, "]"]
pprintTExpr (TConst c i) = sep ["const", "n=" <> pretty i, pretty c]
pprintTExpr (Elemwise op c) = pretty op <> encloseSep lparen rparen comma c <> line
pprintTExpr (TVar (ID tid) n) = "x" <> pretty tid <> "<" <> pretty n <> ">"
pprintTExpr (TInd (Fix (I i n))) = hcat [pprintID i, "<", pretty n, ">"]
pprintTExpr (TInd i) = pprintITerm i

pprintTTerm :: Fix TExpr -> Doc ann
pprintTTerm = cata pprintTExpr

showTTerm :: Fix TExpr -> Text
showTTerm = renderStrict . layoutPretty defaultLayoutOptions . pprintTTerm
