{-# LANGUAGE GeneralisedNewtypeDeriving #-}
module IExpr.Parse (
    parseIExpr, parsePat
) where

import Control.Monad.Trans
import Data.Void

import Text.Megaparsec
import Text.Megaparsec.Char
import qualified Text.Megaparsec.Char.Lexer as Lex
import Data.Text ( Text )

import Data.Equality.Matching.Pattern
import Data.Equality.Utils
import IExpr.Internal
import Control.Monad.Identity


type ParserT m = ParsecT Void Text m

sc :: ParserT m ()
sc = Lex.space space1 lineCmnt blockCmnt
    where
        lineCmnt  = Lex.skipLineComment "//"
        blockCmnt = Lex.skipBlockComment "/*" "*/"

lexeme :: ParserT m a -> ParserT m a
lexeme = Lex.lexeme sc

symbol :: Text -> ParserT m Text
symbol = Lex.symbol sc

pParens :: ParserT m a -> ParserT m a
pParens = between (symbol "(") (symbol ")")

pVar :: ParserT m (Pattern IExpr)
pVar = (lexeme . try) p
    where
        p = fmap (VariablePattern . hashString) $ (:) <$> letterChar
                <*> many (alphaNumChar <|> single '_')

tVar :: (Monad m) => ParserT (TIEnv m) ITerm
tVar = (lexeme . try) p
    where
        p = do
            _ <- letterChar
            _ <- many (alphaNumChar <|> single '_')
            _ <- sc
            n <- lexeme Lex.decimal
            i <- lift newID
            return (Fix (I i n))

iexprPat :: ParserT m (Pattern IExpr)
iexprPat = pat . IConst <$> lexeme Lex.decimal
    <|> pVar
    <|> try (binop "+" Add)
    <|> try (binop "-" Sub)
    <|> try (binop "*" Mul)
    <|> try (binop "/" Div)
    <|> binop "%" Mod
    where
        binop c f = pParens (fmap pat $ f <$> iexprPat <* symbol c <*> iexprPat )

iexprFix :: (Monad m) => ParserT (TIEnv m) ITerm
iexprFix = Fix . IConst <$> lexeme Lex.decimal
    <|> try tVar
    <|> try (binop "+" Add)
    <|> try (binop "-" Sub)
    <|> try (binop "*" Mul)
    <|> try (binop "/" Div)
    <|> binop "%" Mod
    where
        binop c f = pParens (fmap Fix $ f <$> iexprFix <* symbol c <*> iexprFix )

parsePat :: Text -> Pattern IExpr
parsePat t = case runParser iexprPat "" t of
    Right p -> p
    Left e -> error (errorBundlePretty e)

parseIExpr :: Text -> ITerm
parseIExpr t = runIdentity $ evalTIEnv env
    where 
        env :: TIEnv Identity ITerm
        env = do
            r <- runParserT iexprFix "" t
            case r of
                Right p -> return p
                Left e -> error (errorBundlePretty e)
