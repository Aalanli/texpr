{-# LANGUAGE GeneralisedNewtypeDeriving #-}
module IExpr.Parse where

import Control.Monad.Trans
import Data.Void

import Text.Megaparsec
import Text.Megaparsec.Char
import qualified Text.Megaparsec.Char.Lexer as Lex
import Data.Text ( Text )
import Control.Monad.Identity

import IExpr.Internal
import Util

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

pIndex :: (IDEnv m) => ParserT m Index
pIndex = do
    _ <- symbol "i" *> many numberChar *> symbol "<"
    width <- Lex.decimal
    _ <- symbol ">"
    lift $ newIndex width

iexprFix :: (IDEnv m) => ParserT m ITerm
iexprFix = Fix . IConst <$> lexeme Lex.decimal
    <|> try (Fix . I <$> pIndex)
    <|> try (binop "+" Add)
    <|> try (binop "-" Sub)
    <|> try (binop "*" Mul)
    <|> try (binop "/" Div)
    where
        binop c f = pParens (fmap Fix $ f <$> iexprFix <* symbol c <*> iexprFix )

parseIExpr :: IDEnv m => Text -> m ITerm
parseIExpr t = env
    where 
        env = do
            r <- runParserT iexprFix "" t
            case r of
                Right p -> return p
                Left e -> error (errorBundlePretty e)
