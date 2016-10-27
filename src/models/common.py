# 2016 Tyler Hall

""" Definitions of Digits, operators and characters

__all__ = (
    'DIGITS',
    'LLETTERS',
    'OPERATORS'
    'ULETTERS',
    'LETTERS',
    'ALL',
)

DIGITS = "0123456789"
OPERATORS = "+-/*"          # Will probably need to add more later
LLETTERS = "abcdefghijklmnopqrstuvwxyz"
ULETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LETTERS = LLETTERS + ULETTERS
ALL = DIGITS + OPERATORS +LLETTERS + ULETTERS
