VARIABLE=$1
grep --perl-regex "[A-Z|a-z|0-9]+[_][A-Z|a-z|0-9]+[\t][A-Z|a-z|0-9]+[\t][genre|run]+[_][A-Z|a-z|0-9]+" $VARIABLE > "reduce_"$VARIABLE
