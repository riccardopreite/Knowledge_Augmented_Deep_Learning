VARIABLE=$1
grep --perl-regex "[film|genre|run]+[_][A-Z|a-z|0-9]+" $VARIABLE > "reduce_"$VARIABLE
