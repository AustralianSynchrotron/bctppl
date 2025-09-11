#!/bin/bash


chkf () {
  if [ ! -e "$1" ] ; then
    echo "ERROR! Non existing \"$2\" path: $1" >&2
    exit 1
  fi
}

chkhdf () {
  if ((  1 != $(tr -dc ':'  <<< "$1" | wc -c)  )) ; then
    echo "Input ($1) must be of form 'hdfFile:hdfContainer'." >&2
    exit 1
  fi
}

wrong_num() {
  opttxt=""
  if [ -n "$3" ] ; then
    opttxt="given by option $3"
  fi
  echo "String \"$1\" $opttxt $2." >&2
  exit 1
}

chknum () {
  if ! printf '%f' "$1" &>/dev/null ; then
    wrong_num "$1" "is not a number" "$2"
  fi
}

chkint () {
  if ! [ "$1" -eq "$1" ] 2>/dev/null ; then
    wrong_num "$1" "is not an integer" "$2"
  fi
}

chkpos () {
  num="$(printf '%f' "$1")"
  if (( $(echo "0 >= $num" | bc -l) )); then
    wrong_num "$1" "is not strictly positive" "$2"
  fi
}

chkNneg () {
  num="$(printf '%f' "$1")"
  if (( $(echo "0 > $num" | bc -l) )); then
    wrong_num "$1" "is negative" "$2"
  fi
}

beverbose=false
execMe() {
  if $beverbose ; then
    echo "Executing:"
    echo "  $1"
  fi
  if [ -e  "$LOGFILE" ] ; then
    echo "$1" >> "$LOGFILE"
  fi
  eval $1
  if (( $? )) ; then
    echo "Exiting after error in following command:." >&2
    echo "  $1" >&2
    exit 1
  fi
}
