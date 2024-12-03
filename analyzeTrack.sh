#!/bin/bash

EXEPATH="$(dirname "$(realpath "$0")" )"
PATH="$EXEPATH:$PATH"
source "$EXEPATH/commonsource.sh"

printhelp() {
  echo "Usage: $0 -a <proj> -w <image width> -W <template width> <tracked org> <tracked sft>"
  echo "  Calculates extremes of jitter and stitch parameters."
  echo "OPTIONS:"
  echo "  -a INT       Number of steps to cover 180deg ark."
  echo "  -w INT       Width of the image where the ball was tracked."
  echo "  -W INT       Width of the ball template."
  echo "  -h           Prints this help."
}

ark=0
kwidth=0
iwidth=0
while getopts "a:w:W:h" opt ; do
  allargs=" $allargs -$opt $OPTARG"
  case $opt in
    a)  ark=$OPTARG
        chkint "$ark" "-$opt"
        chkpos "$ark" "-$opt"
        ;;
    w)  iwidth=$OPTARG
        chkint "$iwidth" "-$opt"
        chkpos "$iwidth" "-$opt"
        ;;
    W)  kwidth=$OPTARG
        chkint "$kwidth" "-$opt"
        chkpos "$kwidth" "-$opt"
        ;;
    h)  printhelp ; exit 1 ;;
    \?) echo "ERROR! Invalid option: -$OPTARG" >&2 ; exit 1 ;;
    :)  echo "ERROR! Option -$OPTARG requires an argument." >&2 ; exit 1 ;;
  esac
done
shift $((OPTIND-1))


if [ -z "${1}" ] ; then
  echo "No input path for tracked data in original position was given (argument 1)." >&2
  printhelp >&2
  exit 1
fi
chkf "$1"
orgLines=$(cat  "$1" | grep -c "")

if [ -z "${2}" ] ; then
  echo "No input path for tracked data in shifted position was given (argument 2)." >&2
  printhelp >&2
  exit 1
fi
chkf "$2"
sftLines=$(cat  "$2" | grep -c "")

if (( $orgLines != $sftLines )) ; then
  echo "Different number of lines in original and shifted data: $orgLines != $sftLines." >&2
  printhelp >&2
  exit 1
fi

if (( $ark == 0 )) ; then
  ark=$(( orgLines - 1 ))
fi
if (( $ark <4 )) ; then
  echo "Number of steps in acquisition is too small: $ark < 4." >&2
  printhelp >&2
  exit 1
fi

if (( $iwidth == 0 )) ; then
  echo "No image width was provided (-w option)." >&2
  printhelp >&2
  exit 1
fi
if (( $iwidth < 4 )) ; then
  echo "Width of the image is too small to make sense: $iwidth < 4." >&2
  printhelp >&2
  exit 1
fi


if (( $kwidth == 0 )) ; then
  echo "No ball template width was provided (-W option)." >&2
  printhelp >&2
  exit 1
fi
if (( $kwidth < 2 )) ; then
  echo "Width of the ball template is too small to make sense: $kwidth < 2." >&2
  printhelp >&2
  exit 1
fi
if (( $kwidth >= $iwidth )) ; then
  echo "Width of the ball template cannot be larger than image width: $kwidth >= $iwidth 2." >&2
  printhelp >&2
  exit 1
fi



minY=$( cat "$1" | cut -d' ' -f 1 | sort -g | head -n 1)
maxY=$( cat "$1" | cut -d' ' -f 1 | sort -g | tail -n 1)
#amplY=$(( -minY > maxY ? -minY : maxY ))
amplY=$(( maxY - minY ))
minX=$( cat "$1" | cut -d' ' -f 2 | sort -g | head -n 1)
maxX=$( cat "$1" | cut -d' ' -f 2 | sort -g | tail -n 1)
#amplX=$(( -minX > maxX ? -minX : maxX ))
amplX=$(( maxX - minX ))

read corY corX0 posY posX0 <<< $(cat "$1" | head -n 1)
posX0=$(( posX0 - corX0 ))
read corY corXP posY posXP <<< $(cat "$1" | head -n $ark | tail -n 1)
posXP=$(( posXP - corXP ))
cent_org=$( echo " scale=1 ; ( $posX0 + $posXP + $kwidth ) / 2 " | bc )
posY_org=$(( posY - corY ))

read corY corX0 posY posX0 <<< $(cat "$2" | head -n 1)
posX0=$(( posX0 - corX0 ))
read corY corXP posY posXP <<< $(cat "$2" | head -n $ark | tail -n 1)
posXP=$(( posXP - corXP ))
cent_sft=$( echo " scale=1 ; ( $posX0 + $posXP + $kwidth ) / 2 " | bc )
posY_sft=$(( posY - corY ))

shiftX=$( echo " scale=0 ; ( $cent_org - $cent_sft ) / 1 " | bc )
shiftY=$(( posY_org - posY_sft ))

centdiv=$( echo " scale=1 ; ( $cent_org - 0.5 * $iwidth ) " | bc )

echo $amplX $amplY $shiftX $shiftY $centdiv



