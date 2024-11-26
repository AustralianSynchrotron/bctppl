#!/bin/bash

EXEPATH="$(dirname "$(realpath "$0")" )"



printhelp() {
  echo "Usage: $0 [OPTIONS] <input hdf> <output dir>"
  echo "  BCT processing pipeline."
  echo "OPTIONS:"
  echo "  -b PATH      Background image in original position."
  echo "  -B PATH      Background image in shifted position."
  echo "  -d PATH      Dark field image in original position."
  echo "  -D PATH      Dark field image in shifted position."
  echo "  -m PATH      Image containing map of pixels to fill."
#  echo "  -M PATH      Image to process same as input."
  echo "  -a INT       Number of steps to cover 180deg ark."
  echo "  -f INT       First frame in original data set."
  echo "  -F INT       First frame in shifted data set."
  echo "  -e INT       Number of projections to process. Ark+1 if not given."
#  echo "  -z INT       Binning over multiple input prrojections."
#  echo "  -Z INT[,INT] Binn factor(s)."
  echo "  -c SEG,SEG   Crop input image."
  echo "  -r FLOAT     Rotate projections."
#  echo "  -i str       Type of gap fill algorithm: NO(default), NS, AT, AM"
#  echo "  -t INT       Test mode: keeps intermediate images for the given projection."
  echo "  -v           Be verbose to show progress."
  echo "  -h           Prints this help."
}



#printhelp() {
#  echo "Usage: $0 -a <proj> -w <image width> -W <template width> <tracked org> <tracked sft>"
#  echo "  Calculates extremes of jitter and stitch parameters."
#  echo "OPTIONS:"
#  echo "  -a INT       Number of steps to cover 180deg ark."
#  echo "  -w INT       Width of the image where the ball was tracked."
#  echo "  -W INT       Width of the ball template."
#  echo "  -h           Prints this help."
#}

chkf () {
  if [ ! -e "$1" ] ; then
    echo "ERROR! Non existing $2 path: \"$1\"" >&2
    exit 1
  fi
}

wrong_num() {
  opttxt=""
  if [ -n "$3" ] ; then
    opttxt="given by option $3"
  fi
  echo "String \"$1\" $opttxt $2." >&2
  printhelp >&2
  exit 1
}

chknum () {
  if ! (( $( echo " $1 == $1 " | bc -l 2>/dev/null ) )) ; then
    wrong_num "$1" "is not a number" "$2"
  fi
}

chkint () {
  if ! [ "$1" -eq "$1" ] 2>/dev/null ; then
    wrong_num "$1" "is not an integer" "$2"
  fi
}

chkpos () {
  if (( $(echo "0 >= $1" | bc -l) )); then
    wrong_num "$1" "is not strictly positive" "$2"
  fi
}

chkNneg () {
  if (( $(echo "0 > $1" | bc -l) )); then
    wrong_num "$1" "is negative" "$2"
  fi
}


ark=""
bgO=""
bgS=""
dfO=""
dfS=""
gmask=""
pmask=""
cropStr=""
firstO=0
firstS=""
end=""
rotate=""
fill=""
testme=""
beverbose=false
allargs=""
binn=""
zinn=""
while getopts "b:B:d:D:m:M:a:f:F:e:c:r:z:Z:i:t:hv" opt ; do
  allargs=" $allargs -$opt $OPTARG"
  case $opt in
    a)  ark=$OPTARG
        chkint "$ark" "-$opt"
        chkpos "$ark" "-$opt"
        ;;
    b)  bgO=$OPTARG;;
    B)  bgS=$OPTARG;;
    d)  dfO=$OPTARG;;
    D)  dfS=$OPTARG;;
    m)  gmask=$OPTARG;;
    M)  pmask=$OPTARG;;
    f)  firstO=$OPTARG
        chkint "$firstO" "-$opt"
        chkNneg "$firstO" "-$opt"
        ;;
    F)  firstS=$OPTARG
        chkint "$firstS" "-$opt"
        chkNneg "$firstS" "-$opt"
        ;;
    e)  end=$OPTARG
        chkint "$end" "-$opt"
        chkpos "$end" "-$opt"
        ;;
    c)  cropStr=$OPTARG
        ;;
    r)  rotate=$OPTARG
        chknum "$rotate" "-$opt"
        ;;
    z)  zinn=$OPTARG
        chkint "$zinn" "-$opt"
        #chkpos "$zinn" "-$opt"
        ;;
    Z)  binn=$OPTARG;;
    i)  fill="$OPTARG";;
    t)  testme="$OPTARG";;
    v)  beverbose=true;;
    h)  printhelp ; exit 1 ;;
    \?) echo "ERROR! Invalid option: -$OPTARG" >&2 ; exit 1 ;;
    :)  echo "ERROR! Option -$OPTARG requires an argument." >&2 ; exit 1 ;;
  esac
done
shift $((OPTIND-1))


if [ -z "${1}" ] ; then
  echo "No input path was given." >&2
  printhelp >&2
  exit 1
fi
chkhdf "$1"
inp="$1"

if [ -z "${2}" ] ; then
  echo "No output directory was given." >&2
  printhelp >&2
  exit 1
fi
out="$2"

if [ -z "$firstS" ] ; then
  echo "No first frame in shifted data was given (-F)." >&2
  printhelp >&2
  exit 1
fi

if [ -z "$ark" ] ; then
  echo "No 180-deg ark was given (-a option)." >&2
  printhelp >&2
  exit 1
fi
if [ -z "$end" ] ; then
  end=$(( ark + 1 ))
fi
if (( end <= ark )) ; then
  echo "Number of projections to process is less or equal to ark: $end <= $ark." >&2
  printhelp >&2
  exit 1
fi

beverboseO=""
if $beverbose ; then
  beverboseO=" -v "
fi



addOpt() {
  if [ -n "$2" ] ; then
    echo "$1 $2"
  fi
}

execMe() {
  if $beverbose ; then
    echo "Executing:"
    echo "  $1"
  fi
  eval $1
}



# create output dir
execMe "mkdir -p \"$out\""

# split
splitOpt="$beverboseO"
splitOpt="$splitOpt $( addOpt -b \"$bgO\" ) "
splitOpt="$splitOpt $( addOpt -B \"$bgS\" ) "
splitOpt="$splitOpt $( addOpt -d \"$dfO\" ) "
if [ -n "$dfS" ] ; then
  splitOpt="$splitOpt -D \"$dfS\" "
elif [ -n "$dfO" ] ; then # same DF for org and sft
  splitOpt="$splitOpt -D \"$dfO\" "
fi
splitOpt="$splitOpt $( addOpt -m \"$gmask\" ) "
splitOpt="$splitOpt $( addOpt -M \"$pmask\" ) "
splitOpt="$splitOpt $( addOpt -c $cropStr ) "
splitOpt="$splitOpt $( addOpt -r $rotate ) "
splitOpt="$splitOpt $( addOpt -z $binn ) "
splitOpt="$splitOpt $( addOpt -Z $zinn ) "
splitOpt="$splitOpt $( addOpt -i $fill ) "
splitOut="\"$out\"/split_"
execMe "$EXEPATH/split.sh  -f $firstO -F $firstO -e $end \"$inp\" $splitOpt $splitOut "

# track the ball
trackOpt="$beverboseO"
if [ -n "$gmask" ] || [ -n "$pmask" ] ; then
  trackOpt="$trackOpt -m ${splitOut}mask.tif "
fi
trackOut="\"$out\"/track_"
execMe "$EXEPATH/trackme.py ${splitOut}org.hdf:/data -o ${trackOut}org.dat $trackOpt "
execMe "$EXEPATH/trackme.py ${splitOut}sft.hdf:/data -o ${trackOut}sft.dat $trackOpt "

# analyze track results
splitWidth=$( h5ls -rf "${splitOut}org.hdf" | grep "/data" | sed "s:.* \([0-9]*\)}:\1:g" )
ballWidth=$( identify -quiet "$EXEPATH/ball.tif" | cut -d' ' -f 3 |  cut -d'x' -f 1)
resStr="$(execMe "$EXEPATH/analyzeTrack.sh -a $ark -w $splitWidth -W $ballWidth ${trackOut}*.dat")"
read amplX amplY shiftX shiftY centdiv <<< "$resStr"

# align
































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

shiftX=$( echo " scale=1 ; ( $cent_org - $cent_sft ) " | bc )
shiftY=$(( posY_org - posY_sft ))

centdiv=$( echo " scale=1 ; ( $cent_org - 0.5 * $iwidth ) " | bc )

echo $amplX $amplY $shiftX $shiftY $centdiv



