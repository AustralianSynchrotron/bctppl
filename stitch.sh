#!/bin/bash

EXEPATH="$(dirname "$(realpath "$0")" )"
PATH="$EXEPATH:$PATH"
source "$EXEPATH/commonsource.sh"

printhelp() {
  echo "Usage: $0 [OPTIONS] <original> [shifted] <output>"
  echo "  Performs projection formation in accordance with shift-in-scan approach."
  echo "OPTIONS:"
  echo "  -b PATH      Background image in original position."
  echo "  -B PATH      Background image in shifted position."
  echo "  -d PATH      Dark field image in original position."
  echo "  -D PATH      Dark field image in shifted position."
  echo "  -m PATH      Image containing map of gaps in original position."
  echo "  -M PATH      Image containing map of gaps in shifted position."
  echo "  -a INT       Number of frames constituting 180 degree ark."
  echo "  -f INT       First frame in original data set."
  echo "  -F INT       First frame in shifted data set."
  echo "  -s INT       Position of the first frame in shifted data set relative to that in original."
  echo "  -e INT       Number of last projection to process."
  echo "  -g INT:INT   Spatial shift in pixels (X,Y)."
  echo "  -c FLOAT     Deviation of rotation axis from the center of original image."
  echo "  -C T,L,B,R   Crop final image (all INT). T, L, B and R give cropping from the edges of the"
  echo "               images: top, left, bottom, right."
  echo "  -R FLOAT     Rotate projections."
  echo "  -I str       Type of gap fill algorithm: NO(default), NS, AT, AM"
  echo "  -t INT       Test mode: keeps intermediate images for the given projection."
  echo "  -v           Be verbose to show progress."
  echo "  -h           Prints this help."
}

bgO=""
bgS=""
dfO=""
dfS=""
gmaskOrg=""
gmaskSft=""
cropT=0
cropL=0
cropB=0
cropR=0
shiftX=""
shiftY=""
piark=""
firstO=0
firstS=""
delta=""
end=""
cent=0
rotate=0
fill=""
testme=""
beverbose=false
allargs=""
while getopts "b:B:d:D:m:M:g:f:F:s:e:c:a:C:R:I:t:hv" opt ; do
  allargs=" $allargs -$opt $OPTARG"
  case $opt in
    b)  bgO=$OPTARG;;
    B)  bgS=$OPTARG;;
    d)  dfO=$OPTARG;;
    D)  dfS=$OPTARG;;
    m)  gmaskOrg=$OPTARG;;
    M)  gmaskSft=$OPTARG;;
    g)  IFS=',:' read shiftX shiftY <<< "$OPTARG"
        chkint "$shiftX" " from the string '$OPTARG' determined by option -$opt"
        chkint "$shiftY" " from the string '$OPTARG' determined by option -$opt"
        ;;
    a)  piark=$OPTARG
        chkint "$piark" "-$opt"
        chkpos "$piark" "-$opt"
        if [ "$piark" -lt "3" ] ; then
            wrong_num "$piark" "is less than 3" "-$opt"
        fi
        ;;
    f)  firstO=$OPTARG
        chkint "$firstO" "-$opt"
        chkNneg "$firstO" "-$opt"
        ;;
    F)  firstS=$OPTARG
        chkint "$firstS" "-$opt"
        chkNneg "$firstS" "-$opt"
        ;;
    s)  delta=$OPTARG
        chkint "$delta" "-$opt"
        chkNneg "$delta" "-$opt"
        ;;
    e)  end=$OPTARG
        chkint "$end" "-$opt"
        chkpos "$end" "-$opt"
        ;;
    c)  cent=$OPTARG
        chknum "$cent" "-$opt"
        ;;
    C)  IFS=',:' read cropT cropL cropB cropR <<< "$OPTARG"
        chkint "$cropT" "-$opt"
        chkint "$cropL" "-$opt"
        chkint "$cropB" "-$opt"
        chkint "$cropR" "-$opt"
        chkNneg "$cropT" "-$opt"
        chkNneg "$cropL" "-$opt"
        chkNneg "$cropB" "-$opt"
        chkNneg "$cropR" "-$opt"
        ;;
    R)  rotate=$OPTARG
        chknum "$rotate" "-$opt"
        ;;
    I)  fill="$OPTARG";;
    t)  testme="$OPTARG";;
    v)  beverbose=true;;
    h)  printhelp ; exit 1 ;;
    \?) echo "ERROR! Invalid option: -$OPTARG" >&2 ; exit 1 ;;
    :)  echo "ERROR! Option -$OPTARG requires an argument." >&2 ; exit 1 ;;
  esac
done
shift $((OPTIND-1))

if [ -n "$bgO" ] ; then
  args="$args -B $bgO"
fi
if [ -n "$bgS" ] ; then
  args="$args -B $bgS"
fi
if [ -n "$dfO" ] ; then
  args="$args -D $dfO"
fi
if [ -n "$dfS" ] ; then
  args="$args -D $dfS"
fi
if [ -n "$gmaskOrg" ] ; then
  args="$args -M $gmaskOrg"
fi
if [ -n "$gmaskSft" ] ; then
    args="$args -M $gmaskSft"
fi
if [ -n "$testme" ] ; then
  args="$args -t $testme"
fi
if [ "$rotate" !=  "0" ]  ; then
  args="$args -r $rotate"
fi
if [ -n "$fill" ] ; then
  args="$args -I $fill"
fi

if $beverbose ; then
  args="$args -v"
fi



samO="${1}"
samS=""
outVol=""
if [ -z "${1}" ] ; then # only print shifts
  samO=""
elif [ -z "$2" ] ; then # 1 positional argument
  echo "No output path was given." >&2
  printhelp >&2
  exit 1
elif [ -z "$3" ] ; then # 2 positional arguments
  samS="${1}"
  outVol="${2}"
  chkhdf "$outVol"
  if [ -z "$firstS" ] && [ -z "$delta" ] ; then
    echo "With a single input file either -F or -s options must be provided." >&2
    printhelp >&2
    exit 1
  elif [ -n "$firstS" ] && [ -n "$delta" ] ; then
    echo "With a single input only one of -F and -s options can be provided." >&2
    printhelp >&2
    exit 1
  elif [ -z "$delta" ] ; then
    delta=$(( $firstS - $firstO ))
  elif [ -z "$firstS" ] ; then
    firstS=$(( $firstO + $delta ))
  fi
else # 3 positional arguments
  samS="${2}"
  chkhdf "$2"
  outVol="$3"
  chkhdf "$outVol"
  if [ -z "$delta" ] ; then
    delta=0
  fi
  if [ -z "$firstS" ] ; then
    firstS=0
  fi
fi
if [ -n "$outVol" ] ; then
  chkf "$samO" "original input"
  chkhdf "$samO"
  chkf "$samS" "shifted input"
  chkhdf "$samS"
fi


if [ -z "$piark" ] ; then
  echo "No option -a was given for step angle." >&2
  printhelp >&2
  exit 1
fi
if [ -z "$end" ] ; then
  end="$piark"
fi
if (( $end < $piark )) ; then
  echo "Last projection $end is less than that at 180deg $piark." >&2
  exit 1
fi
delta=$(( $delta % (2*$piark) )) # to make sure it is in [0..360) deg

if [ -z "$shiftX" ] || [ -z "$shiftY" ] ; then
  echo "No option -g was given for spacial shift." >&2
  printhelp >&2
  exit 1
fi


roundToInt() {
  printf "%.0f\n" "$1"
}

abs() {
  echo "${1/#-}"
}

maxNum() {
  echo -e "$1" | tr -d ' ' | sort -n | tail -1
}

minNum() {
  echo -e "$1" | tr -d ' ' | sort -n | head -1
}

centshift=$( roundToInt $(echo " 2 * $cent - $shiftX " | bc) )
norgx=$(  maxNum "0 \n $shiftX \n $centshift" )
norgxD=$( minNum "0 \n $shiftX" )
norgxF=$( minNum "0 \n $centshift" )
nendx=$(  minNum "0 \n $shiftX \n $centshift" )
nendxD=$( maxNum "0 \n $shiftX" )
nendxF=$( maxNum "0 \n $centshift" )

cropTB=$(abs "$shiftY")
cropD="$(($norgx-$norgxD+$cropL))-$(($nendxD-$nendx+$cropR)),$(($cropTB+$cropT))-$(($cropTB+$cropB))"
cropF="$(($norgx-$norgxF+$cropL))-$(($nendxF-$nendx+$cropR)),$(($cropTB+$cropT))-$(($cropTB+$cropB))"
spshD="$shiftX,$shiftY"
spshF="$centshift,$shiftY"
argD="-C $cropD -g $spshD"
argF="-C $cropF -f $spshF"


doStitch() {
  stO=$(($firstO+$1))
  stS=$(($firstS+$2))
  if [ -z "$outVol" ] ; then # empty input: just print shifts
    for (( cnt=0 ; cnt<=$3 ; cnt++ )) ; do
      echo $(( stO + cnt )) $(( stS + cnt )) "$4"
    done
  else
    outStr=""
    if [ -n "$testme" ] ; then
      outStr="T${testme}_O${stO}_S${stS}__${outVol}"
    else
      outStr="$outVol:$1+$3,$end"
    fi
    toExec="ctas proj $args $4 -o $outStr  $samO:${stO}+${3}  $samS:${stS}+${3}"
    if $beverbose ; then
      echo "Executing:"
      echo "  $toExec"
      #echo "  $*"
    fi
    eval $toExec
  fi
}

if (( $delta == 0 )) ; then
  doStitch 0 0 $(( end + 1 )) "$argD"
elif (( $delta <= $piark  )) ; then
  doStitch 0      $(($piark - $delta)) $delta                 "$argF"
  doStitch $delta 0                    $(($end - $delta + 1)) "$argD"
else
  tailO=$(( 2*$piark - $delta ))
  tailS=$(( $end - $tailO ))
  doStitch 0      $tailO               $tailS           "$argD"
  doStitch $tailS $(( $end - $piark )) $(( tailO + 1 )) "$argF"
fi

exit $?







