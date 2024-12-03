#!/bin/bash

EXEPATH="$(dirname "$(realpath "$0")" )"
PATH="$EXEPATH:$PATH"
source "$EXEPATH/commonsource.sh"

printhelp() {
  echo "Usage: $0 [OPTIONS] <source> <output prefix>"
  echo "  Performs projection formation in accordance with shift-in-scan approach."
  echo "OPTIONS:"
  echo "  -b PATH      Background image in original position."
  echo "  -B PATH      Background image in shifted position."
  echo "  -d PATH      Dark field image in original position."
  echo "  -D PATH      Dark field image in shifted position."
  echo "  -m PATH      Image containing map of pixels to fill."
  echo "  -M PATH      Image to process same as input."
  echo "  -f INT       First frame in original data set."
  echo "  -F INT       First frame in shifted data set."
  echo "  -e INT       Number of projections to process."
  echo "  -z INT       Binning over multiple input prrojections."
  echo "  -Z INT[,INT] Binn factor(s)."
  echo "  -c SEG,SEG   Crop image."
  echo "  -r FLOAT     Rotate projections."
  echo "  -i str       Type of gap fill algorithm: NO(default), NS, AT, AM"
  echo "  -t INT       Test mode: keeps intermediate images for the given projection."
  echo "  -v           Be verbose to show progress."
  echo "  -h           Prints this help."
}

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
rotate=0
fill=""
testme=""
beverbose=false
allargs=""
binn=""
zinn=""
while getopts "b:B:d:D:m:M:f:F:e:c:r:z:Z:i:t:hv" opt ; do
  allargs=" $allargs -$opt $OPTARG"
  case $opt in
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
if [ -z "${2}" ] ; then
  echo "No output prefix was given." >&2
  printhelp >&2
  exit 1
fi
if [ -z "$firstS" ] ; then
  echo "No first frame in shifted data was given (-F)." >&2
  printhelp >&2
  exit 1
fi
if [ -z "$end" ] ; then
  echo "No number of frames to process was given (-e)." >&2
  printhelp >&2
  exit 1
fi

args=" "
if [ -n "$gmask" ] ; then
  args="$args -M $gmask "
fi
if [ -n "$testme" ] ; then
  args="$args -t $testme "
fi
if [ -n "$cropStr" ]  ; then
  args="$args -c $cropStr "
fi
if [ "$rotate" !=  "0" ]  ; then
  args="$args -r $rotate "
fi
if [ -n "$binn" ] ; then
  args="$args -b $binn "
fi
if [ -n "$zinn" ] ; then
  args="$args -z $zinn "
fi
if [ -n "$fill" ] ; then
  args="$args -I $fill "
fi
if $beverbose ; then
  args="$args -v"
fi


# original position
org_args=" $args -o ${2}org.hdf:/data ${1}:${firstO}-$(( firstO + end )) "
if [ -n "$bgO" ] ; then
  org_args="$org_args -B $bgO "
fi
if [ -n "$dfO" ] ; then
  org_args="$org_args -D $dfO "
fi
execMe "ctas proj $org_args"

# shifted position
sft_args=" $args -o ${2}sft.hdf:/data ${1}:${firstS}-$(( firstS + end )) "
if [ -n "$bgS" ] ; then
  sft_args="$sft_args -B $bgS "
fi
if [ -n "$dfS" ] ; then
  sft_args="$sft_args -D $dfS "
fi
execMe "ctas proj $sft_args"

# mask
if [ -n "$pmask" ] ; then
  toExec="ctas v2v -o ${2}mask.tif -i 8 $pmask "
  if [ -n "$cropStr" ]  ; then
    toExec="$toExec -c ${cropStr}, "
  fi
  if [ "$rotate" !=  "0" ]  ; then
    toExec="$toExec -r $rotate "
  fi
  if [ -n "$binn" ] ; then
    IFS=',' read b1 b2 <<< "$binn"
    if [ -z "$b2" ] ; then
      b2="$b1"
    fi
    binnStr="${b1},${b2},1"
    toExec="$toExec -b $binnStr "
  fi
  execMe "$toExec"
fi

exit $?







