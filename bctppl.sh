#!/bin/bash

export EXEPATH="$(dirname "$(realpath "$0")" )"
source "$EXEPATH/commonsource.sh"


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
#  echo "  -C SEG,SEG   Crop stitched image."
  echo "  -r FLOAT     Rotate projections (deg)."
  echo "  -z FLOAT     Obect to detector distance (mm)."
  echo "  -w FLOAT     Wawelength (Angstrom)."
  echo "  -p FLOAT     Pixel size (mum)."
  echo "  -i FLOAT     Delta to beta ratio for phase retreival."
  echo "  -R INT       Width of ring artefact filter."
#  echo "  -i str       Type of gap fill algorithm: NO(default), NS, AT, AM"
#  echo "  -t INT       Test mode: keeps intermediate images for the given projection."
  echo "  -E           Skip stage if it's results are already present."
  echo "  -v           Be verbose to show progress."
  echo "  -h           Prints this help."
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
binn=""
zinn=""
o2d=0
wav=""
d2b=""
pix=""
ring=""
skipExisting=false
allargs=""
#while getopts "b:B:d:D:m:M:a:f:F:e:c:r:z:Z:i:t:hv" opt ; do
while getopts "b:B:d:D:m:a:f:F:e:c:r:z:w:p:i:R:Ehv" opt ; do
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
    #M)  pmask=$OPTARG;;
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
    z)  o2d=$OPTARG
        chknum "$o2d" "-$opt"
        chkpos "$o2d" "-$opt"
        ;;
    w)  wav=$OPTARG
        chknum "$wav" "-$opt"
        chkpos "$wav" "-$opt"
        ;;
    p)  pix=$OPTARG
        chknum "$pix" "-$opt"
        chkpos "$pix" "-$opt"
        ;;
    i)  d2b=$OPTARG
        chknum "$d2b" "-$opt"
        chkpos "$d2b" "-$opt"
        ;;
    R)  ring=$OPTARG
        chkint "$ring" "-$opt"
        chkpos "$ring" "-$opt"
        ;;
    #z)  zinn=$OPTARG
    #    chkint "$zinn" "-$opt"
    #    chkpos "$zinn" "-$opt"
    #    ;;
    #Z)  binn=$OPTARG;;
    #i)  fill="$OPTARG";;
    #t)  testme="$OPTARG";;
    E)  skipExisting=true;;
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

out=""
if [ -n "${2}" ] ; then
  out="${2}/"
fi
LOGFILE="${out}.ppl.log"
echo "" >> "$LOGFILE"
echo "# In \"$PWD\"" >> "$LOGFILE"
echo "# $0 $*" >> "$LOGFILE"

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
  echo "$1" >> "$LOGFILE"
  eval $1
  if (( $? )) ; then
    echo "Exiting after error in following command:." >&2
    echo "  $1" >&2
    exit 1
  fi
}

announceStage() {
  if $beverbose ; then
    echo
    echo "Stage ${1}: ${2}."
  fi
}

needToMake() {
  #echo searching for "$@"
  if $skipExisting && ls "$@" > /dev/null 2> /dev/null ; then
    echo "Found existing $*. Will NOT reproduce."
    return 1
  else
    #echo "Not found" "$@"
    return 0
  fi
}


averageHdf2Tif () {
  if ((  1 == $(tr -dc ':'  <<< "$1" | wc -c)  )) ; then # is hdf
    outtif="$2"
    if needToMake "$outtif" ; then
      execMe "ctas v2v $beverboseO -b ,,0 $1 -o $outtif"
    fi
    echo "$outtif"
  else
    echo "$1"
  fi
}



# create output dir
announceStage 1 "preparing"
if [ -n "$out" ] ; then
  execMe "mkdir -p $out"
fi
bgO=$(averageHdf2Tif "$bgO" "${out}bg_org.tif")
bgS=$(averageHdf2Tif "$bgS" "${out}bg_sft.tif")
dfO=$(averageHdf2Tif "$dfO" "${out}df_org.tif")
dfS=$(averageHdf2Tif "$dfS" "${out}df_sft.tif")


# split into org and sft
announceStage 2 "splitting input into original and shifted components"
splitOut="${out}split_"
if needToMake "${splitOut}mask.tif" "${splitOut}org.hdf" "${splitOut}sft.hdf" ; then
  splitOpt="$beverboseO"
  splitOpt="$splitOpt $( addOpt -b "$bgO" ) "
  splitOpt="$splitOpt $( addOpt -B "$bgS" ) "
  splitOpt="$splitOpt $( addOpt -d "$dfO" ) "
  if [ -n "$dfS" ] ; then
    splitOpt="$splitOpt -D $dfS "
  elif [ -n "$dfO" ] ; then # same DF for org and sft
    splitOpt="$splitOpt -D $dfO "
  fi
  splitOpt="$splitOpt $( addOpt -m "$gmask" ) "
  pmask="$gmask"
  splitOpt="$splitOpt $( addOpt -M "$pmask" ) "
  splitOpt="$splitOpt $( addOpt -c $cropStr ) "
  splitOpt="$splitOpt $( addOpt -r $rotate ) "
  splitOpt="$splitOpt $( addOpt -z $binn ) "
  splitOpt="$splitOpt $( addOpt -Z $zinn ) "
  splitOpt="$splitOpt $( addOpt -i $fill ) "
  execMe "$EXEPATH/split.sh  -f $firstO -F $firstS -e $end $splitOpt $inp $splitOut "
fi


# track the ball
announceStage 2 "tracking for jitter"
trackOpt="$beverboseO"
if [ -n "$gmask" ] || [ -n "$pmask" ] ; then
  trackOpt="$trackOpt -m ${splitOut}mask.tif "
fi
trackOut="${out}track_"
if needToMake "${trackOut}org.dat"  ; then
  announceStage 2.1 "tracking jitter in original set"
  execMe "$EXEPATH/trackme.py ${splitOut}org.hdf:/data -o ${trackOut}org.dat $trackOpt "
fi
if needToMake "${trackOut}sft.dat" ; then
  announceStage 2.2 "tracking jitter in shifted set"
  execMe "$EXEPATH/trackme.py ${splitOut}sft.hdf:/data -o ${trackOut}sft.dat $trackOpt "
fi


# analyze track results
announceStage 2.3 "analyzing jitter tracking"
splitWidth=$( h5ls -rf "${splitOut}org.hdf" | grep "/data" | sed "s:.* \([0-9]*\)}:\1:g" )
ballWidth=$( identify -quiet "$EXEPATH/ball.tif" | cut -d' ' -f 3 |  cut -d'x' -f 1 )
resStr=$( "$EXEPATH/analyzeTrack.sh" -a $ark -w $splitWidth -W $ballWidth "${trackOut}"*.dat )
read amplX amplY shiftX shiftY centdiv <<< "$resStr"
amplX=$(( 2 * amplX ))
amplY=$(( 2 * amplY ))


# align
announceStage 3 "aligning"
alignOpt="$beverboseO"
alignOpt="$alignOpt -S ${amplX},${amplY} -m ${splitOut}mask.tif "
alignCom="$EXEPATH/build/align $alignOpt"
alignOut="${out}align_"
if needToMake "${alignOut}org.hdf" ; then
  announceStage 3.1  "aligning original set"
  execMe "$alignCom ${splitOut}org.hdf:/data -s ${trackOut}org.dat -o ${alignOut}org.hdf:/data"
fi
if needToMake "${alignOut}sft.hdf"  ; then
  announceStage 3.2 "aligning shifted set"
  execMe "$alignCom ${splitOut}sft.hdf:/data -s ${trackOut}sft.dat -o ${alignOut}sft.hdf:/data"
fi


# fill gaps
announceStage 4 "filling gaps"
fillOpt="$beverboseO"
fillCom="$EXEPATH/sinogapme.py $fillOpt"
fillOut="${out}fill_"
if needToMake "${fillOut}org.hdf" ; then
  announceStage 4.1 "filling gaps in original set"
  execMe "$fillCom ${alignOut}org.hdf:/data -m ${alignOut}org_mask.tif ${fillOut}org.hdf:/data"
fi
if needToMake "${fillOut}sft.hdf" ; then
  announceStage 4.2 "filling gaps in shifted set."
  execMe "$fillCom ${alignOut}sft.hdf:/data -m ${alignOut}sft_mask.tif ${fillOut}sft.hdf:/data"
fi


# stitch
announceStage 5  "stitching original and shifted sets"
stitchOut="stitched.hdf"
if needToMake "$stitchOut" ; then
  stitchOpt="$beverboseO"
  stitchOpt="$stitchOpt -f 0 -F 0 -a $ark -s $firstS -g ${shiftX},${shiftY} -c $centdiv"
  stitchOpt="$stitchOpt -m ${fillOut}org_mask.tif "
  execMe "imbl-shift.sh $stitchOpt ${fillOut}org.hdf:/data ${fillOut}sft.hdf:/data ${stitchOut}:/data"
fi


# phase contrast
announceStage 6 "inline phase contrast"
ipcOut="$stitchOut"
if [ -z "$d2b" ] ; then # no IPC
  if $beverbose ; then
    echo "Skipping this stage because no delta to beta ratio provided (-i option)"
  fi
else
  ipcOut="${out}ipc.hdf"
  if needToMake "$ipcOut" ; then
    ipcOpt="$beverboseO"
    ipcOpt="$ipcOpt -e -d $d2b"
    if (( o2d == 0 )) ; then
      echo "No object to detector provided for phase contrast (-z option). Exiting" >&2
      exit 1
    else
      ipcOpt="$ipcOpt -z $o2d"
    fi
    if [ -z "$wav" ] ; then
      echo "No wavelength provided for phase contrast (-w option). Exiting" >&2
      exit 1
    else
      ipcOpt="$ipcOpt -w $wav"
    fi
    if [ -z "$pix" ] ; then
      echo "No pixel size provided for phase contrast (-p option). Exiting" >&2
      exit 1
    else
      ipcOpt="$ipcOpt -r $pix"
    fi
    execMe "ctas ipc $ipcOpt ${stitchOut}:/data -o ${ipcOut}:/data"
  fi
fi


# ring artefact removal
announceStage 7 "ring artefact correction"
ringOut="$ipcOut"
if [ -z "$ring" ] ; then # no ring removal
  if $beverbose ; then
    echo "Skipping this stage because no ring filter size provided (-R option)."
  fi
else
  ringOut="${out}ring.hdf"
  if needToMake "$ringOut" ; then
    ringOpt="$beverboseO"
    execMe "ctas ring $ringOpt -R $ring -o ${ringOut}:/data:y ${ipcOut}:/data:y"
  fi
fi


# CT
announceStage 8 "CT reconstruction"
ctOut="${out}rec.hdf"
if needToMake "$ctOut" ; then
  ctOpt="$beverboseO"
  ctOpt="$ctOpt $( addOpt -r $pix ) "
  ctOpt="$ctOpt $( addOpt -w $wav ) "
  step=$(echo "scale=8 ; 180 / ( $ark - 1 )" | bc )
  execMe "ctas ct $ctOpt -k a -a $step ${ringOut}:/data:y -o ${ctOut}:/data"
fi

if $beverbose ; then
  echo
  echo "All done."
fi



