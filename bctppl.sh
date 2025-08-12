#!/bin/bash

allOpts="$*"
export EXEPATH="$(dirname "$(realpath "$0")" )"
source "$EXEPATH/commonsource.sh"
LOCALCFG="$EXEPATH/.local.cfg"
if [ -e "$LOCALCFG" ] ; then
  source "$LOCALCFG"
  export CUDA_VISIBLE_DEVICES
fi


# Parse input arguments

printhelp() {
  echo "Usage: $0 [OPTIONS] <input hdf or data directory> <output dir>"
  echo "  BCT processing pipeline."
  echo "OPTIONS:"
  echo "  -b PATH      Background image in original position."
  echo "  -B PATH      Background image in shifted position."
  echo "  -d PATH      Dark field image in original position."
  echo "  -D PATH      Dark field image in shifted position."
  echo "  -m PATH      Image containing map of pixels to fill."
  echo "  -a INT       Number of steps to cover 180deg ark or *PPSstream.txt file to estimate it."
  echo "  -f INT       First frame in original data set."
  echo "  -F INT       First frame in shifted data set."
  echo "  -e INT       Number of projections to process. Ark+1 if not given."
  echo "  -Z INT       Fraction of total input projections (1)."
  echo "  -c SEG,SEG   Crop input image."
  echo "  -r FLOAT     Rotate projections (deg)."
  echo "  -z FLOAT     Obect to detector distance (mm)."
  echo "  -w FLOAT     Wawelength (Angstrom)."
  echo "  -p FLOAT     Pixel size (mum)."
  echo "  -i FLOAT     Delta to beta ratio for phase retreival."
  echo "  -R INT       Width of ring artefact filter."
  echo "  -J           Correct jitter only in vertical axis."
  echo "  -K           Keep iterim files."
  echo "  -P           Keep clean and stiched projections container."
  echo "  -I           Use output folder instead of memory to store interim files."
  echo "  -S INT       Start from given stage."
  echo "  -T INT       Terminate after given stage."
  echo "  -E           Skip stages which have their results already present."
  echo "  -v           Be verbose to show progress."
  echo "  -h           Prints this help."
}

ark=""
bgO=""
bgS=""
dfO=""
dfS=""
gmask=""
cropStr=""
firstO=0
firstS=""
end=""
rotate=""
fill=""
binn=""
zinn=1
o2d=0
wav=""
d2b=""
pix=""
ring=""
jonly=false
skipExisting=false
beverbose=false
cleanup=true
keepClean=false
inplace=false
fromStage=0
termStage=9999
#forcedInp=""
#pplvariant="new"

allargs=""
while getopts "b:B:d:D:m:a:f:F:e:Z:c:r:z:w:p:i:R:IKPJS:T:Ehv" opt ; do
  allargs=" $allargs -$opt $OPTARG"
  case $opt in
    a)  ark=$OPTARG
        #chkint "$ark" "-$opt"
        #chkpos "$ark" "-$opt"
        ;;
    b)  bgO="$bgO $OPTARG";;
    B)  bgS="$bgS $OPTARG";;
    d)  dfO="$dfO $OPTARG";;
    D)  dfS="$dfS $OPTARG";;
    m)  gmask=$OPTARG;;
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
    Z)  zinn=$OPTARG
        chkint "$zinn" "-$opt"
        chkpos "$zinn" "-$opt"
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
        #chkpos "$ring" "-$opt"
        ;;
    I)  inplace=true;;
    J)  jonly=true;;
    S)  fromStage=$OPTARG
        chkint "$fromStage" "-$opt"
        chkpos "$fromStage" "-$opt"
        ;;
    T)  termStage=$OPTARG
        chkint "$termStage" "-$opt"
        chkpos "$termStage" "-$opt"
        ;;
    E)  skipExisting=true;;
    K)  cleanup=false;;
    P)  keepClean=true;;
    v)  beverbose=true;;
    h)  printhelp ; exit 1 ;;
    \?) echo "ERROR! Invalid option: -$OPTARG" >&2 ; exit 1 ;;
    :)  echo "ERROR! Option -$OPTARG requires an argument." >&2 ; exit 1 ;;
  esac
done
shift $((OPTIND-1))


# Check and modify inputs

# Deal with input path
if [ -z "${1}" ] ; then
  echo "No input path was given." >&2
  printhelp >&2
  exit 1
fi
addHDFpath() {
  outLine=""
  for file in $1 ; do
    outLine="$outLine ${file}:${2}"
  done
  echo "$outLine"
}
if [[ -d "$(realpath "${1}" 2> /dev/null)" ]]; then # input is a directory

  hdfEntry="/entry/data/data"
  # sample
  listOfFiles="$(find -L "$1" -maxdepth 1 -iname '*sample*hdf')"
  nofSamples=$(grep "hdf" -c <<< "$listOfFiles")
  if (( ! nofSamples )) ; then
    echo "No sample file(s) found in ${1}." >&2
    exit 1
  fi
  if (( nofSamples > 1 )) ; then
    echo "Error! More than a single sample file found in ${1}:" >&2
    echo "$listOfFiles" >&2
    exit 1
  fi
  inp="${listOfFiles}:${hdfEntry}"
  if $beverbose ; then
    echo "Sample file found: $listOfFiles"
  fi
  # BG org
  if [ -z "$bgO" ] ; then
    listOfFiles="$(find -L "$1" -maxdepth 1 -iname  '*BG*org*hdf')"
    if (( $(grep "hdf" -c <<< "$listOfFiles") )) ; then
      bgO="$(addHDFpath "$listOfFiles" "$hdfEntry")"
    fi
    if $beverbose ; then
      echo "Backgrounds in original position found:" $listOfFiles
    fi
  fi
  # BG sft
  if [ -z "$bgS" ] ; then
    listOfFiles="$(find -L "$1" -maxdepth 1 -iname  '*BG*sft*hdf')"
    if (( $(grep "hdf" -c <<< "$listOfFiles") )) ; then
      bgS="$(addHDFpath "$listOfFiles" "$hdfEntry")"
    fi
    if $beverbose ; then
      echo "Backgrounds in shifted position found:" $listOfFiles
    fi
  fi
  # non-specific DF
  dfC=""
  if [ -z "$dfO" ] || [ -z "$dfS" ] ; then # try non-specific DFs
    listOfFiles="$(find -L "$1" -maxdepth 1 -iname  '*DF*hdf')"
    if (( $(grep "hdf" -c <<< "$listOfFiles") )) ; then
      dfC="$(addHDFpath "$listOfFiles" "$hdfEntry")"
    fi
    if $beverbose ; then
      echo "All dark fields found:" $listOfFiles
    fi
  fi
  # DF org
  if [ -z "$dfO" ] ; then
    listOfFiles="$(find -L "$1" -maxdepth 1 -iname  '*DF*org*hdf')"
    if (( $(grep "hdf" -c <<< "$listOfFiles") )) ; then
      dfO="$(addHDFpath "$listOfFiles" "$hdfEntry")"
    elif [ -n "$dfC" ] ; then
      dfO="$dfC"
    fi
    if $beverbose ; then
      echo "Dark fields in original position found:" $listOfFiles
    fi
  fi
  # DF sft
  if [ -z "$dfS" ] ; then
    listOfFiles="$(find -L "$1" -maxdepth 1 -iname  '*DF*sft*hdf')"
    if (( $(grep "hdf" -c <<< "$listOfFiles") )) ; then
      dfS="$(addHDFpath "$listOfFiles" "$hdfEntry")"
    elif [ -n "$dfC" ] ; then
      dfS="$dfC"
    fi
    if $beverbose ; then
      echo "Dark fields in shifted position found:" $listOfFiles
    fi
  fi
  # PPS stream
  if [ -z "$ark" ] ; then
    listOfFiles="$(find -L "$1" -maxdepth 1 -iname  '*PPSstream.txt')"
    if (( $(grep "txt" -c <<< "$listOfFiles") > 1 )) ; then
      echo "Error! More than a single stream file found:" >&2
      echo "$listOfFiles" >&2
      echo "Choose one of them and use with -a option." >&2
      exit 1
    fi
    if $beverbose ; then
      echo "Stream data found: $listOfFiles"
    fi
    ark="$listOfFiles"
  fi

else
  inp="$1"
fi
chkhdf "$inp"


# Prepare output and log

out=""
outDest="${2}"
if [ -n "${2}" ] ; then
  if [ "${outDest: -4}" == ".hdf" ] ; then
    out="$(dirname "${2}")/"
  else
    out="${2}/"
  fi
  mkdir -p "${out}"
else
  out="./"
  outDest="./"
fi
iout=""
if $inplace ; then
  iout="${out}"
else
  iout="/dev/shm/bctppl/"
  if ! $skipExisting  &&  ! ((fromStage)) ; then
    rm -rf "$iout" # to clean up after any previous left overs
  fi
  mkdir -p "$iout"
fi
LOGFILE="${out}.ppl.log"
EXECRES="${iout}.res"
touch "$LOGFILE"
echo "# In \"$PWD\"" >> "$LOGFILE"
echo "# $(realpath "$0") $allOpts " >> "$LOGFILE"


# Check the rest of input arguments

if [ -z "$firstS" ] ; then
  echo "No first frame in shifted data was given (-F)." >&2
  printhelp >&2
  exit 1
fi

if [ -z "$ark" ] ; then
  echo "Neither 180-deg ark provided (-a option), nor PPS stream file found." >&2
  printhelp >&2
  exit 1
fi
if ! [ "$ark" -eq "$ark" ] 2>/dev/null ; then # not an integer: stream file assumed
  # try to derive it from stram.txt
  streamFile="$ark"
  #cleanStream="00_stream.dat"
  #sedFiler='s:.*FrameNumber=\"\([^\"]*\)\".*Angle=\"\([^\"]*\)\".*:\1 \2:g'
  #cat "$streamFile" | sed "$sedFiler"  > "$cleanStream"
  ark=$("$EXEPATH/analyzeStream.py" "$streamFile")
  chkint "$ark" "$streamFile"
  if $beverbose ; then
    echo "Found 180-deg ark: $ark"
  fi
fi
chkpos "$ark" "-a"
if (( firstO + ark > firstS )) ; then
  echo "Error! First frame in shifted data set $firstS is less than the last frame in original data set $(( firstO + ark ))" >&2
  exit 1
fi

if [ -z "$end" ] ; then
  end=$(( ark + zinn ))
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


# Some functions to be used below

addOpt() {
  if [ -n "$2" ] ; then
    echo "$1 $2"
  fi
}

stage=0
pstage=""
bumpstage() {
  if (( stage >= termStage )) ; then
    echo "Terminated at stage $stage." >&2
    exit 0
  fi
  stage=$(( stage + 1 ))
  pstage="$( printf '%02i' $stage )"
  return $(( stage >= fromStage ))
}

announceStage() {
  if $beverbose ; then
    fstage="$stage"
    message="$1"
    if [ -n "$2" ] ; then
      fstage="${fstage}.${1}"
      message="$2"
    fi
    echo
    echo "Stage ${fstage}: ${message}."
  fi
}


needToMake() {
  if ! $skipExisting ; then
    return 0
  fi
  for file in "$@" ; do
    if [ -e "$file" ] ; then
      if [ $beverbose ] ; then
        echo "Found existing $file. Will not reproduce"
      fi
      continue
    fi
    base="$(basename "$file")"
    if [ -e "${out}/${base}" ] ; then
      if [ $beverbose ] ; then
        echo "Found existing ${out}/${base}. Copying to $file."
      fi
      cp "$beverboseO" "${out}/${base}" "$file"
      continue
    fi
    if [ -e "${iout}/${base}" ] ; then
      if [ $beverbose ] ; then
        echo "Found existing ${iout}/${base}. Copying to $file."
      fi
      cp "$beverboseO" "${iout}/${base}" "$file"
      continue
    fi
    return 0
  done
  return 1
}


averageHdf2Tif () {
  if [ -z "$1" ] ; then
    : > "$EXECRES"
  else
    outtif="$2"
    if needToMake "$outtif" ; then
      execMe "ctas v2v $beverboseO -b ,,0 $1 -o $outtif"
    fi
    echo "$outtif" > "$EXECRES"
  fi
}


cleanUp() {
  if $cleanup ; then
    if $beverbose ; then
      echo "Cleaning up in the background:" "${1}"*
    fi
    rm -rf "${1}"* &
  elif ! $inplace ; then
    if $beverbose ; then
      echo "Moving interim volumes in the background to keep in ${out}:" "${1}"*
    fi
    mv "${1}"* "${out}" &
  fi
}





# Actual processing starts from here

# prepare averaged BG's
bumpstage
bgOu="${iout}${pstage}_bg_org.tif"
bgSu="${iout}${pstage}_bg_sft.tif"
dfOu="${iout}${pstage}_df_org.tif"
dfSu="${iout}${pstage}_df_sft.tif"
createMask=false
if [ -z "$gmask" ] ; then
  createMask=true
  gmask="${iout}${pstage}_mask.tif"
fi
if (( stage >= fromStage )) ; then
  announceStage "preparing"
  averageHdf2Tif "$bgO" "$bgOu"
  averageHdf2Tif "$bgS" "$bgSu"
  averageHdf2Tif "$dfO" "$dfOu"
  averageHdf2Tif "$dfS" "$dfSu"
  if $createMask ; then
    if $beverbose ; then
      echo "No mask provided (-m option)."
      echo "Creating mask from background image \"$bgOu\" and save to \"$gmask\"."
    fi
    premask="${iout}.premask.tif"
    execMe "ctas v2v $bgOu -i 8 -m 65534 -M 65535 -o $premask"
    execMe "convert $premask -morphology dilate square -negate ${gmask}"
    rm "$premask"
  fi
fi




# split into org and sft
bumpstage
splitOut="${iout}${pstage}_split_"
if (( stage >= fromStage )) ; then
  announceStage "splitting input into original and shifted components"
  if needToMake "${splitOut}mask.tif" "${splitOut}org.hdf" "${splitOut}sft.hdf" ; then
    splitOpt="$beverboseO"
    if [ -n "$bgO" ] ; then splitOpt="$splitOpt $( addOpt -b "$bgOu" ) " ; fi
    if [ -n "$bgS" ] ; then splitOpt="$splitOpt $( addOpt -B "$bgSu" ) " ; fi
    if [ -n "$dfO" ] ; then splitOpt="$splitOpt $( addOpt -d "$dfOu" ) " ; fi
    if [ -n "$dfS" ] ; then
      splitOpt="$splitOpt -D $dfSu "
    elif [ -n "$dfO" ] ; then # same DF for org and sft
      splitOpt="$splitOpt -D $dfOu "
    fi
    splitOpt="$splitOpt $( addOpt -m "$gmask -M $gmask" ) "
    splitOpt="$splitOpt $( addOpt -c "$cropStr" ) "
    splitOpt="$splitOpt $( addOpt -r "$rotate" ) "
    splitOpt="$splitOpt $( addOpt -z "$binn" ) "
    splitOpt="$splitOpt $( addOpt -s "$zinn" ) "
    splitOpt="$splitOpt $( addOpt -i "$fill" ) "
    execMe "$EXEPATH/split.sh  -f $firstO -F $firstS -e $end $splitOpt $inp $splitOut "
    cp  "${splitOut}mask.tif" .split_shape.tif
  fi
fi
if (( zinn != 1)) ; then # dose reduction by selecting fraction of images
  rzin=$((zinn/2))
  ark=$(( ( ark + rzin ) / zinn ))
  firstS=$(( (firstS - firstO + rzin ) / zinn ))
  firstO=0
fi



# find pair wise shifts
bumpstage
shiftsOut="${iout}${pstage}_shifts.dat"
if (( stage >= fromStage )) ; then
  announceStage "find pairwse shifts"
  if needToMake "$shiftsOut" ; then
    stFile="${iout}.stitches.txt"
    execMe "$EXEPATH/stitch.sh -f 0 -F 0 -a $ark -s $(( firstS - firstO )) -g 0,0 -c 0  > $stFile "
    shiftsOpt="$beverboseO"
    shiftsOpt="$shiftsOpt -s $stFile -m ${splitOut}mask.tif"
    execMe "$EXEPATH/pairShift.py $shiftsOpt ${splitOut}org.hdf:/data ${splitOut}sft.hdf:/data -o $shiftsOut"
    rm -rf "$stFile"
  fi
fi


# patch the gaps
bumpstage
patchOut="${iout}${pstage}_patched"
if (( stage >= fromStage )) ; then
  announceStage "patching gaps"
  if needToMake "${patchOut}ForProc.hdf" "${patchOut}ForTrack.hdf" ; then
    patchOpt="$beverboseO"
    patchOpt="$patchOpt -m ${splitOut}mask.tif -s $shiftsOut"
    patchOpt="$patchOpt -o ${patchOut}ForProc.hdf:/data -w ${patchOut}ForTrack.hdf:/data"
    execMe "$EXEPATH/patchMe.py $patchOpt ${splitOut}org.hdf:/data ${splitOut}sft.hdf:/data "
  fi
  cleanUp "${splitOut}"
fi


# track the ball
bumpstage
trackOut="${iout}${pstage}_track.dat"
if (( stage >= fromStage )) ; then
  announceStage "tracking the ball"
  if needToMake "$trackOut" ; then
    execMe "$EXEPATH/trackme.py ${patchOut}ForTrack.hdf:/data -o $trackOut $beverboseO -m 0"
    ctas v2v "${patchOut}ForTrack.hdf:/data:0" -o .split_shape.tif
  fi
  cleanUp "${patchOut}ForTrack.hdf"
fi
splitWidth=$( identify -quiet ".split_shape.tif" | cut -d' ' -f 3 |  cut -d'x' -f 1 )
ballWidth=$( identify -quiet "$EXEPATH/ball.tif" | cut -d' ' -f 3 |  cut -d'x' -f 1 )
execMe "$EXEPATH/analyzeTrack.py -a $ark -s $(( firstS - firstO )) -w $splitWidth -W $ballWidth $trackOut > $EXECRES"
read -r amplX amplY shiftX shiftY centdiv trueArk < "$EXECRES"
#centdiv=$( echo "scale=2; $centdiv - $amplX" | bc )
if $beverbose ; then
  echo "Jitter amplitudes: $amplX $amplY"
  echo "Rotation centre deviation: $centdiv"
  echo "Tracked 180-deg ark (for indication only): $trueArk"
fi


# align
bumpstage
alignOut="${iout}${pstage}_align"
if (( stage >= fromStage )) ; then
  announceStage "aligning"
  if needToMake "${alignOut}.hdf" ; then
    alignOpt="$beverboseO"
    alignOpt="$alignOpt -S ${amplX},${amplY} -m 0 -s $trackOut"
    if $jonly ; then
      alignOpt="$alignOpt -J "
    fi
    execMe "$EXEPATH/align $alignOpt ${patchOut}ForProc.hdf:/data -o ${alignOut}.hdf:/data"
  fi
  cleanUp "${patchOut}ForProc.hdf"
fi


# fill gaps
bumpstage
fillOut="${iout}${pstage}_fill"
if (( stage >= fromStage )) ; then
  announceStage "filling gaps in sinograms"
  if needToMake "${fillOut}.hdf" ; then
    fillOpt="$beverboseO"
    fillCom="$EXEPATH/sinogapme.py $fillOpt"
    execMe "$fillCom ${alignOut}.hdf:/data -m ${alignOut}_mask.tif ${fillOut}.hdf:/data"
    leftMask="${fillOut}_mask_left.tif"
    if [ -e "$leftMask" ] && [ "1" != "$( convert "$leftMask" -format '%[fx:minima]' info: )" ] ; then
      announceStage "1" "filling gaps left after sinogap"
      execMe "ctas proj $beverboseO ${fillOut}.hdf:/data -o ${fillOut}_am.hdf:/data -M $leftMask -I AM"
      rm "${fillOut}.hdf"
      mv "${fillOut}_am.hdf" "${fillOut}.hdf"
    fi
  fi
  cleanUp "${alignOut}.hdf"
fi



# phase contrast
bumpstage
ipcIn="${fillOut}.hdf"
ipcOut="${iout}${pstage}_ipc.hdf"
if (( stage >= fromStage )) ; then
  announceStage "inline phase contrast"
  if [ -z "$d2b" ] ; then # no IPC
    if $beverbose ; then
      echo 'Skipping this stage because no delta to beta ratio provided (-i option)'
    fi
    if $keepClean  ||  ! $cleanup ; then
      ln -s "$ipcIn" "$ipcOut"
    else
      mv "$ipcIn" "$ipcOut"
    fi
  else
    if needToMake "$ipcOut" ; then
      ipcOpt="$beverboseO"
      ipcOpt="$ipcOpt -e -d $d2b -p "
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
      execMe "ctas ipc $ipcOpt ${ipcIn}:/data -o ${ipcOut}:/data"
    fi
    if ! $keepClean ; then
      cleanUp "${ipcIn}"
    fi
  fi
fi


# ring artefact removal
bumpstage
fillCom="$EXEPATH/ring.py"
ringOut="${iout}${pstage}_ring.hdf"
if (( stage >= fromStage )) ; then
  announceStage "ring artefact correction"
  if [ -z "$ring" ] ; then # no ring removal
    if $beverbose ; then
      echo "Skipping this stage because no ring filter size provided (-R option)."
    fi
    if $cleanup ; then
      mv "$ipcOut" "$ringOut"
    else
      ln -s "$ipcOut" "$ringOut"
    fi
  else
    if needToMake "$ringOut" ; then
      ringOpt="$beverboseO"
      # first ring removal algorithm (from Ashkan)
      execMe "$EXEPATH/ring.py $ringOpt --correct ${ipcOut}:/data ${ringOut}:/data"
      # second ring removal algorithm (from ctas), applied only to sinogapped regions
      if (( $ring > 0 )) ; then
        execMe "mv  ${ringOut}  ${ringOut}.hdf"
        execMe "ctas ring $ringOpt -R $ring -o ${ringOut}:/data:y -m ${alignOut}_mask.tif  ${ringOut}.hdf:/data:y "
        rm  "${ringOut}.hdf"
      fi
    fi
    cleanUp "${ipcOut}"
    #cleanUp "${alignOut}_mask.tif"
  fi
fi


# CT
bumpstage
ctOut="${iout}${pstage}_rec.hdf"
if (( stage >= fromStage )) ; then
  announceStage "CT reconstruction"
  if needToMake "$ctOut" ; then
    ctOpt="$beverboseO"
    ctOpt="$ctOpt $( addOpt -r "$pix" ) "
    ctOpt="$ctOpt $( addOpt -w "$wav" ) "
    step=$(echo "scale=8 ; 180 / $ark " | bc )
    execMe "ctas ct $ctOpt -k a -a $step -c $centdiv ${ringOut}:/data:y -o ${ctOut}:/data"
  fi
  cleanUp "${ringOut}"
fi


# moving results and cleanup
announceStage "Moving result"
if $cleanup ; then
  execMe "mv ${ctOut} $outDest"
else
  execMe "cp ${ctOut} $outDest"
fi
if $cleanup  &&  [ "$iout" == "/dev/shm/bctppl/" ] ; then
  rm -rf "$iout"
fi


if (( stage < fromStage )) ; then
  echo "Error! Start stage $fromStage is greater than the last stage $stage." >&2
  exit 1
fi
if $beverbose ; then
  echo "Waiting for background jobs to complete ... "
fi
fg
if $beverbose ; then
  echo
  echo "All done."
fi
exit 0



