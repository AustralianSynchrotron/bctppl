#include <poptmx.h>
#include <ctas/external.world.h>
#include <ctas/parallel.world.h>
#include <ctas/matrix.world.h>

using namespace std;

struct clargs {
  Path command;               ///< Command name as it was invoked.
  deque<ImagePath> images;        ///< images to align
  Path shifts;
  Path outmask;
  bool beverbose;
    /// \CLARGSF
  clargs(int argc, char *argv[]);
};


clargs::clargs(int argc, char *argv[])
  : beverbose(false)
{
  poptmx::OptionTable table
  ("Aligns stack of images.",
   "Aligns images in accordance with relative shifts provided as input.");

  table
    .add(poptmx::NOTE, "ARGUMENTS:")
    .add(poptmx::ARGUMENT, &images, "images", "Input 2D or 3D images.",
         "All images must be of the same face size. HDF5 format:\n"
         "    file:dataset[:[slice dimension][slice(s)]]\n" + DimSliceOptionDesc)

    .add(poptmx::NOTE, "OPTIONS:")
    .add(poptmx::OPTION, &outmask, 'o', "output", "Output result mask or filename.",
           "Output filename if output is a single file. Output mask otherwise. " + MaskDesc, outmask)
    .add(poptmx::OPTION, &shifts, 's', "shifts", "Text file with shifts.",
         "Shifts are two columns of numbers representinf X and Y shifts for each slice in input stack."
         " Must contain at least same number of shifts as input slices.")
    .add_standard_options(&beverbose);

  if ( ! table.parse(argc,argv) )
    exit(0);
  if ( ! table.count() ) {
    table.usage();
    exit(0);
  }
  command = table.name();

  if ( ! table.count(&images) )
    exit_on_error(command, string () +
                  "Missing required argument: "+table.desc(&images)+".");
  if ( ! table.count(&shifts) )
    exit_on_error(command, string () +
                  "Missing required argument: "+table.desc(&shifts)+".");
  if ( ! table.count(&outmask) )
    exit_on_error(command, string () +
                  "Missing required argument: "+table.desc(&outmask)+".");

}



class ProcInThread : public InThread {

  bool inThread(long int idx) {
    if (idx >= reader.slices())
      return false;

    bar.update();
    return true;
  }

  ReadVolumeBySlice & reader;
  SaveVolumeBySlice & writer;
  blitz::Array<int,2> & iShfts;
  Crop<2> & crp;

public:

  ProcInThread(ReadVolumeBySlice & reader, SaveVolumeBySlice & writer,
               blitz::Array<int,2> & iShfts, Crop<2> & crp, bool beverbose)
    : InThread(beverbose, "Tracking pattern", reader.slices())
    , reader(reader)
    , writer(writer)
    , iShfts(iShfts)
    , crp(crp)
  {}


};


\

int bz_lround(double x) {return lround(x);}
BZ_DECLARE_FUNCTION_RET(bz_lround, int);

int main(int argc, char *argv[]) { {
  const clargs args(argc, argv) ;

  ReadVolumeBySlice iVol(args.images);
  Map shifts = LoadData(args.shifts);
  const Shape<2> ish = iVol.face();
  const int nofIm = iVol.slices();
  if ( shifts.shape()[1] != 2 )
    exit_on_error(args.command, "Unexpected number of columns inside input file \""+args.shifts+"\""
                                " ("+toString(shifts.shape()[0])+" where 2 are expected.)");
  if ( shifts.shape()[0] != nofIm )
    exit_on_error(args.command, "Number of slices inside input stack ("+toString(nofIm)+")"
                                " is same as number of shifts inside file \""+args.shifts+"\""
                                " ("+toString(shifts.shape()[1])+".");
  blitz::Array<int,2> iShifts(-bz_lround(shifts));
  blitz::Array<int,1> xShifts = iShifts(all,0);
  xShifts -= min(xShifts);
  const int xWid = max(xShifts);
  blitz::Array<int,1> yShifts = iShifts(all,1);
  yShifts -= min(yShifts);
  const int yWid = max(yShifts);
  const Crop<2> crp( Segment(yWid, ish(0)-yWid), Segment(xWid, ish(1)-xWid) );
  const Shape<2> osh( ish(0)-2*yWid , ish(1)-2*xWid );
  SaveVolumeBySlice oVol( args.outmask, Shape<3>( nofIm, osh(0), osh(1) ) );
  ProgressBar bar(args.beverbose, "Aligning", nofIm);
  InThread::execute( nofIm, [&](long int curIm){
    Map iIm(ish);
    Map genMap(ish(0)+yWid, ish(1)+xWid);
    iVol.readTo(curIm, iIm);
    genMap( blitz::Range(yShifts(curIm), yShifts(curIm) + ish(0)-1),
            blitz::Range(xShifts(curIm), xShifts(curIm) + ish(1)-1) ) = iIm;
    oVol.save(curIm, crp.apply(genMap));
    bar.update();
  } );


} exit(0); }

