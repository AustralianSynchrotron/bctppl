#include <poptmx.h>
#include <ctas/external.world.h>
#include <ctas/parallel.world.h>
#include <ctas/matrix.world.h>

using namespace std;

struct clargs {
  Path command;               ///< Command name as it was invoked.
  deque<ImagePath> images;        ///< images to align
  Path shifts;
  ImagePath outimages;
  Path mask;
  PointI<2> maxShifts;
  bool only; //< align only y (vertical axis)
  bool beverbose;
    /// \CLARGSF
  clargs(int argc, char *argv[]);
};


clargs::clargs(int argc, char *argv[])
  : maxShifts(0,0)
  , only(false)
  , beverbose(false)
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
    .add(poptmx::OPTION, &outimages, 'o', "output", "Output result prefix or filename.",
           "Output filename if output is a single file. Output mask otherwise. " + MaskDesc, outimages)
    .add(poptmx::OPTION, &shifts, 's', "shifts", "Text file with shifts.",
         "Shifts are two columns of numbers representinf X and Y shifts for each slice in input stack."
         " Must contain at least same number of shifts as input slices.")
    .add(poptmx::OPTION, &maxShifts, 'S', "maxShifts", "Maximum shifts.", "" )
    .add(poptmx::OPTION, &mask, 'm', "mask", "Mask of the original input volume.",
         "If provided, corresponding combined mask will be created.")
    .add(poptmx::OPTION, &only, 'J', "onlyY", "Align only vertical axis.", "" )
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
  if ( ! table.count(&outimages) )
    exit_on_error(command, string () +
                  "Missing required argument: "+table.desc(&outimages)+".");

}




int main(int argc, char *argv[]) { {
  const clargs args(argc, argv) ;

  ReadVolumeBySlice iVol(args.images);
  Map shifts = LoadData(args.shifts);
  const Shape<2> ish = iVol.face();
  const int nofIm = iVol.slices();
  if ( shifts.shape()[1] < 2 )
    exit_on_error(args.command, "Unexpected number of columns inside input file \""+args.shifts+"\""
                                " ("+toString(shifts.shape()[0])+" where 2 are expected.)");
  if ( shifts.shape()[0] != nofIm )
    exit_on_error(args.command, "Number of slices inside input stack ("+toString(nofIm)+")"
                                " is same as number of shifts inside file \""+args.shifts+"\""
                                " ("+toString(shifts.shape()[1])+".");
  blitz::Array<int,2> iShifts(-blitz::cast<int>(shifts));

  blitz::Array<int,1> xShifts = iShifts(all,1);
  if (args.only)
    xShifts = 0;
  const int xWid = max( abs(args.maxShifts(1)),  long(max(xShifts)-min(xShifts)) );
  xShifts += xWid;

  blitz::Array<int,1> yShifts = iShifts(all,0);
  const int yWid = max( abs(args.maxShifts(0)),  long(max(yShifts)-min(yShifts)) );
  yShifts += yWid;

  const Shape<2> osh( ish(0)-2*yWid , ish(1)-2*xWid );
  Map mask, omask;
  if (!args.mask.empty()) {
    omask.resize(osh);
    omask = 1.0;
    if (args.mask != "0") {
      ReadImage(args.mask, mask, ish);
      mask /= max(mask);
    }
  }
  const Crop<2> crp( Segment(2*yWid, osh(0)+2*yWid), Segment(2*xWid, osh(1)+2*xWid) );
  SaveVolumeBySlice oVol( args.outimages, Shape<3>( nofIm, osh(0), osh(1) ) );

  ProgressBar bar(args.beverbose, "Aligning", nofIm);
  InThread::execute( nofIm,
    [&](long int curIm){
      Map iIm(ish);
      iVol.readTo(curIm, iIm);
      Map genMap(ish(0)+2*yWid, ish(1)+2*xWid);
      Map subGen = genMap( blitz::Range(yShifts(curIm), yShifts(curIm) + ish(0)-1),
                           blitz::Range(xShifts(curIm), xShifts(curIm) + ish(1)-1) );
      subGen = iIm;
      oVol.save(curIm, crp.apply(genMap));
      if (omask.size()) {
        subGen = mask.size() ? mask : Map(blitz::where( iIm == 0 , 0.0, 1.0 ));
        omask *= crp.apply(genMap);
      }
      bar.update();
    } );
  if (omask.size()) {
    const string omaskName = args.outimages.dtitle() + "_mask.tif";
    SaveImage(omaskName, omask);
  }


} exit(0); }

