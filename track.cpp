#include <poptmx.h>
#include <ctas/external.world.h>
#include <ctas/parallel.world.h>
#include <ctas/matrix.world.h>

#include <map>
#include <opencv2/core.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/highgui.hpp>


using namespace std;



/// \CLARGSins
struct clargs {
  Path command;               ///< Command name as it was invoked.
  deque<ImagePath> images;        ///< images to combine
  ImagePath bgs;        ///< Array of the background images.
  ImagePath dfs;        ///< Array of the dark field images.
  ImagePath dgs;        ///< Array of the dark field images.
  ImagePath mss;        ///< Mask Array.
  Path out_name;
  Crop<2> crp;                  ///< Crop input projection image
  uint denoiseRad;
  float denoiseThr;
  bool beverbose;
    /// \CLARGSF
  clargs(int argc, char *argv[]);
};

clargs::
clargs(int argc, char *argv[])
  : denoiseRad(0)
  , denoiseThr(0.0)
  , beverbose(false)
{

  poptmx::OptionTable table
  ("Object tracking.",
   "Tracks manually selected opbject in the train of images.");

  table
    .add(poptmx::NOTE, "ARGUMENTS:")
    .add(poptmx::ARGUMENT, &images, "images", "Input 2D or 3D images.",
         "All images must be of the same face size. HDF5 format:\n"
         "    file:dataset[:[slice dimension][slice(s)]]\n" + DimSliceOptionDesc)

    .add(poptmx::NOTE, "OPTIONS:")
    .add(poptmx::OPTION, &out_name, 'o', "output", "Output image.", "", out_name)
    .add(poptmx::OPTION, &crp, 'c', "crop", "Crop input volume.", CropOptionDesc)
    .add(poptmx::OPTION, &bgs, 'B', "bg", "Background image", "")
    .add(poptmx::OPTION, &dfs, 'D', "df", "Dark field image", "")
    .add(poptmx::OPTION, &dgs, 'F', "dg", "Dark field image(s) for backgrounds", "")
    .add(poptmx::OPTION, &mss, 'M', "mask", "Mask image",
         "Image where values are weights of corresponding pixels in superimposition operations."
         " F.e. 0 values exclude corresponding or affected pixels from further use.")
    .add(poptmx::OPTION, &denoiseRad, 'n', "noise-rad", "Radius for noise reducing algorithm.",
         "Radius of the local region used to calculate average to replace the pixel value." )
    .add(poptmx::OPTION, &denoiseThr, 'N', "noise-thr", "Noise threshold.",
         "If positive, then gives maximum absolute deviation from the average;"
         " if negative, then negated value is maximum relative deviation.")
    .add_standard_options(&beverbose)
    .add(poptmx::MAN, "SEE ALSO:", SeeAlsoList);

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
}



struct FrameProvider {
  ReadVolumeBySlice & reader;
  const Denoiser & dnz;
  const FlatFieldProc & ffp;
  const Crop<2> & crop;
  FrameProvider(ReadVolumeBySlice & reader, const Denoiser & dnz, const FlatFieldProc & ffp, const Crop<2> & crp)
    : reader(reader), dnz(dnz), ffp(ffp), crop(crp) {}
  Map operator()(int index) {
    Map toRet;
    reader.read(index, toRet, crop);
    dnz.proc(toRet);
    ffp.process(toRet);
    return toRet;
  }
};
FrameProvider * frames = 0;

void getFrame(ReadVolumeBySlice & reader, int index, Map & target,
              const Denoiser & dnz, const FlatFieldProc & ffp ) {
  reader.readTo(index, target);
  dnz.proc(target);
  ffp.process(target);
}




const string windowName = "tracker";
const string frameBar = "Frame";
const string loThBar = "Low threshold";
const string hiThBar = "High threshold";
int currentIndex = -1;
cv::Point currentPos(0,0);
cv::Rect roi(-1,-1,0,0);
cv::Size ocvIsh;
PointF<2> extremes(0,0);
PointF<2> clip(0,0);


Map preProcMap(Map & im) {
  const float delta = clip[1]-clip[0];
  if ( delta > 0 ) {
    im = blitz::where(im<clip[0], clip[0], im);
    im = blitz::where(im>clip[1], clip[1], im);
    im -=  clip[0];
    im /= clip[1]-clip[0];
  } else {
    im = blitz::where(im<clip[0], 0, 1);
  }
  return im;
}

cv::Mat preProcFrame(Map im) {
  Map pim = preProcMap(im);
  return cv::Mat(im.shape()[0], im.shape()[1], CV_32F, pim.data());
}


void updateFrame(int index = -1, void* = 0) {

  if (index>0)
    currentIndex = index;
  else if (currentIndex<0)
    return;

  Map im = (*frames)(currentIndex);
  const float immin = min(im);
  const float immax = max(im);
  if (extremes[0]>immin)
    extremes[0] = immin;
  if (extremes[1]<immax)
    extremes[1] = immax;
  const float delta = extremes[1]-extremes[0];
  if (delta>0.0) {
    clip[0] = extremes[0] + delta * cv::getTrackbarPos(loThBar, windowName) / 100.00;
    clip[1] = extremes[1] - delta * (1 - cv::getTrackbarPos(hiThBar, windowName) / 100.00);
  } else {
    clip[0] = extremes[0];
    clip[1] = extremes[0];
  }

  cv::Mat oim = preProcFrame(im), rgboim;
  cv::cvtColor(oim, rgboim, cv::COLOR_GRAY2BGR);
  if (roi.x>=0) {
    cv::Rect plotRoi = roi;
    if (roi.empty()) {
      cv::Point end( max(0, min( ocvIsh.width -1, currentPos.x) ),
                     max(0, min( ocvIsh.height-1, currentPos.y) ) );
      plotRoi = cv::Rect( cv::Point( min(roi.x, end.x), min(roi.y, end.y) ),
                          cv::Point( max(roi.x, end.x), max(roi.y, end.y) ) );
    }
    cv::rectangle(rgboim, plotRoi, cv::Scalar(0,0,255), 2, cv::LINE_AA);
  }
  cv::setTrackbarPos(frameBar, windowName, currentIndex);
  cv::imshow(windowName, rgboim);
  cv::waitKey(1);

}


void onHiThreshold(int val, void* = 0);
void onLoThreshold(int val, void* = 0);

void onLoThreshold(int val, void*) {
  cv::setTrackbarPos(loThBar, windowName, val);
  if (val > cv::getTrackbarPos(hiThBar, windowName))
    onHiThreshold(val);
  updateFrame();
}

void onHiThreshold(int val, void*) {
  cv::setTrackbarPos(hiThBar, windowName, val);
  if (val < cv::getTrackbarPos(loThBar, windowName))
    onLoThreshold(val);
  updateFrame();
}


void onMouse(int event, int x, int y, int flags, void * ish) {
  if (currentIndex < 0)
      return;
  currentPos = cv::Point(x,y);
  if ( event == cv::EVENT_RBUTTONDOWN and x > 0 and y > 0 ) {
      roi = cv::Rect(x,y, 0, 0);
  } else if (event == cv::EVENT_RBUTTONUP and roi.x >= 0 ) {
    if ( currentPos.x == roi.x or currentPos.y == roi.y  )
      roi = cv::Rect(-1,-1,0,0);
    else if (roi.empty()) {
      cv::Point end( max(0, min( ocvIsh.width -1, x) ),
                     max(0, min( ocvIsh.height-1, y) ) );
      roi = cv::Rect( cv::Point( min(roi.x, end.x), min(roi.y, end.y) ),
                      cv::Point( max(roi.x, end.x)-1, max(roi.y, end.y)-1 ) );
    }
  } else if ( event != cv::EVENT_MOUSEMOVE or not (flags & cv::EVENT_FLAG_RBUTTON) )
    return;
  updateFrame();
};





Map & normalizeMap(Map & im, const blitz::Array<uchar,2>  & mask) {
  const int maskSum = blitz::count(mask);
  const float mn = blitz::sum( im * mask ) / maskSum;
  im -= mn;
  const float st = sqrt(blitz::sum(im*im*mask) / maskSum);
  im /= st;
  return im;
}



class ProcInThread : public InThread {

  bool inThread(long int idx) {
    if (idx >= provider.reader.slices())
      return false;

    Map im = provider(idx);
    preProcMap(im);
    normalizeMap(im,mask);
    float curMin;
    PointI<2> curMinPos;
    for (ssize_t y=0 ; y<sh(0)-tsh(0) ; ++y) {
      Segment seg0(y,y+tsh[0]);
      for (ssize_t x=0 ; x<sh(1)-tsh(1) ; ++x) {
        Segment seg1(x,x+tsh[1]);
        Crop cropBox( blitz::TinyVector<Segment,2>{seg0,seg1} );
        Map box = cropBox.apply(im);
        blitz::Array<uchar,2> boxMask = cropBox.apply(mask).copy();
        //normalizeMap(box, boxMask);
        float resSum = blitz::sum( box * trg * boxMask * trgMask ) / blitz::count( boxMask & trgMask );
        if ( !(x+y) || resSum < curMin) {
          curMin = resSum;
          curMinPos = PointI<2>(y,x);
        }
      }
    }
    lock();
    cout << idx << " : " << curMinPos << endl ;

    Crop cropBox( blitz::TinyVector<Segment,2>{Segment(curMinPos[0],curMinPos[0]+tsh[0]),
                                               Segment(curMinPos[1],curMinPos[1]+tsh[1])} );
    blitz::Array<uchar,2> boxMask = cropBox.apply(mask).copy();
    Map maskF(boxMask.shape());
    maskF = 1.0 * boxMask;
    SaveImage("tmp_mask.tif", maskF);


    Map maskTF(trgMask.shape());
    maskTF = maskF * trgMask;
    SaveImage("tmp_mask_trg.tif", maskTF);

    SaveImage("tmp_trg.tif", cropBox.apply(im));

    unlock();
    bar.update();
    return true;
  }

  FrameProvider & provider;
  const Shape<2> sh;
  const blitz::Array<uchar,2> mask;
  const blitz::Array<uchar,2> trgMask;
  const Map trg;
  const Shape<2> tsh;

public:

  ProcInThread(FrameProvider & _provider, int initIdx, Crop<2> & initBox, const Map & inMask, bool beverbose)
    : InThread(beverbose, "Tracking pattern", _provider.reader.slices())
    , provider(_provider)
    , sh(provider.reader.face())
    , mask( [](const Map & inm) {
        blitz::Array<uchar,2> myMask(inm.shape());
        myMask = blitz::where(inm<1.0, 0, 1);
        return myMask;
      } (inMask) )
    , trgMask( initBox.apply(mask) )
    , trg( [this](FrameProvider & provider, int initIdx, Crop<2> & initBox){
        Map frame = provider(initIdx);
        Map pFrame = preProcMap(frame);
        Map toRet = initBox.apply(pFrame).copy();
        normalizeMap(toRet, trgMask);
        return toRet;
      }(provider, initIdx, initBox) )
    , tsh(trg.shape())
  {
  }


};






int main(int argc, char *argv[]) { {

  const clargs args(argc, argv) ;

  ReadVolumeBySlice allIn(args.images);
  const int nofProj = allIn.slices();
  const Shape<2> ish(args.crp.shape(allIn.face()));
  ocvIsh = cv::Size(ish[1],ish[0]);

  // Read auxiliary images
  auto readIm = [&ish, &args, &allIn](const ImagePath & path, Map & image) {
    if (!path.empty())
      ReadImage(path, image, args.crp, allIn.face());
    else {
      image.resize(0,0);
    }
  }  ;
  Map bg, df, dg, ms;
  readIm(args.bgs, bg);
  readIm(args.dfs, df);
  readIm(args.dgs, dg);
  readIm(args.mss, ms);
  // normalize mask
  const float vmin = min(ms);
  const float delta = max(ms) - vmin;
  if (delta==0.0) ms = 1.0;
  else ms = (ms-vmin)/delta;

  // create canonical denoiser and denoise bg and dg
  Denoiser canonDZ(ish, (int) args.denoiseRad, args.denoiseThr);
  canonDZ.proc(bg);
  canonDZ.proc(df);
  // create canonical flatfielder
  FlatFieldProc canonFF(bg, df, dg, ms);

  FrameProvider provider(allIn, canonDZ, canonFF, args.crp);
  frames = &provider;


  // GUI

  cv::namedWindow(windowName, cv::WINDOW_GUI_NORMAL);
  cv::setMouseCallback(windowName, onMouse);
  cv::createTrackbar(frameBar,windowName, 0, nofProj-1, updateFrame);
  cv::createTrackbar(loThBar, windowName, 0, 100, onLoThreshold);
  cv::createTrackbar(hiThBar, windowName, 0, 100, onHiThreshold);
  onLoThreshold(0);
  onHiThreshold(100);
  updateFrame(1);
  while (int key = cv::waitKey())
    if (key == 27)
      break;

  Crop cropTarget( blitz::TinyVector<Segment,2>{ Segment(roi.y,roi.y+roi.height),
                                                 Segment(roi.x,roi.x+roi.width) } );
  ProcInThread proc(provider, currentIndex, cropTarget, ms, args.beverbose);
  proc.execute();

} }
















