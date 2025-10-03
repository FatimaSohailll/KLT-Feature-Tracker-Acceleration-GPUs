/**********************************************************************
Finds the 150 best features in the first frame (img1.pgm) and tracks
them through the next 9 frames (total 10 frames: img1.pgm → img10.pgm).
**********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "pnmio.h"
#include "klt.h"

/* #define REPLACE */

#ifdef WIN32
int RunExample3()
#else
int main()
#endif
{
  unsigned char *img1, *img2;
  char fnamein[256], fnameout[256];
  KLT_TrackingContext tc;
  KLT_FeatureList fl;
  KLT_FeatureTable ft;
  int nFeatures = 150;
  int nFrames = 550;  
  int ncols, nrows;
  int i;

  printf("Running on %d frames (img1.pgm ... img10.pgm)\n", nFrames);

  /* -------- Initialize KLT -------- */
  tc = KLTCreateTrackingContext();
  fl = KLTCreateFeatureList(nFeatures);
  ft = KLTCreateFeatureTable(nFrames, nFeatures);

  tc->sequentialMode = TRUE;
  tc->writeInternalImages = FALSE;
  tc->affineConsistencyCheck = -1;  /* set to 2 for affine consistency check */

  /* -------- Read first frame (img1.pgm) -------- */
  img1 = pgmReadFile("../data/images_laptops/img1.pgm", NULL, &ncols, &nrows);
  img2 = (unsigned char *) malloc(ncols * nrows * sizeof(unsigned char));

  /* -------- Select features in first frame -------- */
  KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
  KLTStoreFeatureList(fl, ft, 0);
  KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "./feat/feat1.ppm");

  for (i = 2; i <= nFrames; i++) {   /* now goes up to img10.pgm */
    sprintf(fnamein, "../data/images_laptops/img%d.pgm", i);
    if (pgmReadFile(fnamein, img2, &ncols, &nrows) == NULL) {
      printf("Error: Could not read %s\n", fnamein);
      break;
    }

    KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);

#ifdef REPLACE
    KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);
#endif

    KLTStoreFeatureList(fl, ft, i - 1);
    sprintf(fnameout, "./feat/featF%d.ppm", i);
    KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, fnameout);

    /* make img2 the new reference for next iteration */
    memcpy(img1, img2, ncols * nrows * sizeof(unsigned char));
}
  /* -------- Save feature table -------- */
  KLTWriteFeatureTable(ft, "./feat/features2.txt", "%5.1f");
  KLTWriteFeatureTable(ft, "./feat/features2.ft", NULL);

  /* -------- Cleanup -------- */
  KLTFreeFeatureTable(ft);
  KLTFreeFeatureList(fl);
  KLTFreeTrackingContext(tc);
  free(img1);
  free(img2);

  return 0;
}

