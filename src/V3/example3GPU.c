#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "pnmio.h"
#include "kltGPU.h"
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
  int nFrames = 10;  
  int ncols, nrows;
  int i;

  /* -------- Initialize KLT -------- */
  tc = KLTCreateTrackingContext();
  fl = KLTCreateFeatureList(nFeatures);
  ft = KLTCreateFeatureTable(nFrames, nFeatures);

  tc->sequentialMode = TRUE;
  tc->writeInternalImages = FALSE;
  tc->affineConsistencyCheck = -1;  /* set to 2 for affine consistency check */

  /* -------- Read first frame (img1.pgm) -------- */
  img1 = pgmReadFile("../../data/images_provided/img0.pgm", NULL, &ncols, &nrows);
  img2 = (unsigned char *) malloc(ncols * nrows * sizeof(unsigned char));

  /* -------- Select features in first frame -------- */
  KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
  KLTStoreFeatureList(fl, ft, 0);
  KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "./feat/feat1.ppm");

  double timeTotal = 0.0; 
  for (i = 1; i < nFrames; i++) { 
    sprintf(fnamein, "../../data/images_provided/img%d.pgm", i);
    if (pgmReadFile(fnamein, img2, &ncols, &nrows) == NULL) {
      printf("Error: Could not read %s\n", fnamein);
      break;
    }

    clock_t start = clock();
    KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);
    clock_t end = clock();

    timeTotal += (double)(end-start);

#ifdef REPLACE
    KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);
#endif

    KLTStoreFeatureList(fl, ft, i - 1);
    sprintf(fnameout, "./feat/feat%d.ppm", i);
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

  timeTotal = ((double)(timeTotal))/ CLOCKS_PER_SEC;
  printf("Total tracking time: %.6f seconds\n", timeTotal);

  return 0;
}

