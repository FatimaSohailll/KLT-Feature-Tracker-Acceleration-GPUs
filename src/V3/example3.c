#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "pnmio.h"
#include "klt.h"
/* #define REPLACE */

#ifdef WIN32
int RunExample3()
#else
int main(int argc, char *argv[])
#endif
{
  if (argc < 4) {
    printf("Usage: %s <dataset_name> <num_features> <num_frames>\n", argv[0]);
    printf("Example: ./example3_gpu images_provided 150 10\n");
    return -1;
  }

  char dataset[256];
  strcpy(dataset, argv[1]);
  int nFeatures = atoi(argv[2]);
  int nFrames = atoi(argv[3]);

  unsigned char *img1, *img2;
  char fnamein[256], fnameout[256];
  KLT_TrackingContext tc;
  KLT_FeatureList fl;
  KLT_FeatureTable ft;
  int ncols, nrows;
  int i;

  /* -------- Initialize KLT -------- */
  tc = KLTCreateTrackingContext();
  fl = KLTCreateFeatureList(nFeatures);
  ft = KLTCreateFeatureTable(nFrames, nFeatures);

  tc->sequentialMode = TRUE;
  tc->writeInternalImages = FALSE;
  tc->affineConsistencyCheck = -1;

  /* -------- Read first frame -------- */
  sprintf(fnamein, "../../data/%s/img1.pgm", dataset);
  img1 = pgmReadFile(fnamein, NULL, &ncols, &nrows);
  img2 = (unsigned char *) malloc(ncols * nrows * sizeof(unsigned char));

  /* -------- Select features -------- */
  KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
  KLTStoreFeatureList(fl, ft, 0);
  KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "./feat/feat1.ppm");

  double timeTotal = 0.0; 
  for (i = 1; i < nFrames; i++) { 
    sprintf(fnamein, "../../data/%s/img%d.pgm", dataset, i);
    if (pgmReadFile(fnamein, img2, &ncols, &nrows) == NULL) {
      printf("Error: Could not read %s\n", fnamein);
      break;
    }

    clock_t start = clock();
    KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);
    clock_t end = clock();

    timeTotal += (double)(end - start);

#ifdef REPLACE
    KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);
#endif

    KLTStoreFeatureList(fl, ft, i - 1);
    sprintf(fnameout, "./feat/feat%d.ppm", i);
    KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, fnameout);

    memcpy(img1, img2, ncols * nrows * sizeof(unsigned char));
  }

  /* -------- Save and cleanup -------- */
  KLTWriteFeatureTable(ft, "./feat/features2.txt", "%5.1f");
  KLTWriteFeatureTable(ft, "./feat/features2.ft", NULL);
  KLTFreeFeatureTable(ft);
  KLTFreeFeatureList(fl);
  KLTFreeTrackingContext(tc);
  free(img1);
  free(img2);

  timeTotal /= CLOCKS_PER_SEC;
  printf("Total tracking time: %.6f seconds\n", timeTotal);

  return 0;
}
