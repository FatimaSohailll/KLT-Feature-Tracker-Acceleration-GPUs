/*********************************************************************
 * pnmio.h
 *********************************************************************/

#ifndef _PNMIO_H_
#define _PNMIO_H_

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**********
 * With pgmReadFile and pgmRead, setting img to NULL causes memory
 * to be allocated
 */

/**********
 * used for reading from/writing to files
 */
extern unsigned char* pgmReadFile(
  char *fname,
  unsigned char *img,
  int *ncols, 
  int *nrows);
extern void pgmWriteFile(
  char *fname,
  unsigned char *img,
  int ncols,
  int nrows);
extern void ppmWriteFileRGB(
  char *fname,
  unsigned char *redimg,
  unsigned char *greenimg,
  unsigned char *blueimg,
  int ncols,
  int nrows);

/**********
 * used for communicating with stdin and stdout
 */
extern unsigned char* pgmRead(
  FILE *fp,
  unsigned char *img,
  int *ncols, int *nrows);
extern void pgmWrite(
  FILE *fp,
  unsigned char *img,
  int ncols,
  int nrows);
extern void ppmWrite(
  FILE *fp,
  unsigned char *redimg,
  unsigned char *greenimg,
  unsigned char *blueimg,
  int ncols,
  int nrows);

#ifdef __cplusplus
}
#endif

#endif /* _PNMIO_H_ */