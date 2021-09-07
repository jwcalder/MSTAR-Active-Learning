/*=========================================================================
 *=========================================================================
  == [DISCLAIMER]: THIS SOFTWARE AND ANY ACCOMPANYING DOCUMENTATION IS   ==
  == RELEASED "AS IS".  THE U.S. GOVERNMENT MAKES NO WARRANTY OF ANY     ==
  == KIND, EXPRESS OR IMPLIED, CONCERNING THIS SOFTWARE AND ANY          ==
  == ACCOMPANYING DOCUMENTATION, INCLUDING, WITHOUT LIMITATION, ANY      ==
  == WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.  ==
  == IN NO EVENT WILL THE U.S. GOVERNMENT BE LIABLE FOR ANY DAMAGES      ==
  == ARISING OUT OF THE USE, OR INABILITY TO USE, THIS SOFTWARE OR ANY   ==
  == ACCOMPANYING DOCUMENTATION, EVEN IF INFORMED IN ADVANCE OF THE      ==
  == POSSIBLITY OF SUCH DAMAGES.                                         ==
  =========================================================================
  =========================================================================*/


/*-------------------------------------------------------------------------
 *                        Routine: mstar2raw (Version 2.0)
 *                         Author: John F. Querns, Veridian Engineering
 *                           Date: 25 September 1998
 *
 * What's new:
 *
 *  (1) Added code to check current CPU for byte ordering. If little-endian,
 *      code automatically byteswaps input data before any further processing
 *      is done.  
 *
 *  (2) Added Magnitude data output only option.
 *
 *-------------------------------------------------------------------------
 *
 * Purpose: This routine inputs MSTAR fullscene and target chip images   
 *          and outputs:
 *
 *          MSTAR TARGET CHIPS:
 *
 *             32-bit float Magnitude + 32-bit float phase
 *             32-bit float Magnitude
 *
 *          MSTAR FULLSCENES (including Clutter scenes):
 *
 *             16-bit UINT Magnitude + 16-bit UINT (12-bits signif) Phase
 *             16-bit UINT Magnitude
 *
 *          MSTAR Phoenix (Summary) header as an ASCII file. 
 *
 *-------------------------------------------------------------------------
 *
 * [Calls]: 
 *
 *     float
 *     byteswap_SR_IR()      -- Does big-endian to little-endian float
 *                              byteswap..this is specifically for the
 *                              case of Sun big-endian to PC-Intel
 *                              little-endian data.
 *
 *     unsigned short
 *     byteswap_SUS_IUS()    -- Does big-endian to little-endian swap for
 *                              unsigned short (16-bit) numbers. This is
 *                              specifically for the case of Sun big-
 *                              endian to PC-Intel little-endian data.
 *
 *     int
 *     CheckByteOrder()      -- This checks the byte order for the CPU that
 *                              this routine is compiled run on. If the
 *                              CPU is little-endian, it will return a 0
 *                              value (LSB_FIRST); else, it will return a 1
 *                              (MSB_FIRST).
 *
 *                              Taken from:                     
 *                                 
 *                                Encyclopedia of Graphic File  
 *                                Formats, Murray & Van Ryper,  
 *                                O'Reilly & Associates, 1994,  
 *                                pp. 114-115. 
 *
 *------------------------------------------------------------------------
 *
 * [Syntax/Usage]:  
 *
 *       mstar2raw <MSTAR Input> [Output Option] [enter]
 *
 *         where:
 *               Output Option = [0] --> Output all data (MAG+PHASE)
 *                               [1] --> Output MAG data only
 *                   
 *                                    
 *      Example 1: Generate RAW binary image and ASCII headers for MSTAR
 *                 fullscene image: hb00001. Output ALL data for output
 *                 image file.
 *
 *                 % mstar2raw hb00001 0 [enter]
 *
 *                 Example 1 will generate the following output files:
 *
 *                 hb00001.all   <-- RAW binary 16-bit mag+12-bit phase data
 *                 hb00001.hdr   <-- ASCII Phoenix (Summary) header
 *
 *     Example 2:  Generate RAW binary image and ASCII headers for MSTAR
 *                 target chip image: hb3900.0015.  Output ALL data for
 *                 output image file. 
 *
 *                 % mstar2raw hb3900.0015 0 [enter]
 *
 *                 Example 2 will generate the following output files:
 *
 *                 hb3900.0015.all <-- RAW 32-bit float mag+phase data
 *                 hb3900.0015.hdr <-- ASCII Phoenix (Summary) header
 *
 *                 NOTE: The MSTAR target chip data is float data be-
 *                       cause it is calibrated.  See the file,
 *                       "MSTAR.txt", for an explanation.
 *
 *     Example 3: Generate RAW binary image and ASCII headers for MSTAR
 *                fullscene clutter image "hb12345". Output ONLY the 
 *                magnitude data in the RAW output image.
 *
 *                 % mstar2raw hb12345 1 [enter]
 *
 *                 Example 3 will generate the following output files:
 *
 *                 hb12345.mag   <-- RAW binary 16-bit magnitude data
 *                 hb12345.hdr   <-- ASCII Phoenix (Summary) header
 *
 *------------------------------------------------------------------------
 * 
 * [Contacts]: 
 * 
 *   John F. Querns
 *   Veridian Engineering (Dayton Group) 
 *   5200 Springfield Pike, Dayton, OH 45431  
 *   email: jquerns@dytn.veridian.com 
 *  
 *   Veridian Contractor Area  
 *   Area B  Bldg 23  Rm 115
 *   2010 Fifth Street 
 *   Wright Patterson AFB, OH  45433-7001  
 *   Work : (937) 255-1116, Ext 2818  
 *   email: jquerns@mbvlab.wpafb.af.mil 
 * 
 *------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/unistd.h>

/* Define MSTAR image type */
#define CHIP_IMAGE   0
#define FSCENE_IMAGE 1
 
#define ALL_DATA   0
#define MAG_DATA   1

#define SUCCESS    0
#define FAILURE   -1

#define LSB_FIRST    0             /* Implies little-endian CPU... */
#define MSB_FIRST    1             /* Implies big-endian CPU...    */

/* Function Declarations */
static float          byteswap_SR_IR();
static unsigned short byteswap_SUS_IUS();
static int            CheckByteOrder();





int main(argc, argv)

  int   argc;
  char *argv[];

{

/************************* D E C L A R A T I O N S *************************/

  FILE *MSTARfp=NULL;        /* Input FILE ptr to MSTAR image file     */
  FILE *RAWfp=NULL;          /* Output FILE ptr to MSTAR RAW data file */
  FILE *HDRfp=NULL;          /* Output FILE ptr to MSTAR header file   */

  int   i, j, rv, n, numrows, numcols, numgot;

  char *MSTARname=NULL;      /* Input MSTAR filename           */
  char  RAWname[80];         /* Output MSTAR RAW filename      */
  char  HDRname[80];         /* Phoenix header filename buffer */

  int            outOpt;     /* ALL data, or MAG ONLY...    */
  int            phlen, nhlen, mstartype; 
  long           magloc, bytesPerImage, nchunks, totchunks;

  char          *tptr=NULL;  /* Temp buffer ptr */
  char          *phdr=NULL;  /* Ptr to buffer to hold Phoenix header */
  unsigned char  tbuff[1024];

  unsigned short *FSCENEdata=NULL; /* Ptr to Fullscene data buffer */
  float          *CHIPdata=NULL;   /* Ptr to CHIp data buffer      */

  /* Byte Order Variables */
  int            byteorder;
  unsigned char  bigfloatbuf[4];   /* BigEndian float buffer... */
  float          littlefloatval;   /* LittleEndian float value  */
  unsigned char  bigushortbuf[2];  /* BigEndian ushort buffer...*/ 
  unsigned short littleushortval;  /* LittleEndian ushort value.*/


  
/************************ B E G I N  C O D E ****************************/

if (argc < 3)
   {
     fprintf(stderr,"\nUsage: mstar2raw <MSTAR Input> [Output Option]\n");
     fprintf(stderr,"where: Output Option = [0] --> Output ALL image data\n");
     fprintf(stderr,"                       [1] --> Output MAG data only\n\n");
     exit(1);
   } else
        {
         MSTARname = argv[1];
         outOpt    = atoi(argv[2]);
         if ( (outOpt != ALL_DATA) &&
              (outOpt != MAG_DATA) )
            {
             fprintf(stderr,
                     "\nError: Incorrect image output option (0:All data or 1:MAG data only)!\n\n");
             exit(1);
            }
        } 

/* Form output Phoenix header filename */
HDRname[0] = '\0';
tptr = (char *) rindex(MSTARname, '/');
if (tptr == NULL)
   {
    strcat(HDRname, MSTARname);
   } else
        {
         sprintf(HDRname, "%s", tptr+1);
        }
strcat(HDRname, ".hdr");

/* Form output MSTAR RAW filename */
RAWname[0] = '\0';
tptr = (char *) rindex(MSTARname, '/');
if (tptr == NULL)
   {
    strcat(RAWname, MSTARname);
   } else
        {
         sprintf(RAWname, "%s", tptr+1);
        }

switch (outOpt)
  {
   case ALL_DATA:
     strcat(RAWname, ".all");
     break;
   case MAG_DATA:
     strcat(RAWname, ".mag");
     break;
  }

printf("\nmstar2raw conversion: started!\n\n");


/***************** MAIN MSTAR PROCESSING AREA ********************/

MSTARfp = fopen(MSTARname,"rb");
if (MSTARfp == NULL)
   {
    fprintf(stderr,
      "\n\nError: Unable to open [%s] for reading!\n\n", MSTARname);
    exit(1);
   }

/****************************************************
 * Read first 512 bytes to figure out some header   *
 * parameters....                                   *
 ****************************************************/
printf("Determining input image/hdr information...\n");

fread(tbuff, sizeof(char), 1024, MSTARfp);
rewind(MSTARfp);

/* Extract Phoenix Summary header length */
tptr = (char *) strstr(tbuff,"PhoenixHeaderLength= ");
if (tptr == (char *) NULL)
   {
    fprintf(stderr,"Can not determine Phoenix header length!\n");
    fclose(MSTARfp);
    exit(1);
   } else
        {
         sscanf((tptr+20), "%d", &phlen);
        }

/* Check for and extract native header length */
tptr = (char *) strstr(tbuff,"native_header_length= ");
if (tptr == (char *) NULL)
   {
    fprintf(stderr,"Can not determine native header length!\n");
    fclose(MSTARfp);
    exit(1);
   } else
        {
         sscanf((tptr+21), "%d", &nhlen);
        }

/* Extract MSTAR column width */
tptr = (char *) strstr(tbuff,"NumberOfColumns= ");
if (tptr == (char *) NULL)
   {
    fprintf(stderr,
           "Error: Can not determine MSTAR image width");
    fclose(MSTARfp);
    exit(1);
   } else
        {
         sscanf((tptr+16), "%d", &numcols);
        }

/* Extract MSTAR row height */
tptr = (char *) strstr(tbuff,"NumberOfRows= ");
if (tptr == (char *) NULL)
   {
    fprintf(stderr,
           "Error: Can not determine MSTAR image height!");
    fclose(MSTARfp);
    exit(1);
   } else
        {
         sscanf((tptr+13), "%d",&numrows); 
        }

//printf("numrows=%d\n",numrows);
//printf("numcols=%d\n",numcols);
//printf("phlen=%d\n",phlen);
//printf("nhlen=%d\n",nhlen);
/* Set MSTAR image type */
if (nhlen == 0)
   {/* Implies FLOAT MSTAR chip image */
    mstartype = CHIP_IMAGE;
   } else
        {
         mstartype = FSCENE_IMAGE; /* UnShort Fullscene */
        }

/*******************************************************
 * Allocate memory to header buffer, read Phoenix hdr, *
 * and write out to output file...                     *
 *******************************************************/
HDRfp = fopen(HDRname, "w");
if (HDRfp == NULL)
   {
    fprintf(stderr,
            "Error: unable to open header file [%s] for writing!\n",
            HDRname);
    fclose(MSTARfp);
    exit(1);
   }

/* Allocate memory to Phoenix header buffer */
phdr = (char *) malloc(phlen+1);
if (phdr == (char *) NULL)
   {
    fprintf(stderr,
              "Error: unable to allocate Phoenix header memory!\n");
    fclose(MSTARfp);
    fclose(HDRfp);
    unlink(HDRname);
    exit(1);
   }

/* Read Phoenix header into buffer */
n = fread(phdr, sizeof(char), phlen, MSTARfp);
if (n != phlen)
   {
    fprintf(stderr,
            "Error: in reading Phoenix header..only read [%d of %d] bytes\n",
              n, phlen);
    free(phdr);
    fclose(MSTARfp);
    fclose(HDRfp);
    unlink(HDRname);  /* Remove Phoenix hdr output file */
    exit(1);
   }

/* Write Phoenix header to output header file */
n = fwrite(phdr, sizeof(char), phlen, HDRfp);
if (n != phlen)
   {
    fprintf(stderr,
           "Error: in writing Phoenix header..only read [%d of %d] bytes\n",
            n, phlen);
    free(phdr);
    fclose(MSTARfp);
    fclose(HDRfp);
    unlink(HDRname);  /* Remove Phoenix hdr output file */
    exit(1);
   }

/* Free Phoenix header memory...*/
fclose(HDRfp);
free(phdr);

printf("Phoenix header written to [%s]...\n", HDRname);


/******************************************************
 * Set up location to point to MSTAR magnitude data.. *
 ******************************************************/
switch (mstartype)
  {
   case CHIP_IMAGE: 
     magloc  = phlen; 
     fseek(MSTARfp, magloc, 0);
     nchunks = numrows * numcols;
     break;
   case FSCENE_IMAGE:
     magloc  = phlen + nhlen; /* nhlen = 512 */
     fseek(MSTARfp, magloc, 0);
     nchunks = numrows * numcols;
     break;
  }

/******************************************************
 * Check byte-order, swap bytes if necessary...       *
 * Allocate memory, read data,  & convert to 8-bit    *
 * based on 'mstartype'                               *
 ******************************************************/

/* Check byteorder */
byteorder = (int) CheckByteOrder();
switch (byteorder)
  {
   case LSB_FIRST:
     printf("==> Little-Endian CPU detected: Will byteswap before scaling data!\n");
     break;
   case MSB_FIRST:
     printf("==> Big-Endian CPU detected: No byteswap needed!\n");
     break;
  }

/******************************************************
 * Allocate memory, read data,  & write out based on  *
 * type of MSTAR image...and which data to write out  *
 *                                                    *
 * NOTE: For Chip data, I allocate all of the memory  *
 *       needed (magnitude+phase), read and then write*
 *       all of it out...                             *
 *                                                    *
 *       For fullscene data, because of the size of   *
 *       memory needed, I allocate only enough to     *
 *       grab the magnitude or the phase.  I then     *
 *       process first the magnitude and then the     *
 *       phase using the same buffer pointer....      *
 *                                                    *
 *       The code will read & write out ONLY the MAG  *
 *       image data if so specified by the user...    *
 ******************************************************/

/* Open output file for writing... */
RAWfp = fopen(RAWname, "wb");
if (RAWfp == NULL)
   {
    fprintf(stderr,"Error: unable to open [%s] for writing!\n",
                   RAWname);
    fclose(MSTARfp);
    exit(1);
   }

switch (mstartype)
  {
   case CHIP_IMAGE:
     switch (outOpt)
       {
        case ALL_DATA: 
          totchunks = nchunks * 2;
          bytesPerImage = totchunks * sizeof(float);
          CHIPdata = (float *) malloc(bytesPerImage);
          break;
        case MAG_DATA:
          totchunks = nchunks;
          bytesPerImage = totchunks * sizeof(float);
          CHIPdata = (float *) malloc(bytesPerImage);
          break; 
       } /* End of 'outOpt' switch for CHIP_IMAGE */

     if (CHIPdata == (float *) NULL)
        {
         fprintf(stderr,
                 "Error: Unable to malloc CHIP memory!\n");
         fclose(MSTARfp);
         fclose(RAWfp);
         unlink(RAWname); /* Delete output file */
         exit(1);
        }
     
     
     if (outOpt == ALL_DATA)
        {
         printf("Reading & writing ALL MSTAR chip data to [%s]\n", RAWname);
        } else
             {
              printf("Reading & writing MAG MSTAR chip data to [%s]\n", RAWname);
             }

     switch (byteorder)
       {
        case LSB_FIRST: /* Little-endian..do byteswap */

          printf("Performing auto-byteswap...\n"); 
          for (i = 0; i < totchunks; i++)
              {
               fread(bigfloatbuf, sizeof(char), 4, MSTARfp);
               littlefloatval = byteswap_SR_IR(bigfloatbuf);
               CHIPdata[i] = littlefloatval;
              }
          break; 

        case MSB_FIRST: /* Big-endian..no swap */

          numgot = fread(CHIPdata, sizeof(float), totchunks, MSTARfp);
          break;
       }

     /* Writes ALL data or MAG only data based on totchunks */
     n = fwrite(CHIPdata, sizeof(float), totchunks, RAWfp);     
     if (n != totchunks)
        {
         fprintf(stderr, "Error: in writing MSTAR Chip data!");
         fclose(MSTARfp);
         fclose(RAWfp);
         unlink(RAWname);  /* Delete output file */
         exit(1);
        }

     /* Cleanup: Close file..free memory */
     free(CHIPdata);
     break; /* End of CHIP_IMAGE case */

   case FSCENE_IMAGE:  
     bytesPerImage = nchunks * sizeof(short);
     FSCENEdata = (unsigned short *) malloc( bytesPerImage );
     if (FSCENEdata == (unsigned short *) NULL)
        {
         fprintf(stderr,
                 "Error: Unable to malloc fullscene memory!\n");
         fclose(MSTARfp);
         exit(1);
        }

     switch(outOpt)
       {
        case ALL_DATA:
          printf("Reading MSTAR fullscene magnitude data from [%s]\n", MSTARname);

          switch (byteorder)
           {
            case LSB_FIRST: /* Little-endian..do byteswap */
              printf("Performing auto-byteswap...\n"); 
              for (i = 0; i < nchunks; i++)
                  {
                   fread(bigushortbuf, sizeof(char), 2, MSTARfp);
                   littleushortval = byteswap_SUS_IUS(bigushortbuf);
                   FSCENEdata[i] = littleushortval;
                  }
              break; 
  
            case MSB_FIRST: /* Big-endian..no swap */
              numgot = fread(FSCENEdata, sizeof(short), nchunks, MSTARfp);
              break;
           }

          printf("Writing MSTAR fullscene magnitude data to [%s]\n", RAWname);
          n = fwrite(FSCENEdata, sizeof(short), nchunks, RAWfp);     
          if (n != nchunks)
             {
              fprintf(stderr, "Error: in writing MSTAR Fullscene data!");
              fclose(MSTARfp);
              fclose(RAWfp);
              unlink(RAWname);  /* Delete output file */
              exit(1);
             }

          printf("Reading MSTAR fullscene phase data from [%s]\n", MSTARname);

          switch (byteorder)
           {
            case LSB_FIRST: /* Little-endian..do byteswap */
              printf("Performing auto-byteswap...\n"); 
              for (i = 0; i < nchunks; i++)
                  {
                   fread(bigushortbuf, sizeof(char), 2, MSTARfp);
                   littleushortval = byteswap_SUS_IUS(bigushortbuf);
                   FSCENEdata[i] = littleushortval;
                  }
              break; 
  
            case MSB_FIRST: /* Big-endian..no swap */
              numgot = fread(FSCENEdata, sizeof(short), nchunks, MSTARfp);
              break;
           }

          printf("Writing MSTAR fullscene phase data to [%s]\n", RAWname);
          n = fwrite(FSCENEdata, sizeof(short), nchunks, RAWfp);     
          if (n != nchunks)
             {
              fprintf(stderr, 
                      "Error: in writing MSTAR Fullscene Phase data!");
              fclose(MSTARfp);
              fclose(RAWfp);
              unlink(RAWname);  /* Delete output file */
              exit(1);
             }

          /* Cleanup: free memory */
          free(FSCENEdata);
          break; /* End of ALL_DATA case */

        case MAG_DATA:

          printf("Reading MSTAR fullscene magnitude data from [%s]\n", MSTARname);

          switch (byteorder)
           {
            case LSB_FIRST: /* Little-endian..do byteswap */
              printf("Performing auto-byteswap...\n"); 
              for (i = 0; i < nchunks; i++)
                  {
                   fread(bigushortbuf, sizeof(char), 2, MSTARfp);
                   littleushortval = byteswap_SUS_IUS(bigushortbuf);
                   FSCENEdata[i] = littleushortval;
                  }
              break; 
  
            case MSB_FIRST: /* Big-endian..no swap */
              numgot = fread(FSCENEdata, sizeof(short), nchunks, MSTARfp);
              break;
           }

          printf("Writing MSTAR fullscene magnitude data to [%s]\n", RAWname);
          n = fwrite(FSCENEdata, sizeof(short), nchunks, RAWfp);     
          if (n != nchunks)
             {
              fprintf(stderr, "Error: in writing MSTAR Fullscene data!");
              fclose(MSTARfp);
              fclose(RAWfp);
              unlink(RAWname);  /* Delete output file */
              exit(1);
             }

          /* Cleanup: free memory */
          free(FSCENEdata);
          break; /* End of MAG_DATA case */

       } /* End of 'outOpt' switch for FSCENE_IMAGE */

     break; /* End of FSCENE_IMAGE case */

  } /* End of 'mstartype' switch */


/* Cleanup: close files */
fclose(MSTARfp);
fclose(RAWfp);

printf("\nmstar2raw conversion: completed!\n\n");

exit(0);
}


/****************************** STATIC FUNCTIONS ******************************/


/************************************************
 * Function:    byteswap_SR_IR                  *
 *   Author:    Dave Hascher (Veridian Inc.)    *
 *     Date:    06/05/97                        *
 *    Email:    dhascher@dytn.veridian.com      *
 ************************************************
 * 'SR' --> Sun 32-bit float value              *
 * 'IR' --> PC-Intel 32-bit float value         *
 ************************************************/

static float byteswap_SR_IR(pointer)
unsigned char *pointer;
{
  float *temp;
  unsigned char iarray[4], *charptr;

  iarray[0] = *(pointer + 3);
  iarray[1] = *(pointer + 2);
  iarray[2] = *(pointer + 1);
  iarray[3] = *(pointer );
  charptr = iarray;
  temp    = (float *) charptr;
  return *(temp);
}


/************************************************
 * Function:    byteswap_SUS_IUS                *
 *   Author:    John Querns (Veridian Inc.)     *
 *     Date:    06/05/97                        *
 *    Email:    jquerns@dytn.veridian.com       *
 ************************************************
 * 'SUS' --> Sun 16-bit uns short value         *
 * 'IUS' --> PC-Intel 16-bit uns short value    *
 ************************************************/

static unsigned short byteswap_SUS_IUS(pointer)
unsigned char *pointer;
{
  unsigned short *temp;
  unsigned char iarray[2], *charptr;

  iarray[0] = *(pointer + 1);
  iarray[1] = *(pointer );
  charptr = iarray;
  temp    = (unsigned short *) charptr;
  return *(temp);
}



/**********************************
 *   checkByteOrder()             *
 **********************************
 * Taken from:                    *
 *                                *
 *   Encyclopedia of Graphic File *
 *   Formats, Murray & Van Ryper, *
 *   O'Reilly & Associates, 1994, *
 *   pp. 114-115.                 *
 *                                *
 * Desc: Checks byte-order of CPU.*
 **********************************/

static int CheckByteOrder(void)

{
  short   w = 0x0001;
  char   *b = (char *) &w;

  return(b[0] ? LSB_FIRST : MSB_FIRST);
}

/************************** LAST LINE of mstar2raw.c **************************/
