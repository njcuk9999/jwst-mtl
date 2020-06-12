#Name of C compiler
CCMPL = gcc
#Name of Fortran compiler
FPATH = 
F90 = $(FPATH)ifort
F77 = $(FPATH)ifort
#compiling object file flags
OPT1 = -O3 
#OPT1 = -O0 -g -CB
#OPT1 = -O3
OPT2 =
FFLAGS = -c $(OPT1) $(OPT2)
#FFLAGS = -c -O0 -g -CB -warn $(OPT2)
#linking flags
LFLAGS = $(OPT1) $(OPT2)
#LFLAGS = -O0 -g -CB -warn $(OPT2)
#testing flags
#LFLAGS = -O0 -g -CB
#fitsio libraries
FITSIODIR = /usr/local/lib
#Pgplot plot libraries
PGPLOTDIR = /usr/local/lib
#X11 libraries
X11DIR = /usr/X11R6/lib
# libraries for linking PGPLOT
PLPLOTDIR = -I/usr/local/Cellar/plplot/5.9.11/lib/fortran/modules/plplot -I/usr/local/Cellar/plplot/5.9.11/include/plplot -L/usr/local/Cellar/plplot/5.9.11/lib
LIBS = $(PLPLOTDIR) -lplplotf95d -lplplotf95cd
#LIBS = -L$(PGPLOTDIR) -L$(X11DIR) -lX11 -lpgplot -lpng
# libraries for linking CFITSIO
LIBS2 = -L$(PGPLOTDIR) -L$(X11DIR) -L$(FITSIODIR) -lX11 -lpgplot -lcfitsio -lpng -lgfortran
#Directory where executable are placed
BIN = /data/JWST/bin/
#utils source directory
UTILS = utils/

#Listing of programs to create.
all: spgen

spgenV2incl = precision.o response.o getfilename.o writefits_phdu.o closefits.o\
              deletefile.o ran2.o readkernels.o getfits.o readresponse.o\
              spline.o readheader.o readmodel.o
spgenV2: spgen_v2.f90 $(spgenV2incl)
	$(F90) $(LFLAGS) -o $(BIN)$@ spgen_v2.f90 $(spgenV2incl) $(LIBS2) -lfftw3

gentimeseriesincl = precision.o ran2.o
gentimeseries: gentimeseries.f90 $(gentimeseriesincl)
	$(F90) $(LFLAGS) -o $(BIN)$@ gentimeseries.f90 $(gentimeseriesincl)

makecubeincl = precision.o getfits.o writedatacube.o deletefile.o
makecube: makecube.f90 $(makecubeincl)
	$(F90) $(LFLAGS) -o $(BIN)$@ makecube.f90 $(makecubeincl) $(LIBS2)

rescaleincl = precision.o
rescale: rescale.f90 $(rescaleincl)
	$(F90) $(LFLAGS) -o $(BIN)$@ rescale.f90 $(rescaleincl) $(LIBS2)

fftwexample: fftwexample.f90 
	$(F90) $(LFLAGS) -o $(BIN)$@ fftwexample.f90 -L$(PGPLOTDIR) -lfftw3

genoskernelincl = precision.o readkernels.o getfits.o spline.o writefits.o deletefile.o
genoskernel: genoskernel.f90 $(genoskernelincl)
	$(F90) $(LFLAGS) -o $(BIN)$@ genoskernel.f90 $(genoskernelincl) $(LIBS2)

spgenincl = precision.o response.o readmodel.o writefits.o deletefile.o binmodel.o rqsort.o addflux2pix.o readkernels.o getfits.o genkernel.o convolveft.o ovrwrt.o readresponse.o readheader.o addgnoise.o ran2.o tmodel.o mandelagol.o readpmodel.o spline.o addshotnoise.o
spgen: spgen.f90 $(spgenincl)
	$(F90) $(LFLAGS) -o $(BIN)$@ spgen.f90 $(spgenincl) $(LIBS2) -lfftw3


#building object libraries
%_mp.o : $(UTILS)%.f90
	$(F90) $(FFLAGS) $(LMP) -o $@ $(UTILS)$*.f90

%_mp.o : $(UTILS)%.f
	$(F77) $(FFLAGS) $(LMP) -o $@ $(UTILS)$*.f

%.o : $(UTILS)%.f90
	$(F90) $(FFLAGS) -o $@ $<

%.o : $(UTILS)%.f
	$(F77) $(FFLAGS) -o $@ $<

# Removing object files
.PHONY : clean
clean :
	rm *.o
	rm *.mod
