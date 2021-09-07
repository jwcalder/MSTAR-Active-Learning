#*----------------------------------------------------------- 
#*
#* Makefile: mstar2raw.mk
#*     Date: 25 September98
#* 
#*   Author: John F. Querns, Veridian-Veda 
#* 
#*  To make: make -f mstar2raw.mk 
#* 
#*---------------------------------------------------------- 
 
BIN = mstar2raw

CC  = gcc 

CFLAGS = 

CLIBS = 

OBJECTS = mstar2raw.o 

mstar2raw: $(OBJECTS) 
	${CC} ${CFLAGS} -o mstar2raw $(OBJECTS) $(CLIBS)
