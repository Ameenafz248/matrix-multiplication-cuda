#----------------------------------------

NVCC    = nvcc
NVCCFLAGS = 

#----------------------------------------


out: prog.cu
	${NVCC} ${NVCCFLAGS} -o  out prog.cu

.PHONY: clean

clean:
	rm -f out
