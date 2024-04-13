#----------------------------------------

NVCC    = nvcc
NVCCFLAGS = 

#----------------------------------------


out: p2.cu
	${NVCC} ${NVCCFLAGS} -o  out p2.cu

.PHONY: clean

clean:
	rm -f out
