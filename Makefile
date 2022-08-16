CC=gcc
CFLAGS=-lm -lgsl -lgslcblas -fopenmp -O3

%: %.c
		$(CC) -o $@ $^ $(CFLAGS)
