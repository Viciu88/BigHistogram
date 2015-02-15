How to use:
	program supports up to 49000 bins (on cuda.iti.pk.edu.pl devices GT480)

To compile: 
	make
	
To run:
	./histogram
	./histogram -device=devID
	./histogram -bin binCount 
	
	e.g: ./histogram -device=1 -bin 48000