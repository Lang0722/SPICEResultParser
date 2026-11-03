* post=3: two pwl nodes with the same number of breakpoints at different times
va a 0 pwl(0 0 1n 0 2n 1 4n 1)
vb b 0 pwl(0 0 1.5n 0 2.5n 1 4n 1)
ra a 0 1k
rb b 0 1k
.option post=3
.tran 0.1n 4n
.end
