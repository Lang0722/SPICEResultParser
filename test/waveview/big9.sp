* values of 1e33 to 1e36: WaveView ends its scan early on some of them
vin in 0 pulse(0 1 0 1n 1n 5n 10n)
r1 in 0 1k
e1 g33 0 in 0 1e33
e2 g34 0 in 0 1e34
e3 h34 0 in 0 5e33
e4 g36 0 in 0 1e36
.option post=1 post_version=9601
.tran 1n 20n
.end
