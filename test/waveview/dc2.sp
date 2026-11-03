* temperature sweep and 2-source dc
vin in 0 pulse(0 1 0 1n 1n 5n 10n)
vdd vdd 0 1
r1 in out 1k
r2 vdd out 1k
c1 out 0 1p
.option post=1 post_version=9601
.tran 0.1n 20n sweep temp poi 2 25 85
.dc vin 0 1 0.5 vdd 1 2 0.5
.end
