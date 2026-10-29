* single run nonsweep
vin in 0 sin(0 1 1meg) ac 1
r1 in out 1k
c1 out 0 1n
.option post=1
.tran 10n 3u
.ac dec 5 1k 10meg
.dc vin 0 1 0.25
.end
