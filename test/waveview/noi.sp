* noise
vin in 0 ac 1
r1 in out 1k
c1 out 0 1p
.option post=1 post_version=9601
.ac dec 2 1k 1g
.noise v(out) vin 1
.end
