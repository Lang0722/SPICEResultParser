* monte carlo
.param rval=agauss(1k,0.1k,3)
vin in 0 pulse(0 1 0 1n 1n 5n 10n)
r1 in out rval
c1 out 0 1p
.option post=1 post_version=9601
.tran 0.1n 20n sweep monte=3
.meas tran vmax max v(out)
.meas tran tr trig v(out) val=0.1 rise=1 targ v(out) val=0.9 rise=1
.end
