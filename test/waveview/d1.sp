* data sweep one param
.param rval=1k
vin in 0 pulse(0 1 0 1n 1n 5u 10u)
r1 in out rval
c1 out 0 1n
.option post=1 post_version=9601
.tran 10n 5u sweep data=tab
.data tab rval
+ 1k
+ 2k
.enddata
.meas tran vmax max v(out)
.end
