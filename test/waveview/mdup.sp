* measure name edge cases
.param vmax=1 rval=1k
vin in 0 pulse(0 1 0 1n 1n 5n 10n)
r1 in out rval
c1 out 0 1p
.option post=1 post_version=9601
.tran 0.1n 20n sweep data=tab
.data tab rval vmax
+ 1k 1
+ 2k 2
.enddata
.meas tran vmax max v(out)
.meas tran a_very_long_measure_name_number_one max v(out)
.meas tran a_very_long_measure_name_number_two min v(out)
.meas tran temper avg v(out)
.meas tran index avg v(in)
.meas tran d$x avg v(in)
.end
