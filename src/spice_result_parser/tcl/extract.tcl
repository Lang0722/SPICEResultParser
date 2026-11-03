# extract.tcl: write the points of selected signals of one waveform file: every
# point, or with SRP_XLO and SRP_XHI the points of an x window (WaveView batch mode).
#
# Part of spice_result_parser (waveview.py runs it). Run a copy of it by hand in
# a working directory (WaveView writes ace_tcl.log beside the script):
#   Linux:   SRP_IN=run.tr0 SRP_OUT=out.txt wv -ace_no_gui extract.tcl
#   Windows: set SRP_IN=run.tr0&& set SRP_OUT=out.txt&& wv.exe -ace_no_gui extract.tcl
#
# The points come from a POINT scan, not from sx_export_data: the export drops
# points that repeat an x value (frequent in float32 .tr0 files) and prints only
# 16 digits. The scan gives every point of the file with 17 digits.
#
# Input:  SRP_IN      the waveform file.
#         SRP_SIGNALS optional: a text file with one exact signal name per line,
#                     as list_traces.tcl writes them. Not set: every signal.
#                     No names in it: x only.
#         SRP_XCHECK  optional: set, the script checks that all signals have
#                     the same x points (see write_points).
#         SRP_NPOINTS optional: "all" writes the npoints record of every signal,
#                     not only of the x signal and the written signals.
#         SRP_MAX_VALUES optional: the most values that the script writes. An x
#                     or real value is 1, a complex value 2. The x scan counts each
#                     point as its whole row: x and the value of each written signal.
#         SRP_XLO, SRP_XHI optional: the window [SRP_XLO, SRP_XHI] of x. One of
#                     them alone leaves the other side open. When the x unit of the
#                     x signal is s or Hz, the scans write the window and at most 8
#                     points before it. Else they write every point.
# Output: SRP_OUT     a text file. Records have a tab; value lines do not:
#   srp_extract 1
#   trace   <i>           0-based; one per sweep trace, one when not swept
#   span    <first> <last> with the window: the first and the last x of the trace
#                         of the x signal
#   x       <signal>      then the x value of each point, one per line
#   real    <signal>      then the y value of each point, one per line
#   complex <signal>      then "<real part> <imaginary part>" per point
#   ...                   (the signals in the order of the file)
#   stopped <signal>      with the window, after the values of a scan that ends
#                         before the last x of the trace: WaveView stops early on
#                         some values of about 1e34 or more
#   limit   <values> <trace> <x>
#                         in place of the rest, when the next row would go past
#                         SRP_MAX_VALUES: the values of the rows that fit, the trace,
#                         and the x of the last row that fits in it (empty: none)
#   npoints <signal> <n>  after the last trace: WaveView's point count of the x
#                         signal and of each written signal, in the order of the
#                         file, each once (in a swept file one more per sweep)
#   srp_done
# NaN prints as 1.#QNAN on Windows. A file that does not open, a signal name
# that the file does not have, or signals with their own x points (an HSPICE
# post=3 file) give "error<TAB><text>" then srp_done. A missing srp_done means
# that the script failed: the console (stdout) has the WaveView error. The exit
# code does not show success: it is 0 after an error that the script catches.
#
# Tcl 8.4: no dict, lassign, {*} or "in".

set tcl_precision 17
set out [open $env(SRP_OUT) w]
fconfigure $out -encoding utf-8 -buffersize 1048576

proc finish {{error ""}} {
    global out
    if {$error != ""} {
        puts $out "error\t$error"
    }
    puts $out srp_done
    close $out
    exit
}

# A name as valid UTF-8. WaveView gives the bytes of the file: a Latin-1 byte
# would not equal the name in SRP_SIGNALS, and puts can lose the end of the line.
proc clean {name} {
    return [encoding convertfrom utf-8 [encoding convertto utf-8 $name]]
}

# sx_jump_to_x on trace `trace` of a swept file, or on the one trace.
proc jump {scan target trace} {
    global swept
    if {$swept} {
        return [sx_jump_to_x $scan $target -trace $trace]
    }
    return [sx_jump_to_x $scan $target]
}

# Put scan on a real point of the trace at or before its first point with x >= lo.
# Return 1, or 0 when the trace has no point. A bound within the first 8 points starts at
# FIRST. Else jump to lo, then go back with PREV to the first x below lo. PREV can leave
# the trace (it goes to the end of the trace before): an x that grows on the way back
# shows that, and the scan starts at FIRST. (After a jump to an x, sx_x_value gives that
# x, not the x of a point, and NEXT can skip a point: the rules that WaveView showed are in
# docs/superpowers/specs/2026-11-03-waveview-windowed-extract-design.md.)
proc seek {scan lo trace} {
    if {![jump $scan FIRST $trace]} {
        return 0
    }
    for {set i 0} {$i < 8} {incr i} {
        if {[sx_x_value $scan] >= $lo} {
            return [jump $scan FIRST $trace]
        }
        if {![sx_jump_to_x $scan NEXT]} {
            return [jump $scan FIRST $trace]
        }
    }
    jump $scan $lo $trace
    set last ""
    while {[sx_jump_to_x $scan PREV]} {
        set x [sx_x_value $scan]
        if {$last != "" && $x > $last} {
            break
        }
        if {$x < $lo} {
            return 1
        }
        set last $x
    }
    return [jump $scan FIRST $trace]
}

# Write "span<TAB><first x><TAB><last x>" of trace `trace` of the x signal, and keep
# the last x in span_last. The LAST x of the last sweep of a swept HSPICE file is its
# end marker (1e30 or more): the point before it is the last real one.
proc write_span {sig trace} {
    global out span_last
    set scan [sx_scan_data_points $sig POINT]
    set first ""
    set span_last ""
    if {[jump $scan FIRST $trace]} {
        set first [sx_x_value $scan]
        if {[jump $scan LAST $trace]} {
            set span_last [sx_x_value $scan]
            if {$span_last >= 1e30 && [sx_jump_to_x $scan PREV]} {
                set span_last [sx_x_value $scan]
            }
        }
    }
    sx_release $scan
    puts $out "span\t$first\t$span_last"
}

# Write one value line per point of trace `trace` of signal `sig`. With the window,
# the scan starts at seek and stops before the first x above SRP_XHI, and a scan that
# ends before the last x of the trace writes "stopped<TAB><signal>". With
# SRP_XCHECK, the x scan keeps its values in xs, and a y scan compares its x
# values with them: all the signals of the trace must have the same x points.
# The check costs time, so waveview.py sets SRP_XCHECK only for a file whose
# format does not give one x to all signals. With SRP_MAX_VALUES, the x scan
# counts the values of each row (see stop_at_limit). The loops are in a proc so
# that Tcl compiles them.
proc write_points {sig trace kind} {
    global out xs xname xcheck limit count width windowed xlo xhi span_last nameof
    set scan [sx_scan_data_points $sig POINT]
    if {$windowed && [string length $xlo]} {
        set more [seek $scan $xlo $trace]
    } else {
        set more [jump $scan FIRST $trace]
    }
    set high [expr {$windowed && [string length $xhi]}]
    set k 0
    set n [llength $xs]
    set last ""
    set other 0
    if {$kind == "x"} {
        set xs {}
        while {$more == 1} {
            set x [sx_x_value $scan]
            if {$high && $x > $xhi} {
                break
            }
            if {$limit >= 0 && $count + $width > $limit} {
                stop_at_limit $trace $last
            }
            incr count $width
            if {$xcheck} {
                lappend xs $x
            }
            puts $out $x
            set last $x
            set more [sx_jump_to_x $scan NEXT]
        }
    } else {
        set complex [string equal $kind complex]
        set look [expr {$windowed || $n > 0}]
        while {$more == 1} {
            if {$look} {
                set x [sx_x_value $scan]
                if {$high && $x > $xhi} {
                    break
                }
                if {$k < $n && $x != [lindex $xs $k]} {
                    set other 1
                    break
                }
                set last $x
            }
            if {$complex} {
                puts $out "[sx_y_value $scan real] [sx_y_value $scan imag]"
            } else {
                puts $out [sx_y_value $scan]
            }
            incr k
            set more [sx_jump_to_x $scan NEXT]
        }
    }
    sx_release $scan
    if {$other} {
        finish "the signals have their own x points (an HSPICE post=3 file, for example):\
            [clean [sx_get_name $sig]] has other x points than $xname; extract one signal at a time"
    }
    if {$windowed && $more != 1 && [string length $last] && [string length $span_last] && $last < $span_last} {
        puts $out "stopped\t$nameof($sig)"
    }
}

# Before a row would go past SRP_MAX_VALUES: write the values of the rows that fit,
# the trace and the x of its last row that fits (empty when none), and end. The
# signals have the rows of the x scan, so their scans do not count.
proc stop_at_limit {trace last} {
    global out count
    puts $out "limit\t$count\t$trace\t$last"
    finish
}

# sx_open_sim_file_read splits its argument as a Tcl list: pass one element, in
# quotes when the path has a brace ([list] escapes a lone brace with a backslash,
# which WaveView changes to "/"). It cannot open a path with non-ASCII characters
# or a Windows path of 260 characters or more: try the 8.3 short name.
set path $env(SRP_IN)
if {![string is ascii $path] || [string length $path] >= 260} {
    catch {set path [file attributes $path -shortname]}
}
set arg [list $path]
if {[regexp {[{}]} $path] && [string first "\"" $path] < 0} {
    set arg "\"$path\""
}
set files {}
catch {set files [sx_open_sim_file_read $arg]}
set f [lindex $files 0]
set why ""
if {![string is ascii $path]} {
    set why ": WaveView cannot open a path with non-ASCII characters"
} elseif {[string length $path] >= 260} {
    set why ": WaveView cannot open a path of 260 characters or more"
}
if {$f == ""} {
    finish "cannot open $env(SRP_IN)$why"
}
sx_current_sim_file $f
set all [sx_signal *]
set order {}
foreach s $all {
    set name [clean [sx_get_name $s]]
    set object($name) $s
    set nameof($s) $name
    lappend order $name
}
if {[llength $all] == 0} {
    finish "the file has no signals"
}

if {[info exists env(SRP_SIGNALS)]} {
    set in [open $env(SRP_SIGNALS) r]
    fconfigure $in -encoding utf-8
    while {[gets $in line] >= 0} {
        set name [string trimright $line "\r"]
        if {$name == ""} {
            continue
        }
        if {![info exists object($name)]} {
            close $in
            finish "unknown signal $name"
        }
        set wanted($name) 1
    }
    close $in
} else {
    foreach name $order {
        set wanted($name) 1
    }
}
set sigs {}
foreach name $order {
    if {[info exists wanted($name)]} {
        lappend sigs $object($name)
    }
}

set ntrace [llength [sx_get_sweep_conditions $f]]
set swept [expr {$ntrace > 0}]
if {!$swept} {
    set ntrace 1
}
set xsig [lindex $all 0]
if {[llength $sigs] > 0} {
    set xsig [lindex $sigs 0]
}
set xname $nameof($xsig)
set xs {}
set xcheck [info exists env(SRP_XCHECK)]
set limit -1
if {[info exists env(SRP_MAX_VALUES)]} {
    set limit $env(SRP_MAX_VALUES)
}
set xlo ""
set xhi ""
if {[info exists env(SRP_XLO)]} {
    set xlo $env(SRP_XLO)
}
if {[info exists env(SRP_XHI)]} {
    set xhi $env(SRP_XHI)
}
# The window applies to a time or frequency x only: such x values grow within a
# trace, and seek needs that. Any other x (a DC sweep can go down) gets every point.
set windowed 0
if {([string length $xlo] || [string length $xhi])
        && [lsearch -exact {s Hz} [sx_signal_attribute $xsig -xunit]] >= 0} {
    set windowed 1
}
set span_last ""
set count 0
set width 1
foreach s $sigs {
    if {[sx_signal_attribute $s -datatype] == "complex"} {
        incr width 2
    } else {
        incr width
    }
}
puts $out "srp_extract\t1"
for {set i 0} {$i < $ntrace} {incr i} {
    puts $out "trace\t$i"
    if {$windowed} {
        write_span $xsig $i
    }
    puts $out "x\t$xname"
    write_points $xsig $i x
    foreach s $sigs {
        set kind [sx_signal_attribute $s -datatype]
        if {$kind != "complex"} {
            set kind real
        }
        puts $out "$kind\t$nameof($s)"
        write_points $s $i $kind
    }
}
# The scans loaded these signals, so their counts cost no more loads. The counts of
# all signals cost a load of each signal that the export did not read.
set every [expr {[info exists env(SRP_NPOINTS)] && [string equal $env(SRP_NPOINTS) all]}]
foreach name $order {
    if {$every || [string equal $name $xname] || [info exists wanted($name)]} {
        puts $out "npoints\t$name\t[sx_signal_attribute $object($name) -npoints]"
    }
}
finish
